import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from transformer import Transformer


class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.observation_space_shape = observation_space.shape

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(
                observation_space.shape[0],
                32,
                8,
                4,
            )
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]

        # Memory layer (GRU, LSTM, or Transformer)
        if self.recurrence["layer_type"] == "transformer":
            # Transformer setup with default settings
            self.memory_layer_size = self.recurrence["hidden_state_size"]
            self.lin_hidden = nn.Linear(in_features_next_layer, self.memory_layer_size)
            nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

            # Create transformer config using existing recurrence settings
            transformer_config = {
                "num_blocks": 3,
                "embed_dim": self.memory_layer_size,
                "num_heads": 8,
                "memory_length": self.recurrence["sequence_length"],  # Use sequence_length
                "positional_encoding": "learned",
                "layer_norm": "pre",
                "gtrxl": False,
                "gtrxl_bias": 2.0
            }

            # Transformer blocks
            self.transformer = Transformer(transformer_config, self.memory_layer_size, 1000)  # max_episode_length
            memory_output_size = self.memory_layer_size
        else:
            # Recurrent layer (GRU or LSTM)
            if self.recurrence["layer_type"] == "gru":
                self.recurrent_layer = nn.GRU(
                    in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True
                )
            elif self.recurrence["layer_type"] == "lstm":
                self.recurrent_layer = nn.LSTM(
                    in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True
                )
            # Init recurrent layer
            for name, param in self.recurrent_layer.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, np.sqrt(2))

            # Hidden layer
            self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
            nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))
            memory_output_size = self.hidden_size

        # Decouple policy from value
        if self.recurrence["layer_type"] == "transformer":
            # Hidden layer of the policy (transformer)
            self.lin_policy = nn.Linear(self.memory_layer_size, self.hidden_size)
            nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

            # Hidden layer of the value function (transformer)
            self.lin_value = nn.Linear(self.memory_layer_size, self.hidden_size)
            nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))
        else:
            # Hidden layer of the policy (recurrent)
            self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

            # Hidden layer of the value function (recurrent)
            self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(
        self,
        obs: torch.tensor,
        recurrent_cell: torch.tensor = None,
        device: torch.device = None,
        sequence_length: int = 1,
        memory_mask: torch.tensor = None,
        memory_indices: torch.tensor = None,
    ):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        if self.recurrence["layer_type"] == "transformer":
            # Feed hidden layer for transformer
            h = F.relu(self.lin_hidden(h))
            # Forward transformer blocks
            # For simplicity, use dummy parameters for transformer forward
            dummy_memories = torch.zeros(h.size(0), self.recurrence["sequence_length"], 3, 256).to(h.device)
            dummy_mask = torch.ones(h.size(0), self.recurrence["sequence_length"], dtype=torch.bool).to(h.device)
            dummy_memory_indices = torch.zeros(h.size(0), self.recurrence["sequence_length"], dtype=torch.long).to(h.device)

            h, updated_memories = self.transformer(h, dummy_memories, dummy_mask, dummy_memory_indices)
            memory_out = recurrent_cell  # Keep dummy recurrent_cell for compatibility
        else:
            # Forward recurrent layer (GRU or LSTM) first, then hidden layer
            if sequence_length == 1:
                # Case: sampling training data or model optimization using sequence length == 1
                h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
                h = h.squeeze(1)  # Remove sequence length dimension
            else:
                # Case: Model optimization given a sequence length > 1
                # Reshape the to be fed data to batch_size, sequence_length, data
                h_shape = tuple(h.size())
                h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

                # Forward recurrent layer
                h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

                # Reshape to the original tensor size
                h_shape = tuple(h.size())
                h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

            # Feed hidden layer after recurrent layer
            h = F.relu(self.lin_hidden(h))
            memory_out = recurrent_cell

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]

        return pi, value, memory_out

    def get_conv_output(self, shape: tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        if self.recurrence["layer_type"] == "transformer":
            # For transformer, return None as memory is handled differently
            return None, None

        hxs = torch.zeros(
            (num_sequences),
            self.recurrence["hidden_state_size"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        cxs = None
        if self.recurrence["layer_type"] == "lstm":
            cxs = torch.zeros(
                (num_sequences),
                self.recurrence["hidden_state_size"],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        return hxs, cxs
