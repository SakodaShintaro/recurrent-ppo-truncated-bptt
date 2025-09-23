import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ActorCriticModel(nn.Module):
    def __init__(self, hidden_size, layer_type, observation_space, action_space_shape):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_type = layer_type

        # Observation encoder
        # Visual encoder made of 3 convolutional layers
        self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        # Compute output size of convolutional layers
        in_features_next_layer = self.get_conv_output(observation_space.shape)

        self.lin_hidden_in = nn.Linear(in_features_next_layer, self.hidden_size)

        # Recurrent layer (GRU or LSTM)
        if self.layer_type == "gru":
            self.recurrent_layer = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.layer_type == "lstm":
            self.recurrent_layer = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        # Init recurrent layer

        # Hidden layer
        self.lin_hidden_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        assert len(action_space_shape) == 1
        self.policy = nn.Linear(in_features=self.hidden_size, out_features=action_space_shape[0])

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)

        # Apply weight initialization to all modules
        self.apply(self._init_weights)

    def forward(
        self,
        obs: torch.tensor,
        recurrent_cell: torch.tensor,
        sequence_length: int = 1,
    ):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations  (B, 3, H, W)
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        B = h.shape[0]
        # Propagate input through the visual encoder
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        # Flatten the output of the convolutional layers
        h = h.reshape((B, -1))

        h = F.relu(self.lin_hidden_in(h))

        # Forward recurrent layer (GRU or LSTM) first, then hidden layer
        # Reshape the to be fed data to batch_size, sequence_length, data
        B, D = h.shape
        h = h.reshape((B // sequence_length, sequence_length, D))

        # Forward recurrent layer
        h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

        # Reshape to the original tensor size
        B, T, D = h.shape
        h = h.reshape(B * T, D)

        # Feed hidden layer after recurrent layer
        h = F.relu(self.lin_hidden_out(h))
        memory_out = recurrent_cell

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = Categorical(logits=self.policy(h_policy))

        return pi, value, memory_out

    def _init_weights(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def get_conv_output(self, shape: tuple) -> int:
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        hxs = torch.zeros(
            (num_sequences),
            self.hidden_size,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        cxs = None
        if self.layer_type == "lstm":
            cxs = torch.zeros(
                (num_sequences),
                self.hidden_size,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        return hxs, cxs
