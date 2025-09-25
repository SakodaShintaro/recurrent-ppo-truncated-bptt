import numpy as np
import torch
from gymnasium import spaces


class Buffer:
    """The buffer stores and prepares the training data. It supports recurrent policies."""

    def __init__(
        self,
        worker_steps: int,
        hidden_size: int,
        layer_type: str,
        sequence_length: int,
        observation_space: spaces.Box,
        action_space_shape: tuple,
        device: torch.device,
    ) -> None:
        """
        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {spaces.Box} -- The observation space of the agent
            action_space_shape {tuple} -- Shape of the action space
            device {torch.device} -- The device that will be used for training
        """
        # Setup members
        self.device = device
        self.worker_steps = worker_steps
        self.n_mini_batches = 8
        self.batch_size = self.worker_steps
        self.layer_type = layer_type
        self.sequence_length = sequence_length
        self.true_sequence_length = 0

        # Initialize the buffer's data storage for a single environment
        self.rewards = np.zeros(self.worker_steps, dtype=np.float32)
        self.actions = torch.zeros((self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros(self.worker_steps, dtype=bool)
        self.obs = torch.zeros((self.worker_steps,) + observation_space.shape)
        self.hxs = torch.zeros((self.worker_steps, hidden_size))
        self.cxs = torch.zeros((self.worker_steps, hidden_size))
        self.log_probs = torch.zeros((self.worker_steps, len(action_space_shape)))
        self.values = torch.zeros(self.worker_steps)
        self.advantages = torch.zeros(self.worker_steps)

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        samples = {
            "obs": self.obs,
            "actions": self.actions,
            "loss_mask": torch.ones(self.worker_steps, dtype=torch.bool),
        }

        samples["hxs"] = self.hxs
        if self.layer_type == "lstm":
            samples["cxs"] = self.cxs

        # Determine indices at which episodes terminate
        episode_done_indices = list(np.where(self.dones)[0])
        if not episode_done_indices or episode_done_indices[-1] != self.worker_steps - 1:
            episode_done_indices.append(self.worker_steps - 1)

        index_sequences, max_sequence_length = self._arange_sequences(
            torch.arange(self.worker_steps), episode_done_indices
        )
        self.flat_sequence_indices = np.asarray(
            [seq.tolist() for seq in index_sequences], dtype=object
        )

        for key, value in samples.items():
            value_tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            sequences, _ = self._arange_sequences(value_tensor, episode_done_indices)
            sequences = [
                self._pad_sequence(sequence, max_sequence_length) for sequence in sequences
            ]
            stacked = torch.stack(sequences, dim=0)
            if key in ("hxs", "cxs"):
                stacked = stacked[:, 0]
            samples[key] = stacked

        self.num_sequences = len(self.flat_sequence_indices)
        self.actual_sequence_length = max_sequence_length
        self.true_sequence_length = max_sequence_length

        samples["values"] = self.values
        samples["log_probs"] = self.log_probs
        samples["advantages"] = self.advantages

        self.samples_flat = {}
        for key, value in samples.items():
            if key in ("hxs", "cxs"):
                self.samples_flat[key] = value
            elif key in ("values", "log_probs", "advantages"):
                self.samples_flat[key] = value
            else:
                if value.dim() == 1:
                    self.samples_flat[key] = value
                else:
                    self.samples_flat[key] = value.reshape(
                        value.shape[0] * value.shape[1], *value.shape[2:]
                    )

    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Pads a sequence to the target length using zeros.

        Arguments:
            sequence {np.ndarray} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence

        Returns:
            {torch.tensor} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = torch.zeros(
                ((delta_length,) + sequence.shape[1:]),
                dtype=sequence.dtype,
                device=sequence.device,
            )
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype, device=sequence.device)
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)

    def _arange_sequences(self, data, episode_done_indices):
        """Splits the provided data into episodes and then into sequences.
        The split points are indicated by the environments' done signals.

        Arguments:
            data {torch.tensor} -- The to be split data arrange into num_worker, worker_steps
            episode_done_indices {list} -- Nested list indicating the indices of done signals. Trajectory ends are treated as done

        Returns:
            {list} -- Data arranged into sequences of variable length as list
        """
        sequences = []
        max_length = 1
        start_index = 0
        for done_index in episode_done_indices:
            episode = data[start_index : done_index + 1]
            if self.sequence_length > 0:
                for seq_start in range(0, len(episode), self.sequence_length):
                    seq = episode[seq_start : seq_start + self.sequence_length]
                    sequences.append(seq)
                    max_length = max(max_length, len(seq))
            else:
                sequences.append(episode)
                max_length = max(max_length, len(episode))
            start_index = done_index + 1
        return sequences, max_length

    def recurrent_mini_batch_generator(self):
        """A recurrent generator that returns a dictionary containing the data of a whole minibatch.
        In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences_per_batch = self.num_sequences // self.n_mini_batches
        num_sequences_per_batch = (
            [num_sequences_per_batch] * self.n_mini_batches
        )  # Arrange a list that determines the sequence count for each mini batch
        remainder = self.num_sequences % self.n_mini_batches
        for i in range(remainder):
            num_sequences_per_batch[i] += (
                1  # Add the remainder if the sequence count and the number of mini batches do not share a common divider
            )
        # Prepare indices, but only shuffle the sequence indices and not the entire batch to ensure that sequences are maintained as a whole.
        indices = torch.arange(0, self.num_sequences * self.actual_sequence_length).reshape(
            self.num_sequences, self.actual_sequence_length
        )
        sequence_indices = torch.randperm(self.num_sequences)

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_padded_indices = indices[sequence_indices[start:end]].reshape(-1)
            # Unpadded and flat indices are used to sample unpadded training data
            mini_batch_unpadded_indices = self.flat_sequence_indices[
                sequence_indices[start:end].tolist()
            ]
            mini_batch_unpadded_indices = [
                item for sublist in mini_batch_unpadded_indices for item in sublist
            ]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "hxs" or key == "cxs":
                    # Select recurrent cell states of sequence starts
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
                elif key == "log_probs" or "advantages" in key or key == "values":
                    # Select unpadded data
                    mini_batch[key] = value[mini_batch_unpadded_indices].to(self.device)
                else:
                    # Select padded data
                    mini_batch[key] = value[mini_batch_padded_indices].to(self.device)
            start = end
            yield mini_batch

    @torch.no_grad()
    def calc_advantages(self, last_value: torch.tensor, gamma: float, td_lambda: float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state
            gamma {float} -- Discount factor
            td_lambda {float} -- GAE regularization parameter
        """
        mask = torch.logical_not(torch.from_numpy(self.dones))
        rewards = torch.from_numpy(self.rewards)
        values = self.values
        last_value = last_value.squeeze().cpu()
        last_advantage = torch.zeros_like(last_value)
        for t in reversed(range(self.worker_steps)):
            if not mask[t]:
                last_value = torch.zeros_like(last_value)
                last_advantage = torch.zeros_like(last_advantage)
            delta = rewards[t] + gamma * last_value - values[t]
            last_advantage = delta + gamma * td_lambda * last_advantage
            self.advantages[t] = last_advantage
            last_value = values[t]
