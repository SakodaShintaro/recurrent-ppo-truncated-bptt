import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim

from buffer import Buffer
from minigrid_env import Minigrid
from model import ActorCriticModel


class PPOTrainer:
    def __init__(self) -> None:
        # Set variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        self.env = Minigrid(env_name="MiniGrid-MemoryS9-v0", realtime_mode=False)
        self.observation_space = self.env.observation_space
        self.action_space_shape = (self.env.action_space.n,)

        # Init buffer
        print("Step 2: Init buffer")
        self.worker_steps = 1024
        hidden_size = 256
        self.layer_type = "lstm"
        sequence_length = 8
        self.buffer = Buffer(
            self.worker_steps,
            hidden_size,
            self.layer_type,
            sequence_length,
            self.observation_space,
            self.action_space_shape,
            self.device,
        )

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(
            hidden_size,
            self.layer_type,
            self.observation_space,
            self.action_space_shape,
        ).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2.0e-4)

        # Init environment
        print("Step 4: Init environment")
        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
        if self.layer_type == "gru":
            self.recurrent_cell = hxs
        elif self.layer_type == "lstm":
            self.recurrent_cell = (hxs, cxs)

        # Reset environment
        print("Step 5: Reset environment")
        initial_obs, _ = self.env.reset()
        self.obs = np.asarray(initial_obs, dtype=np.float32)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.save_dir = Path("results") / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 6: Starting training")
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)

        result_dict_list = []

        start = time.time()

        for update in range(2000):
            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats = self._train_epochs()
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = self._process_episode_info(episode_infos)

            elapsed_sec = time.time() - start
            elapsed_min = elapsed_sec / 60

            # Print training statistics
            result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                update,
                episode_result["reward_mean"],
                episode_result["reward_std"],
                episode_result["length_mean"],
                episode_result["length_std"],
                training_stats[0],
                training_stats[1],
                training_stats[3],
                training_stats[2],
                torch.mean(self.buffer.values),
                torch.mean(self.buffer.advantages),
            )
            print(result)
            result_dict = {
                "elapsed_min": elapsed_min,
                "update": update,
                "reward_mean": episode_result["reward_mean"],
                "reward_std": episode_result["reward_std"],
                "length_mean": episode_result["length_mean"],
                "length_std": episode_result["length_std"],
                "policy_loss": training_stats[0],
                "value_loss": training_stats[1],
                "entropy": training_stats[3],
                "loss": training_stats[2],
                "value_mean": torch.mean(self.buffer.values).item(),
                "advantage_mean": torch.mean(self.buffer.advantages).item(),
            }
            result_dict_list.append(result_dict)
            df = pd.DataFrame(result_dict_list)
            df.to_csv(self.save_dir / "result.csv", index=False)

            # Free memory
            del self.buffer.samples_flat
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _sample_training_data(self) -> list:
        """Runs the environment for the configured number of steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        # Sample actions from the model and collect experiences for training
        for t in range(self.worker_steps):
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32)
            self.buffer.obs[t] = obs_tensor.cpu()

            current_cell = self.recurrent_cell
            if self.layer_type == "gru":
                self.buffer.hxs[t] = current_cell.squeeze(0).squeeze(0).detach().cpu()
            elif self.layer_type == "lstm":
                self.buffer.hxs[t] = current_cell[0].squeeze(0).squeeze(0).detach().cpu()
                self.buffer.cxs[t] = current_cell[1].squeeze(0).squeeze(0).detach().cpu()

            # Forward the model to retrieve the policy, the states' value and the recurrent cell states
            policy, value, self.recurrent_cell = self.model(
                obs_tensor.unsqueeze(0).to(self.device), current_cell
            )
            self.buffer.values[t] = value.squeeze(0).detach().cpu()

            # Sample actions from each individual policy branch
            actions = []
            log_probs = []
            action = policy.sample()
            actions.append(action)
            log_probs.append(policy.log_prob(action))
            action_tensor = torch.stack(actions, dim=1).detach()
            log_prob_tensor = torch.stack(log_probs, dim=1).detach()
            self.buffer.actions[t] = action_tensor.squeeze(0).cpu().long()
            self.buffer.log_probs[t] = log_prob_tensor.squeeze(0).cpu()

            # Interact with the environment
            env_action = self.buffer.actions[t].cpu().numpy()
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated
            self.buffer.rewards[t] = reward
            self.buffer.dones[t] = done

            if done:
                episode_infos.append(info)
                obs, _ = self.env.reset()

            self.obs = np.asarray(obs, dtype=np.float32)

        # Calculate advantages
        last_obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        _, last_value, _ = self.model(last_obs_tensor, self.recurrent_cell)
        self.buffer.calc_advantages(last_value, gamma=0.99, td_lambda=0.95)

        return episode_infos

    def _train_epochs(self) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.

        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient

        Returns:
            {list} -- Training statistics of one training epoch"""
        train_info = []
        for _ in range(4):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch))
        return train_info

    def _train_mini_batch(self, samples: dict) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of training statistics (e.g. loss)
        """
        learning_rate = 2.0e-4
        beta = 0.001
        clip_range = 0.2

        # Retrieve sampled recurrent cell states to feed the model
        if self.layer_type == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.layer_type == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        # Forward model
        policy, value, _ = self.model(
            samples["obs"], recurrent_cell, self.buffer.actual_sequence_length
        )

        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        log_probs.append(policy.log_prob(samples["actions"][:, 0]))
        entropies.append(policy.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Remove paddings
        value = value[samples["loss_mask"]]
        log_probs = log_probs[samples["loss_mask"]]
        entropies = entropies[samples["loss_mask"]]

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (
            samples["advantages"].std() + 1e-8
        )
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(
            1, len(self.action_space_shape)
        )  # Repeat is necessary for multi-discrete action spaces
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(
            min=-clip_range, max=clip_range
        )
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - 0.25 * vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return [
            policy_loss.cpu().data.numpy(),
            vf_loss.cpu().data.numpy(),
            loss.cpu().data.numpy(),
            entropy_bonus.cpu().data.numpy(),
        ]

    @staticmethod
    def _process_episode_info(episode_info: list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.

        Arguments:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        result["reward_mean"] = np.mean([info["episode"]["r"] for info in episode_info])
        result["reward_std"] = np.std([info["episode"]["r"] for info in episode_info])
        result["length_mean"] = np.mean([info["episode"]["l"] for info in episode_info])
        result["length_std"] = np.std([info["episode"]["l"] for info in episode_info])
        assert result["reward_mean"] >= 0, "reward mean is not greater than 0"
        return result


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.run_training()
