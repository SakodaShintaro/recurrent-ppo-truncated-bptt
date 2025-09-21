import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from buffer import Buffer
from minigrid_env import Minigrid
from model import ActorCriticModel


def polynomial_decay(
    initial: float, final: float, max_decay_steps: int, power: float, current_step: int
) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly.

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return (initial - final) * ((1 - current_step / max_decay_steps) ** power) + final


class PPOTrainer:
    def __init__(
        self, config: dict, run_id: str = "run", device: torch.device = torch.device("cpu")
    ) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.device = device
        self.run_id = run_id

        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        self.env = Minigrid(env_name="MiniGrid-MemoryS9-v0", realtime_mode=False)
        self.observation_space = self.env.observation_space
        self.action_space_shape = (self.env.action_space.n,)

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(
            self.config, self.observation_space, self.action_space_shape, self.device
        )

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(
            self.config, self.observation_space, self.action_space_shape
        ).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2.0e-4)

        # Init environment
        print("Step 4: Init environment")
        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_cell = hxs
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_cell = (hxs, cxs)

        # Reset environment
        print("Step 5: Reset environment")
        initial_obs = self.env.reset()
        self.obs = np.asarray(initial_obs, dtype=np.float32)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print(timestamp)
        self.save_dir = Path("results") / timestamp
        print(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 6: Starting training")
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)

        result_dict_list = []

        start = time.time()

        for update in range(2000):
            # Decay hyperparameters
            learning_rate = polynomial_decay(2.0e-4, 2.0e-4, 300, 1.0, update)
            beta = polynomial_decay(0.001, 0.001, 300, 1.0, update)
            clip_range = polynomial_decay(0.2, 0.2, 300, 1.0, update)

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats = self._train_epochs(learning_rate, clip_range, beta)
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

            # Write training statistics to tensorboard
            self._write_training_summary(update, training_stats, episode_result)

            # Free memory
            del self.buffer.samples_flat
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _sample_training_data(self) -> list:
        """Runs the environment for the configured number of steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        # Sample actions from the model and collect experiences for training
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                obs_tensor = torch.tensor(self.obs, dtype=torch.float32)
                self.buffer.obs[t] = obs_tensor.cpu()

                current_cell = self.recurrent_cell
                if self.recurrence["layer_type"] == "gru":
                    self.buffer.hxs[t] = current_cell.squeeze(0).squeeze(0).detach().cpu()
                elif self.recurrence["layer_type"] == "lstm":
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
            obs, reward, done, info = self.env.step(env_action)
            self.buffer.rewards[t] = reward
            self.buffer.dones[t] = done

            if info:
                episode_infos.append(info)
                obs = self.env.reset()

            self.obs = np.asarray(obs, dtype=np.float32)

        # Calculate advantages
        last_obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        _, last_value, _ = self.model(last_obs_tensor, self.recurrent_cell)
        self.buffer.calc_advantages(last_value, gamma=0.99, td_lambda=0.95)

        return episode_infos

    def _train_epochs(self, learning_rate: float, clip_range: float, beta: float) -> list:
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
                train_info.append(
                    self._train_mini_batch(mini_batch, learning_rate, clip_range, beta)
                )
        return train_info

    def _train_mini_batch(
        self, samples: dict, learning_rate: float, clip_range: float, beta: float
    ) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of training statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
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

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[3], update)
        self.writer.add_scalar("training/sequence_length", self.buffer.true_sequence_length, update)
        self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar(
            "training/advantage_mean", torch.mean(self.buffer.advantages), update
        )

    @staticmethod
    def _process_episode_info(episode_info: list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.

        Arguments:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == "success":
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        time.sleep(1.0)
        exit(0)


def _load_config(path: str) -> dict:
    """Load the YAML config file and return its contents as a dict."""
    yaml = YAML()
    with open(path, "r", encoding="utf-8") as stream:
        config = {}
        for data in yaml.load_all(stream):
            if data:
                config = dict(data)
    if not config:
        raise ValueError(f"Config file '{path}' did not contain any data.")
    return config


def main():
    run_id = "run"
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = _load_config("./minigrid.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    # Initialize the PPO trainer and commence training
    trainer = PPOTrainer(config, run_id=run_id, device=device)
    trainer.run_training()
    trainer.close()


if __name__ == "__main__":
    main()
