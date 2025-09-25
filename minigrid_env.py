import gymnasium as gym
import minigrid
import numpy as np
from gymnasium import spaces


class Minigrid:
    def __init__(self, env_name, realtime_mode=False):
        # Set the environment rendering mode
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"

        self._env = gym.make(env_name, agent_view_size=3, tile_size=28, render_mode=render_mode)
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        self._env = minigrid.wrappers.RGBImgPartialObsWrapper(self._env, tile_size=28)
        self._env = minigrid.wrappers.ImgObsWrapper(self._env)
        self._env = TransposeAndNormalizeObs(self._env)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return spaces.Discrete(3)

    def reset(self):
        self._rewards = []
        obs, info = self._env.reset(seed=np.random.randint(0, 99))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action[0])
        self._rewards.append(reward)
        if terminated or truncated:
            info["episode"] = {"r": sum(self._rewards), "l": len(self._rewards)}
        else:
            info = None
        return obs, reward, terminated, truncated, info


class TransposeAndNormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, h, w), dtype=np.float32
        )

    def observation(self, obs):
        o = obs.astype(np.float32) / 255.0
        # Convert from (H, W, C) to (C, H, W)
        o = np.transpose(o, (2, 0, 1))
        return o


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id, agent_view_size=3, tile_size=28, render_mode="rgb_array")
    env = minigrid.wrappers.RGBImgPartialObsWrapper(env, tile_size=28)
    env = minigrid.wrappers.ImgObsWrapper(env)
    env = ReduceActionSpaceWrapper(env, n_actions=3)
    # env = DiscreteToContinuousWrapper(env)
    # env = ResizeObs(env, shape=(3, 96, 96))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = TransposeAndNormalizeObs(env)
    return env


class ReduceActionSpaceWrapper(gym.Wrapper):
    """
    Reduce discrete action space to only relevant actions.
    For MiniGrid Memory environments, reduce to 3 actions: turn left, turn right, move forward.
    """

    def __init__(self, env, n_actions):
        super().__init__(env)
        self.n_actions = n_actions
        self.action_space = gym.spaces.Discrete(n_actions)

    def step(self, action):
        return self.env.step(action)
