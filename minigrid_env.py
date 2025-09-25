import gymnasium as gym
import minigrid
import numpy as np


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
