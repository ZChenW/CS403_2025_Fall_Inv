import gymnasium as gym
from envs.Pendulum_gym import MiniArmPendulumEnv


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, gamma: float):
    def thunk():
        env = MiniArmPendulumEnv()

        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(idx)
        env.observation_space.seed(idx)
        return env

    return thunk
