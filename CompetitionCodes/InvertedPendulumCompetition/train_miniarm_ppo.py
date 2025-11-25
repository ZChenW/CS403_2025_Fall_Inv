import gymnasium as gym
from envs.Pendulum_gym import MiniArmPendulumEnv


def make_env(env_id: str, idx: int, capture_video: bool, run_name: str, gamma: float):
    def thunk():
        env = MiniArmPendulumEnv()

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        env.action_space.seed(idx)
        env.observation_space.seed(idx)
        return env

    return thunk
