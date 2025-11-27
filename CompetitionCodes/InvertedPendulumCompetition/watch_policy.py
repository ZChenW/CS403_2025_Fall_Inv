import time
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from envs.Pendulum_gym import MiniArmPendulumEnv
from Online_RL.ppo_continuous_action import Agent
from train_miniarm_ppo import make_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = (
    "runs/MiniArmPendulum-v0__ppo_continuous_action__1__1764216838/"
    "ppo_continuous_action.cleanrl_model"
)


def load_agent(model_path: str) -> Agent:
    envs = gym.vector.SyncVectorEnv([make_env("dummy", 0, False, "eval", gamma=0.99)])
    agent = Agent(envs).to(DEVICE)
    envs.close()

    state_dict = torch.load(model_path, map_location=DEVICE)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def main():
    agent = load_agent(MODEL_PATH)

    env = MiniArmPendulumEnv(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="videos/miniarm_policy/1",
        name_prefix="policy",
        episode_trigger=lambda ep: True,
    )

    for ep in range(3):
        obs, info = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            with torch.no_grad():
                action_mean = agent.actor_mean(obs_tensor)
            action = action_mean.cpu().numpy().flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    print("Video saved to videos/miniarm_policy/2")


if __name__ == "__main__":
    main()
