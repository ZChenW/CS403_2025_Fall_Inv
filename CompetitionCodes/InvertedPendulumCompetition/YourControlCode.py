import mujoco
import numpy as np

import torch
from envs.Pendulum_gym import MiniArmPendulumEnv, CTRL_MAX
from watch_policy import load_agent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "runs/MiniArmPendulum-v0__ppo_continuous_action__1__1764558724/ppo_continuous_action.cleanrl_model"


class YourCtrl:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):  # type: ignore[attr-defined]
        self.model = m
        self.data = d
        self.init_qpos = d.qpos.copy()

        dummy_env = MiniArmPendulumEnv(render_mode=None)
        self.obs_dim = dummy_env.observation_space.shape[0]
        self.act_dim = dummy_env.action_space.shape[0]

        self.agent = load_agent(MODEL_PATH)

    def _build_obs(self) -> np.ndarray:
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        obs = np.concatenate([qpos, qvel], axis=0)
        return obs.astype(np.float32)

    def CtrlUpdate(self):
        obs = self._build_obs()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action_mean = self.agent.actor_mean(obs_tensor)
        action = action_mean.cpu().numpy().flatten()
        action = np.clip(action, -1.0, 1.0)
        torque = action * CTRL_MAX
        self.data.ctrl[: self.act_dim] = torque

        return True
