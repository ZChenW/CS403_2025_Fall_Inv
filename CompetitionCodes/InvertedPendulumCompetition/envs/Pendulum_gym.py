import os
from typing import Optional, Tuple
from typing_extensions import Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


XML_REL_PATH = os.path.join("Robot", "miniArm_with_pendulum.xml")

N_JOINTS = 6
N_FRAME = 10
MAX_EPISODE_STEPS = 2000
CTRL_MAX = np.array([10, 25, 15, 20, 10, 5], dtype=np.float32)


def get_xml_local_file() -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    xml_path = os.path.join(dir_path, "..", XML_REL_PATH)
    if not os.path.exists(xml_path):
        pwd = os.getcwd()
        raise FileNotFoundError(
            f"Xml file not found : {xml_path}\nCurrent file location: {pwd}"
        )
    return xml_path


class MiniArmPendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(self, render_mode: Optional[str] = None):
        xml_file = get_xml_local_file()
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.renderer = MujocoRenderer(self.model, self.data)

        self.dt = self.model.opt.timestep
        self.n_frame = N_FRAME
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.step_counter = 0
        self.n_joints = N_JOINTS
        self.ctrl_max = CTRL_MAX
        self.ctrl_cost_weight = 0.001
        self.terminate_angle = np.deg2rad(60)  # Termiante angle

        self.pen_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum"
        )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_joints,), dtype=np.float32
        )

        n_q = self.model.nq
        n_v = self.model.nv
        self.obs_dim = n_q + n_v
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(self.obs_dim,), dtype=np.float32
        )

        self.render_mode = render_mode

        try:
            mujoco.mj_step(self.model, self.data, nstep=1)
            self._use_nstep = True
        except TypeError:
            self._use_nstep = False

    def _reset_simulation(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        noise_pos = 0.01
        noise_vel = 0.01

        self.data.qpos[: self.n_joints] += self.np_random.uniform(
            low=-noise_pos, high=noise_pos, size=self.n_joints
        )
        self.data.qvel[: self.n_joints] += self.np_random.uniform(
            low=-noise_vel, high=noise_vel, size=self.n_joints
        )
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        obs = np.concatenate([qpos, qvel], axis=0)
        return obs.astype(np.float32)

    def _get_reset_info(self) -> Dict[str, float]:
        return {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_counter = 0

        self._reset_simulation()

        ob = self._get_obs()
        info = self._get_reset_info()

        return ob, info

    ##### step() #####

    def _pendulum_cos_theta(self) -> float:
        quat = np.asarray(self.data.body(self.pen_id).xquat, dtype=np.float64)
        R_quat = np.empty(9, dtype=np.float64)
        mujoco._functions.mju_quat2Mat(R_quat, quat)
        R = R_quat.reshape(3, 3)
        local_z = R[:, 2]
        return float(
            local_z[2]
        )  # local_Z * world_z = local_Z[0]*0 + local_Z[1]*0 + local_Z[2]*1

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_joints,):
            raise ValueError(
                f"Expected action shape {self.n_joints}, get {action.shape}"
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)

        ctrl = action * self.ctrl_max
        self.data.ctrl[: self.n_joints] = ctrl
        return ctrl

    def _simulation(self) -> None:
        if self._use_nstep:
            mujoco.mj_step(self.model, self.data, nstep=self.n_frame)
        else:
            for _ in range(self.n_frame):
                mujoco.mj_step(self.model, self.data)

    def _compute_reward(self, ctrl: np.ndarray) -> Tuple[float, float, float]:
        cos_theta = self._pendulum_cos_theta()  # cos_theta = 1 => theta = 0
        theta_reward = (cos_theta + 1.0) / 2.0
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(ctrl**2))
        reward = theta_reward - ctrl_cost
        return reward, theta_reward, ctrl_cost

    def _check_terminate(self, cos_theta: float) -> bool:
        theta = float(np.arccos(np.clip(cos_theta, -1, 1)))
        return bool(theta > self.terminate_angle)

    def _check_trun(self) -> bool:
        return bool(self.step_counter >= self.max_episode_steps)

    def step(self, action: np.ndarray):
        self.step_counter += 1
        ctrl = self._apply_action(action)
        self._simulation()
        cos_theta = self._pendulum_cos_theta()

        reward, theta_reward, ctrl_cost = self._compute_reward(ctrl)

        terminated = self._check_terminate(cos_theta)
        truncated = self._check_trun()

        obs = self._get_obs()
        info = {
            "cos_theta": cos_theta,
            "theta_reward": theta_reward,
            "ctrl_cost": ctrl_cost,
            "step": self.step_counter,
        }

        return obs, reward, terminated, truncated, info

    #### render() ####
    def render(self):
        if self.render_mode is None:
            return None
        return self.renderer.render(
            render_mode=self.render_mode,
            camera_id=None,
            camera_name=None,
        )

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
