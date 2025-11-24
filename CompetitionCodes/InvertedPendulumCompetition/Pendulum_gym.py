import os
from typing import Optional, Tuple
from typing_extensions import Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from numpy._typing import _Float32Codes

XML_REL_PATH = os.path.join("Robot", "miniArm_with_pendulum.xml")

N_JOINTS = 6
N_FRAME = 10
MAX_EPISODE_STEPS = 2000


def get_xml_local_file() -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    xml_path = os.path.join(dir_path, XML_REL_PATH)
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

        self.dt = self.model.opt.timestep
        self.n_frame = N_FRAME
        self.max_episode_steps = MAX_EPISODE_STEPS
        self.step_counter = 0
        self.n_joints = N_JOINTS

        self.pen_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjNOBJECT, "pendulum")

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_joints,), dtype=np.float32
        )

        n_q = self.model.nq
        n_v = self.model.nv
        self.obs_dim = n_q + n_v
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=-high, shape=(self.obs_dim), dtype=np.float32
        )

        self.render_mode = render_mode

    def _reset_simulation(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.da6ta, 0)

        noise_pos = 0.01
        moise_vel = 0.01

        self.data.qpos[: self.n_joints] += self.np_random.uniform(
            low=-noise_pos, high=noise_pos, size=self.n_joints
        )
        self.data.qvel[: self.n_joints] += self.np_random.uniform(
            low=-moise_vel, high=moise_vel, size=self.n_joints
        )
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        obs = np.concatenate([qpos, qvel], axis=0)
        return obs

    def _get_reset_info(self) -> Dict[str, float]:
        return {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_counter = 0

        self._reset_simulation()

        ob = self._get_obs()
        info = self._get_reset_info()

        return ob, info
