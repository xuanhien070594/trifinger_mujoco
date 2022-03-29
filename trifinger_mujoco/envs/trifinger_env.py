import numpy as np
from typing import Dict, Tuple, Any
from os import path
from gym import utils
from gym.envs.mujoco import mujoco_env


class TrifingerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_contact_forces=False):
        """Instantiate TrifingerEnv object.

        Args:
          use_contact_forces: appending contact forces into observation data

        """
        model_path = path.join(
            path.dirname(__file__), "../models/trifinger_with_cube.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 20)
        utils.EzPickle.__init__(self)

    def step(
        self, action: np.ndarray
    ) -> Tuple(np.ndarray, float, bool, Dict[Any, Any]):
        """Forward simulation.

        Args:
          action: Shape(9, ). Desired joint torque

        """
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        done = self._is_done(obs)
        reward = self._reward(obs, action, is_done=done)
        return (obs, reward, done, {})

    def _reward(self, state: np.ndarray, action: np.ndarray):
        """Define reward function."""
        pass

    def _is_done(self, state: np.ndarray) -> bool:
        """Check if terminating conditions are met, given the current state."""
        pass

    def _get_obs(self) -> np.ndarray:
        """Retrieve the current observation data from Mujoco.

        Returns:

          The total dimensions of observation space is 31, including:

            - cube position: obs[:3]
            - cube rotation: obs[3:7] (represented by quaternion)
            - finger joints position: obs[7:16]
            - cube velocity: obs[16:22]
            - finger joints velocity: obs[22:31]

        """
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self) -> np.ndarray:
        """Set initial conditions for the environment."""

        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self) -> None:
        """Set up camera views here."""
        pass
