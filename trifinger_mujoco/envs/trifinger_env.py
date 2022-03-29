import numpy as np
from typing import Dict, Tuple, Any
from os import path
from gym import utils
from gym.envs.mujoco import mujoco_env


class TrifingerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_contact_forces=False):
        """Instantiate TrifingerEnv object.

        Args:
          use_contact_forces: appending the net forces acting fingers and cube into observation data

        """
        self.use_contact_forces = use_contact_forces
        model_path = path.join(
            path.dirname(__file__), "../models/trifinger_with_cube.xml"
        )
        mujoco_env.MujocoEnv.__init__(self, model_path, 20)
        utils.EzPickle.__init__(self)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Forward simulation.

        Args:
          action: Shape(9, ). Desired joint torque

        """
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        done = self._is_done(obs)
        reward = self._reward(obs, action)
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

          Optional:
            - net forces on fingers: obs[31:40]
            - net forces on cube: obs[40:43]

        """
        obs = np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

        if self.use_contact_forces:
            obs = np.concatenate(
                [obs, self.get_contact_force_fingers(), self.get_contact_force_cube()]
            )

        return obs

    def reset_model(self) -> np.ndarray:
        """Set initial conditions for the environment."""

        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self) -> None:
        """Set up camera views here."""
        pass

    def get_contact_force_fingers(self) -> np.ndarray:
        """Retrieve contact forces on the tips of three fingers.

        Returns:
          contact_forces: Shape(9, ) the concatenated array of 3 force vectors acting
                          on 3 lower links of 3 fingers

        """
        lower_link_finger_0_id = self.sim.model.body_name2id("finger_lower_link_0")
        lower_link_finger_120_id = self.sim.model.body_name2id("finger_lower_link_120")
        lower_link_finger_240_id = self.sim.model.body_name2id("finger_lower_link_240")

        # obtain all force data, note that these are net forces acting on all bodies in Mujoco
        force_data = self.sim.data.cfrc_ext

        return np.concatenate(
            [
                force_data[lower_link_finger_0_id][3:],
                force_data[lower_link_finger_120_id][3:],
                force_data[lower_link_finger_240_id][3:],
            ]
        )

    def get_contact_force_cube(self) -> np.ndarray:
        """Retrieve the net force acting on the cube.

        Returns:
          contact_forces: Shape(3, )

        """
        cube_id = self.sim.model.body_name2id("cube")
        return self.sim.data.cfrc_ext[cube_id][3:]
