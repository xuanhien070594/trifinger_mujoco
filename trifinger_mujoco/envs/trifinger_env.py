import numpy as np
from typing import Dict, Tuple, Any
from os import path
from gym import utils
from gym.envs.mujoco import mujoco_env
import trifinger_mujoco.utils as tf_utils


class TrifingerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, env_configs: Dict[str, Any]):
        """Instantiate TrifingerEnv object.

        Args:
          env_configs: contains all necessary configurations for the environment

        """
        model_path = path.join(
            path.dirname(__file__), "../models/trifinger_with_cube.xml"
        )
        # obtain config parameters
        self.env_configs = env_configs

        # initial positions and velocities for each finger
        self.init_finger_pos = np.array(env_configs["init_finger_pos"])
        self.init_finger_vel = np.array(env_configs["init_finger_vel"])

        # initial positions, orienetations and velocities for the cube
        self.init_cube_pos = np.array(env_configs["init_cube_pos"])
        self.init_cube_vel = np.array(env_configs["init_cube_vel"])
        self.init_cube_axis_angle = np.array(env_configs["init_cube_axis_angle"])
        self.init_cube_quat = tf_utils.axis_angle.to_quaternion(
            self.init_cube_axis_angle[:3], self.init_cube_axis_angle[-1]
        )

        self.max_torque = env_configs["max_torque"]
        self.use_contact_forces = env_configs["use_contact_forces"]
        self.use_random_target = env_configs["use_random_target"]

        # target positions and orienetations of the cube
        if self.use_random_target:
            self.random_target_cube_pos_low = np.array(
                env_configs["random_target_cube_pos_low"]
            )
            self.random_target_cube_pos_high = np.array(
                env_configs["random_target_cube_pos_high"]
            )

            self.target_cube_pos = np.random.uniform(
                self.random_target_cube_pos_low, self.random_target_cube_pos_high
            )

            while (
                abs(self.target_cube_pos[0]) < 0.05
                or abs(self.target_cube_pos[1]) < 0.05
            ):
                self.target_cube_pos = np.random.uniform(
                    self.random_target_cube_pos_low, self.random_target_cube_pos_high
                )
            self.target_cube_quat = tf_utils.quaternion.random()
        else:
            self.target_cube_pos = env_configs["target_cube_pos"]
            self.target_cube_axis_angle = env_configs["target_cube_axis_angle"]
            self.target_cube_quat = tf_utils.axis_angle.to_quaternion(
                self.target_cube_axis_angle[:3], self.target_cube_axis_angle[-1]
            )

        self.init_cube_pos_error = np.linalg.norm(
            self.target_cube_pos - self.init_cube_pos, ord=2
        )

        self.init_cube_quat_error = tf_utils.quaternion.quat_error(
            self.target_cube_quat, self.init_cube_quat
        )

        mujoco_env.MujocoEnv.__init__(self, model_path, 20)
        utils.EzPickle.__init__(self)

        # override the initial system (trifinger + cube) conditions
        self.init_qpos = np.concatenate(
            [
                self.init_cube_pos.ravel().copy(),
                self.init_cube_quat.ravel().copy(),
                np.tile(self.init_finger_pos.ravel().copy(), 3),
            ]
        )

        self.init_qvel = np.concatenate(
            [
                self.init_cube_vel.ravel().copy(),
                np.tile(self.init_finger_vel.ravel().copy(), 3),
            ]
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Forward simulation.

        Args:
          action: Shape(9, ). Desired joint torque

        """
        action = np.clip(action, -1, 1) * self.max_torque
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        done = self._is_done(obs)
        reward = self._reward(obs, action)
        return (obs, reward, done, {})

    def _reward(self, state: np.ndarray, action: np.ndarray):
        """Define reward function."""
        cur_cube_pos = state[:3]
        cur_cube_quat = state[3:7]

        return (
            2
            - np.linalg.norm(cur_cube_pos - self.target_cube_pos, ord=2)
            / self.init_cube_pos_error
            - tf_utils.quaternion.quat_error(self.target_cube_quat, cur_cube_quat)
            / self.init_cube_quat_error
        ) / 2.0

    def _is_done(self, state: np.ndarray) -> bool:
        """Check if terminating conditions are met, given the current state."""
        if np.linalg.norm(state[:3] - self.target_cube_pos, ord=2) < 0.002:
            return True
        return False

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
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())

        # set target positions and orienetations (visualization only)
        target_id = self.sim.model.geom_name2id("target")
        self.sim.model.geom_pos[target_id] = self.target_cube_pos
        self.sim.model.geom_quat[target_id] = self.target_cube_quat

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

    def render_target(self):
        """Function call to render the target."""
