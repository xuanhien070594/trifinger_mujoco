import numpy as np
from os import path
from gym import utils
from gym.envs.mujoco import mujoco_env


class TrifingerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        model_path = path.join(path.dirname(__file__), "../models/trifinger_with_cube.xml")
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        done = False
        ob = self._get_obs()
        reward = self._reward(ob, a, is_done=done)
        return (
            ob,
            reward,
            done,
            {}
        )

    def _reward(self, state, action, is_done=False):
        pass

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=0.0, high=0.0
        )
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0
