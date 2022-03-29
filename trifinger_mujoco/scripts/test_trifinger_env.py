import trifinger_mujoco
import numpy as np
import gym

if __name__ == "__main__":
    env = gym.make("TrifingerForces-v0")
    env.reset()
    for i in range(100000):
        obs, reward, done, _ = env.step(np.zeros(9))
        env.render()
        print(obs.shape)
