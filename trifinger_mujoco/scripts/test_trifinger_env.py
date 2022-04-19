import trifinger_mujoco
import numpy as np
import gym
import time

if __name__ == "__main__":
    env = gym.make("TrifingerForces-v0")
    env.reset()

    sim_step_time_data = []
    for i in range(1000):
        start_time = time.time()
        obs, reward, done, _ = env.step(np.random.uniform(-0.2, 0.2, 9))
        sim_step_time_data.append(time.time() - start_time)
        env.render()
    print(f"Frame skip is {env.frame_skip}")
    print(f"The average time taken to simulate one step is: {np.array(sim_step_time_data).mean() / env.frame_skip}s")
