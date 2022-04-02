import trifinger_mujoco
import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

if __name__ == "__main__":
    env = gym.make("TrifingerForces-v0")
    env.reset()
    # video = VideoRecorder(env, "trifinger.mp4")
    print(env.action_space)

    for i in range(1000):
        obs, reward, done, _ = env.step(np.ones(9))
        # video.capture_frame()
        env.render()
    # video.close()
