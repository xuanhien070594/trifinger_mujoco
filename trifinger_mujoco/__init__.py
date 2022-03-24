from gym.envs.registration import register

# Register the environments with OpenAI Gym
register(
    id="Trifinger-v0",
    entry_point="trifinger_mujoco.envs:TrifingerEnv",
    max_episode_steps=50,
)