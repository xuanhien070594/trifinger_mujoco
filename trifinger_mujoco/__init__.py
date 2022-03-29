import yaml
from os import path
from gym.envs.registration import register

# Parse config file of all custom environments
config_path = path.join(path.dirname(__file__), "envs/env_configs.yaml")
with open(config_path, "r") as f:
    env_configs = yaml.safe_load(f)

# Register the environments with OpenAI Gym
register(
    id="Trifinger-v0",
    entry_point="trifinger_mujoco.envs:TrifingerEnv",
    max_episode_steps=200,
    kwargs={"env_configs": env_configs["Trifinger-v0"]},
)

register(
    id="TrifingerForces-v0",
    entry_point="trifinger_mujoco.envs:TrifingerEnv",
    max_episode_steps=200,
    kwargs={"env_configs": env_configs["TrifingerForces-v0"]},
)
