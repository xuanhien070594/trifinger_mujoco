from setuptools import setup

setup(
    name="trifinger_mujoco",
    version="0.0.1",
    install_requires=["gym==0.19.0", "mujoco-py"],  # Add any other dependencies here
    python_requires=">=3",
    description="trifinger-mujoco: A gym-compatible environment of Trifinger robot for contact-rich manipulation tasks",
    author="Hien Bui",
    author_email="xuanhien@seas.upenn.edu",
    url="https://github.com/xuanhien070594/trifinger-mujoco",
    license="MIT",
)
