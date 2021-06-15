from setuptools import setup

setup(
    name="gym_spaceinvaders",
    version="0.0.1",
    author="Sebastián Baeza",
    description="Custom Space Invaders environment for OpenAI Gym",
    install_requires=["gym", "gym[atari]"],
)
