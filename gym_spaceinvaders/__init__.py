from gym.envs.registration import register

register(
    id="CustomSpaceInvaders-v0",
    entry_point="gym_spaceinvaders.envs:CustomSpaceInvadersEnv",
    max_episode_steps=10000,
)
