import time

import gym
import gym_spaceinvaders

# Before you can make a CustomSpaceInvaders Environment you need to call:
# import gym_CustomSpaceInvaders
# This import statement registers all CustomSpaceInvaders environments
# provided by this package
env_name = "CustomSpaceInvaders-v0"
env = gym.make(env_name)


for i_episode in range(1):
    observation = env.reset()

    for t in range(100):
        env.render()
        action = env.action_space.sample()

        time.sleep(0.05)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            env.render()
            break
    env.close()

time.sleep(1)
