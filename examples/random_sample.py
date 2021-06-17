import time

import gym
import gym_spaceinvaders

# Before you can make a CustomSpaceInvaders Environment you need to call:
# import gym_CustomSpaceInvaders
# This import statement registers all CustomSpaceInvaders environments
# provided by this package
env_name = "CustomSpaceInvaders-v0"
env = gym.make(env_name)

episodes = 5
for i_episode in range(episodes):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        score += reward

    print("Episode: {} Score: {}".format(i_episode + 1, score))

env.close()
