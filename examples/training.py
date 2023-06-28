import gymnasium as gym

import xroute_env

env = gym.make("xroute_env/ordering-training-v0")

observation = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, done = env.step(action)

env.close()
