import numpy as np
import gym


class DefaultWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def reset(self):
        return self.env.reset()
