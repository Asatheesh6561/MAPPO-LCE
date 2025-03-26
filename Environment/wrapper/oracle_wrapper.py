import numpy as np
import gym

from Environment.wrapper.default_wrapper import DefaultWrapper


class OracleWrapper(DefaultWrapper):
    def __init__(self, env):
        self.env = env

    def step(self, actions):
        states, rewards, done, info = self.env.step(actions)
        return states, rewards, done, info

    def reset(self):
        return self.env.reset()
