import gym
import numpy as np


class FlattenWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.agent_count = len(self.env.list_intersection)

    def reset(self):
        states = self.env.reset()
        states = states.reshape((self.agent_count, -1))
        return states

    def step(self, actions):
        actions = actions.reshape((self.agent_count,))
        states, rewards, done, infos = self.env.step(actions)
        states = states.reshape((self.agent_count, -1))
        rewards = rewards.flatten()
        return states, rewards, done, infos
