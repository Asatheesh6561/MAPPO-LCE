import numpy as np
import gym
from gym import spaces
from typing import Tuple


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.n_agents = len(self.list_intersection)
        self.state_dim = 0

        self.observation_space_keys = self.env.observation_space_keys
        self.observation_space = self.env.observation_space["intersections"]
        self.num_lanes_per_ts = 12
        skip_keys = [
            "Envtime",
            "TSprevphases",
            "RoadLinkDirection",
            "DirectionNames",
            "RoadsOut",
            "RoadsIn",
            "RoadLinksOut",
            "RoadLinksIn",
            "RoadOutLanes",
        ]
        self.state_key_dims = {}
        for key in self.observation_space_keys:
            if key in skip_keys:
                continue
            elif key == "TSphase" or key == "LaneCount" or key == "TStime":
                self.state_dim += 1
                self.state_key_dims[key] = 1
            elif isinstance(self.observation_space[0][key], np.ndarray):
                if len(self.observation_space[0][key].shape) >= 1:
                    prod_dim = np.prod(self.observation_space[0][key].shape)
                    self.state_dim += prod_dim // self.num_lanes_per_ts
                    self.state_key_dims[key] = prod_dim // self.num_lanes_per_ts
                else:
                    self.state_dim += 1
                    self.state_key_dims[key] = 1
            elif type(self.observation_space[0][key]) == list:
                prod_dim = np.prod(self.observation_space[0][key])
                self.state_dim += prod_dim // self.num_lanes_per_ts
                self.state_key_dims[key] = prod_dim // self.num_lanes_per_ts
            else:
                self.state_dim += 1
                self.state_key_dims[key] = 1
        # pressure
        self.state_dim += 1
        self.state_key_dims["pressure"] = 1
        self.state = np.zeros(
            (
                self.n_agents,
                self.num_lanes_per_ts,
                self.state_dim,
            )
        )

        # set to anything, only shape matters
        self.agent_observation_space = spaces.Box(
            low=-5000.00,
            high=5000.00,
            shape=(
                self.num_lanes_per_ts,
                self.state_dim,
            ),
            dtype=np.float64,
        )
        self.observation_space = [self.agent_observation_space] * self.n_agents
        self.action_space = self.env.action_space

    def convert_state(self, state):
        new_state = np.zeros(
            (
                self.n_agents,
                self.num_lanes_per_ts,
                self.state_dim,
            )
        )
        for i in range(self.n_agents):
            count = 0
            for key in self.state_key_dims:
                key_dim = self.state_key_dims[key]
                if type(state[i][key]) == np.ndarray:
                    new_state[i, :, count : count + key_dim] = state[i][key].reshape(
                        (self.num_lanes_per_ts, -1)
                    )
                else:
                    new_state[i, :, count : count + key_dim] = np.tile(
                        [state[i][key]], (self.num_lanes_per_ts, 1)
                    )
                count += key_dim
        return new_state

    def step(self, action):
        state, all_reward, done, infos = self.env.step(action)
        self.state = self.convert_state(state)
        return self.state, all_reward, done, infos

    def reset(self):
        new_state, info = self.env.reset()
        self.state = self.convert_state(new_state)
        return self.state
