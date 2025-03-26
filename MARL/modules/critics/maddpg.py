# code adapted from https://github.com/oxwhirl/facmac/
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_agents
        print(f"Input Shape: {self.input_shape}")
        if self.args.config['obs_last_action']:
            self.input_shape += self.n_actions
        self.hidden_dim = args.config["hidden_dim"]
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1)
        # print(inputs.shape)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # print(scheme["state"]["vshape"], scheme["obs"]["vshape"], self.n_agents, scheme["actions_one"])
        # whether to add the individual observation
        if self.args.config['obs_individual_obs']:
            input_shape += scheme["obs"]["vshape"]
        # agent id
        if self.args.config['obs_agent_id']:
            input_shape += self.n_agents
        return input_shape
