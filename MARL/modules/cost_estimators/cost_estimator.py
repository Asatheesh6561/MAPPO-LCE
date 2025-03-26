import torch
import torch.nn as nn
import torch.nn.functional as F


class CostEstimator(nn.Module):
    def __init__(self, state_shape, action_shape, args):
        super(CostEstimator, self).__init__()
        self.args = args
        self.hidden_dim = self.args.config["cost_hidden_dim"]
        input_dim = state_shape + action_shape
        self.device = "cuda" if args.use_cuda else "cpu"
        self.fc1 = nn.Linear(input_dim, self.hidden_dim).to(self.device)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.fc4 = nn.Linear(self.hidden_dim, 1).to(self.device)

    def forward(self, state, action):
        inputs = torch.cat((state.float(), action.float()), dim=-1).to(self.device)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        cost_estimate = self.fc4(x)
        return cost_estimate
