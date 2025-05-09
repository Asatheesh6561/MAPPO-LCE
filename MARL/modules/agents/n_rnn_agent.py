import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import pdb

from MARL.utils.th_utils import orthogonal_init_


class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args
        self.hidden_dim = self.args.config["hidden_dim"]
        self.n_lambda = self.args.config["n_lambda"]
        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        if args.config["use_n_lambda"]:
            self.fc2 = nn.Linear(self.hidden_dim, self.n_lambda * args.n_actions)
        else:
            self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # Batch x Agents x Dim_state
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        hh = self.rnn(x, h_in)

        q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)
