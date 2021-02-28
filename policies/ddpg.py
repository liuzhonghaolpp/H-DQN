import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):

    def __init__(self, num_obs, num_action, num_hidden_1, num_hidden_2):
        super(Actor, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(num_obs, num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.Relu(),
            nn.Linear(num_hidden_2, num_action),
            nn.Tanh()
        )
        self.train()

    def forward(self, x):
        return self.base(x)

    def act(self, state):
        action = self.base(torch.unsqueeze(torch.FloatTensor(state)), 0)
        return action.detach().numpy()


class Critic(nn.Module):

    def __init__(self, num_obs, num_action, num_hidden_1, num_hidden_2):
        super(Critic, self).__init__()
        self.fc_o = nn.Linear(num_obs, num_hidden_1)
        self.fc_a = nn.Linear(num_action, num_hidden_1)
        self.fc_2 = nn.Linear(num_hidden_1*2, num_hidden_2)
        self.out = nn.Linear(num_hidden_2, 1)
        self.Relu = nn.Relu()

    def forward(self, obs, action):
        x_o = self.ReLU(self.fc_o(obs))
        x_a = self.Relu(self.fc_a(action))
        x = torch.cat([x_o, x_a], dim=1)
        x = self.Relu(self.fc_2(x))
        value = self.out(x)
        return value


class Critic_2(nn.Module):

    def __init__(self, num_obs, num_action, num_hidden_1, num_hidden_2):
        super(Critic_2, self).__init__()
        self.fc_1 = nn.Linear(num_obs+num_action, num_hidden_1)
        self.fc_2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.fc_3 = nn.Linear(num_hidden_2, 1)
        self.relu = nn.Relu()

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = self.relu(self.fc_1(x))
        x = self.relu(self.fc_2(x))
        value = self.fc_3(x)
        return value