import numpy as np
import torch


class RandomPolicy(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        action = self.action_space.sample()
        action = torch.FloatTensor([[action]])
        actio.to