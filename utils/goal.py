import numpy as np
import torch


class Goal(object):

    def __init__(self, goal):
        self.num_goal = 7

    def get_one_hot_goal(self, goal):
        goal = torch.nn.functional.one_hot(torch.LongTensor(goal), num_classes=self.num_goal)