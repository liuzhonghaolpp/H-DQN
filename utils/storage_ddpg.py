import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class Storage_ddpg(object):
    def __init__(self, size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []
        self.maxsize = int(size)
        self.next_idx = 0

    def add(self, state, action, reward, next_state, done):
        mask = torch.FloatTensor([[0.0 if done else [1.0]]])
        action = action.unsqueeze(0)
        reward = torch.FloatTensor(np.array([reward])).unsqueeze(1)
        state = state.unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        if self.next_idx >= len(self.states):
            self.states.append(state)
            self.rewards.append(reward)
            self.actions.append(action)
            self.masks.append(mask)
            self.next_states.append(next_state)
        else:
            self.states[self.next_idx] = state
            self.rewards[self.next_idx] = reward
            self.actions[self.next_idx] = action
            self.masks[self.next_idx] = mask
            self.next_states[self.next_idx] = next_state

        self.next_idx += (self.next_idx + 1) % self.maxsize

    def compute(self, ):
        self.states_tensor = torch.cat(self.states)
        self.actions_tensor = torch.cat(self.actions)
        self.rewards_tensor = torch.cat(self.rewards)
        self.masks_tensor = torch.cat(self.masks)
        self.next_states_tensor = torch.cat(self.next_states)

    def sample(self, mini_batch_size):
        batch_size = self.states_