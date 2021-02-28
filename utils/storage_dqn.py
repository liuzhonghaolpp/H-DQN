import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import random


class Storage_dqn(object):
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

    def make_index(self, batch_size):
        idxes = [random.randint(0, len(self.states)-1) for _ in range(batch_size)]
        return idxes

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self.states))
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states, masks = [], [], [], [], []
        for i in idxes:
            states.append(self.states[i])
            actions.append(self.actions[i])
            rewards.append(self.rewards[i])
            next_states.append(self.rewards[i])
            masks.append(self.masks[i])
        return states, actions, rewards, next_states, masks