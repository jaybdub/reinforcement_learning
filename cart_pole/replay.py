# Replay memory for reinforcement learning as a torch dataset
#
# author: John Welsh


from torch.utils.data import Dataset
import numpy as np


class ReplayMemory(Dataset):

    def __init__(self, capacity):
        self._memory = [None] * capacity
        self._capacity = capacity
        self._length = 0
        self._index = 0

    def push(self, item):
        self._memory[self._index] = item
        self._index = (self._index + 1) % self._capacity
        if (self._length < self._capacity):
            self._length += 1

    def reset(self):
        self._memory = [None] * self._capacity
        self._length = 0
        self._index = 0

    def __len__(self):
        return self._length

    def __getitem__(self, items):
        return self._memory[items]
