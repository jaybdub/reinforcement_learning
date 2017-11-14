# Neural network model(s) for the pygym 'CartPoleEnv'
#
# author: John Welsh

import torch
import torch.nn as nn
import torch.nn.functional as F


class CartPoleModel(nn.Module):

    def __init__(self):
        super(CartPoleModel, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
