# Neural network model(s) for the pygym 'CartPoleEnv'
#
# author: John Welsh

import torch
import torch.nn as nn
import torch.nn.functional as F


class CartPoleModel(nn.Module):

    def __init__(self):
        super(CartPoleModel, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
