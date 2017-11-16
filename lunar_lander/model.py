import torch
import torch.nn as nn
import torch.nn.functional as F


class QPredictor(nn.Module):
    """ Predicts the reward expected from a (state, action) pair.
    """

    def __init__(self):
        super(QPredictor, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ValuePredictor(nn.Module):
    """ Predicts the reward expected from a state (over all actions).
    """

    def __init__(self):
        super(ValuePredictor, self).__init__()
        self.fc1 = nn.Linear(3, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 1)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ActionPredictor(nn.Module):
    """ Predicts the action with the best relative Q reward.  That is,
        action = argmax QPredictor(state, action) - ValuePredictor(state)
    """

    def __init__(self):
        super(ActionPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 1)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
