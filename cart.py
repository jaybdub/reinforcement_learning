import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from collections import namedtuple
from gym.envs.classic_control import CartPoleEnv


class CartModel(nn.Module):

    def __init__(self):
        super(CartModel, self).__init__()
        self.fc1 = nn.Linear(4, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train_step(model, memory, optimizer, batch_size=124, gamma=0.99):
    if batch_size >= len(memory):
        return
    batch = memory.sample(batch_size)

    # run model on batch
    non_terminal_mask = torch.ByteTensor([s.next_state is not None
                                          for s in batch])
    next_state_batch = Variable(torch.cat([s.next_state[None, :] for s in batch
                                           if s.next_state is not None]))
    state_batch = Variable(torch.cat([s.state[None, :] for s in batch]))
    action_batch = Variable(torch.cat([s.action[None, :] for s in batch]))
    reward_batch = Variable(torch.cat([s.reward for s in batch]))

    # run model
    state_reward = model(state_batch).gather(1, action_batch)
    next_reward_only = model(next_state_batch).max(1)

    next_reward = torch.zeros(len(batch))
    next_reward[non_terminal_mask] = next_reward_only

    target_reward = reward_batch + gamma * Variable(next_reward)
    loss = F.soft_margin_loss(state_reward, target_reward)

    model.zero_grad()
    loss.backward()
    optimizer.step()


def prob_accept(step_count, eps_start=0.9, eps_end=0.05):
    return 0.5


def train(env, model, optimizer, num_episodes=500, max_episode_length=500,
          memory_size=100000):
    memory = ReplayMemory(memory_size)
    step_count = 0
    for i in range(num_episodes):
        state = torch.from_numpy(env.reset()).float()
        for j in range(max_episode_length):
            env.render()
            action = torch.LongTensor([random.randrange(2)])
            if random.random() < prob_accept(step_count):
                pass
            next_state, reward, done, info = env.step(action.numpy().argmax())
            next_state = torch.from_numpy(next_state).float()
            reward = torch.Tensor([reward])
            if done:
                next_state = None
            memory.push(state, action, reward, next_state)
            state = next_state
            step_count += 1
            train_step(model, memory, optimizer)
            if done:
                break


if __name__ == '__main__':
    model = CartModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    env = CartPoleEnv()
    train(env, model, optimizer)
