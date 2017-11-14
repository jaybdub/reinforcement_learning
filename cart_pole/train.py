import torch
from transition import Transition
from torch import Tensor
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from replay import ReplayMemory
from model import CartPoleModel
from torch.optim import SGD, Adam
from torch.utils.data.dataloader import DataLoader, BatchSampler, \
                                        RandomSampler, SequentialSampler

# arguments
DEFAULT_NUM_EPISODES = 500
DEFAULT_MAX_EPISODE_LENGTH = 500
DEFAULT_VISUALIZE = True
DEFAULT_REPLAY_CAPACITY = 100000
DEFAULT_BATCH_SIZE = 124
DEFAULT_DROP_LAST = False
DEFAULT_SAMPLER = 'random'
DEFAULT_OPTIMIZER = 'sgd'
DEFAULT_OPTIMIZER_LR = 0.01
DEFAULT_OPTIMIZER_MOMENTUM = 0.0

optimizers = {
    'sgd': SGD
}

samplers = {
    'random': RandomSampler,
    'sequential': SequentialSampler
}

def tensor_argmax(t):
    return t.data.numpy().argmax()

if __name__ == '__main__':

    model = CartPoleModel()

    replay = ReplayMemory(DEFAULT_REPLAY_CAPACITY)

    sampler = DataLoader(
        replay, 
        batch_sampler=BatchSampler(
            sampler=samplers[DEFAULT_SAMPLER](replay), 
            batch_size=DEFAULT_BATCH_SIZE, \
            drop_last=DEFAULT_DROP_LAST
        )
    )

    optimizer = optimizers[DEFAULT_OPTIMIZER](
        model.parameters(), 
        lr=DEFAULT_OPTIMIZER_LR,
        momentum=DEFAULT_OPTIMIZER_MOMENTUM
    )

    env = CartPoleEnv()

    for episode in range(DEFAULT_OPTIMIZER_LR):

        state = env.reset() 
        done = False
        step = 0

        while step < DEFAULT_MAX_EPISODE_LENGTH and not done:
            # sample action
            action = tensor_argmax(model(Tensor(state)))

            # step environment
            next_state, reward, done, info = env.step(action)

            # add to replay memory
            replay.push(Transition((state, action, reward, next_state)))

            # (update model?)
            step += 1

