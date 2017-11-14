import pdb
import torch
from transition import Transition
from torch import Tensor, LongTensor
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from replay import ReplayMemory
from model import CartPoleModel
from torch.optim import SGD, Adam
from torch.utils.data.dataloader import DataLoader, BatchSampler, \
                                        RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.autograd import Variable

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
GAMMA = 0.99

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

    for episode in range(DEFAULT_NUM_EPISODES):

        state = env.reset() 
        done = False
        step = 0

        while step < DEFAULT_MAX_EPISODE_LENGTH and not done:
            # env.render()

            # sample action
            action = tensor_argmax(model(Variable(Tensor(state))))

            # step environment
            next_state, reward, done, info = env.step(action)

            # add to replay memory
            replay.push(Transition(state, action, reward, next_state))
            state = next_state

            # (update model?)
            step += 1


        # update model
        pdb.set_trace()
        batch = next(sampler.__iter__())

        q_now = model(Variable(batch[0].float()))
        q_next = model(Variable(batch[3].float(), volatile=True))

        reward_now = q_now.gather(1, Variable(batch[1][:, None]))
        reward_next = q_next.max(1)[0][:, None]
        reward_next.volatile = False
        reward_now_target = reward_now + GAMMA * reward_next 

        loss = F.soft_margin_loss(reward_now, reward_now_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

