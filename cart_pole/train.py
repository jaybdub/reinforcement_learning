import random
import pdb
import torch
from transition import Transition
from torch import Tensor, LongTensor
import numpy as np
from gym.envs.classic_control import CartPoleEnv
from replay import ReplayMemory
from model import CartPoleModel
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data.dataloader import DataLoader, BatchSampler, \
                                        RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.autograd import Variable

# arguments
DEFAULT_NUM_EPISODES = 10000
DEFAULT_MAX_EPISODE_LENGTH = 3000
DEFAULT_VISUALIZE = True
DEFAULT_REPLAY_CAPACITY = 100000
DEFAULT_BATCH_SIZE = 1024
DEFAULT_DROP_LAST = False
DEFAULT_SAMPLER = 'random'
DEFAULT_OPTIMIZER = 'rmsprop'
DEFAULT_OPTIMIZER_LR = 0.0001
DEFAULT_OPTIMIZER_MOMENTUM = 0.0
GAMMA = 0.99

optimizers = {
    'sgd': SGD,
    'rmsprop': RMSprop
}

samplers = {
    'random': RandomSampler,
    'sequential': SequentialSampler
}

def tensor_argmax(t):
    return int(t.data.numpy().argmax())


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

    optimizer = optimizers[DEFAULT_OPTIMIZER](model.parameters())

    env = CartPoleEnv()

    for episode in range(DEFAULT_NUM_EPISODES):

        state = env.reset() 
        done = False
        step = 0

        while step < DEFAULT_MAX_EPISODE_LENGTH and not done:
            env.render()

            # sample action
            action = random.randrange(2)
            if random.random() < 0.95:
                action = tensor_argmax(model(Variable(Tensor(state), volatile=True)))

            # step environment
            next_state, reward, done, info = env.step(action)
            valid = 1
            if done:
                valid = 0
            # add to replay memory
            replay.push(Transition(state, action, reward, next_state, valid))
            state = next_state

            # (update model?)
            step += 1

            # update model
            #pdb.set_trace()

        batch = next(sampler.__iter__())

        q_now = model(Variable(batch[0].float()))
        q_next = model(Variable(batch[3].float(), volatile=True))

        reward_now = q_now.gather(1, Variable(batch[1][:, None]))
        reward_next = q_next.max(1)[0][:, None]
        reward_next.volatile = False
        reward_trans_target = Variable(batch[2][:, None].float())
        reward_now_target = reward_trans_target + GAMMA * reward_next * \
            Variable(batch[4][:, None].float())

        loss = F.smooth_l1_loss(reward_now, reward_now_target)
        #print(loss, step, reward_now_target.max())

        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(episode, step)

