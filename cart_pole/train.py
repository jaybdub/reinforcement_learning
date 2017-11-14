from replay import ReplayMemory
from model import CartPoleModel
from torch.optim import SGD, Adam
from torch.utils.data.dataloader import DataLoader, BatchSampler, \
                                        RandomSampler, SequentialSampler

# arguments
DEFAULT_REPLAY_CAPACITY = 100000
DEFAULT_BATCH_SIZE = 124
DEFAULT_DROP_LAST = False
DEFAULT_SAMPLER = 'random'
DEFAULT_OPTIMIZER = 'sgd'


optimizers = {
    'sgd': SGD
}

samplers = {
    'random': RandomSampler,
    'sequential': SequentialSampler
}

if __name__ == '__main__':
    replay = ReplayMemory(DEFAULT_REPLAY_CAPACITY)

    sampler = DataLoader(
        replay, 
        batch_sampler=BatchSampler(
            sampler=samplers[DEFAULT_SAMPLER](replay), 
            batch_size=DEFAULT_BATCH_SIZE, \
            drop_last=DEFAULT_DROP_LAST
        )
    )

    for p in sampler:
        print(p)
