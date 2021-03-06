# Neural network model(s) for the pygym 'CartPoleEnv'
#
# author: John Welsh

from torch.utils.data.dataloader import DataLoader
from replay import ReplayMemory
import unittest


class ReplayMemoryTest(unittest.TestCase):

    def setUp(self):
        self.replay = ReplayMemory(10)

    def testPush(self):
        for i in range(10):
            self.replay.push(i)
            self.assertEqual(len(self.replay), i + 1)
            self.assertEqual(self.replay[i], i)
        self.replay.push(11)
        self.assertEqual(len(self.replay), 10)
        self.assertEqual(self.replay[0], 11)


if __name__ == '__main__':
    unittest.main()
