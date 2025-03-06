from unittest import TestCase
import torch
from utils import unravel_index
from tqdm import trange


class Test(TestCase):
    def test_2d(self):
        shape = [4, 2]
        for idx in trange(shape[0] * shape[1]):
            expected = torch.unravel_index(torch.tensor(idx), shape)
            expected = [e.item() for e in expected]
            actual = unravel_index(idx, shape)
            self.assertEqual(expected, actual)

    def test_3d(self):
        shape = [64, 64, 64]
        for idx in trange(shape[0] * shape[1] * shape[2]):
            expected = torch.unravel_index(torch.tensor(idx), shape)
            expected = [e.item() for e in expected]
            actual = unravel_index(idx, shape)
            self.assertEqual(expected, actual)