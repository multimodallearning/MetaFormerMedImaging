# some tests for unravel functions used in generate_block_mask in flex_token_mixer

from unittest import TestCase
import torch
from utils import unravel_index, unravel_index_2d, unravel_index_3d
from tqdm import trange


class TestGeneralUnravel(TestCase):
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


class TestSpecificUnravel(TestCase):
    def test_2d(self):
        shape = [4, 2]
        for idx in trange(shape[0] * shape[1]):
            expected = torch.unravel_index(torch.tensor(idx), shape)
            expected = [e.item() for e in expected]
            actual = unravel_index_2d(idx, shape)
            self.assertEqual(expected, list(actual))

    def test_3d(self):
        shape = [64, 32, 16]
        for idx in trange(shape[0] * shape[1] * shape[2]):
            expected = torch.unravel_index(torch.tensor(idx), shape)
            expected = [e.item() for e in expected]
            actual = unravel_index_3d(idx, shape)
            self.assertEqual(expected, list(actual))
