import unittest
import torch
from src.utils import repeat_or_cut_wave


class TestRepeatOrCutWave(unittest.TestCase):
    def test_repeat_no_remainder(self):
        data = torch.tensor([[1, 2, 3]])
        max_len = 6
        expected_output = torch.tensor([[1, 2, 3, 1, 2, 3]])
        output = repeat_or_cut_wave(data, max_len)
        self.assertTrue(torch.equal(output, expected_output))

    def test_repeat_with_remainder(self):
        data = torch.tensor([[1, 2, 3]])
        max_len = 7
        expected_output = torch.tensor([[1, 2, 3, 1, 2, 3, 1]])
        output = repeat_or_cut_wave(data, max_len)
        self.assertTrue(torch.equal(output, expected_output))

    def test_cut_wave(self):
        data = torch.tensor([[1, 2, 3, 4, 5, 6]])
        max_len = 3
        expected_output = torch.tensor([[1, 2, 3]])
        output = repeat_or_cut_wave(data, max_len)
        self.assertTrue(torch.equal(output, expected_output))

    def test_equal_length(self):
        data = torch.tensor([[1, 2, 3]])
        max_len = 3
        expected_output = torch.tensor([[1, 2, 3]])
        output = repeat_or_cut_wave(data, max_len)
        self.assertTrue(torch.equal(output, expected_output))


if __name__ == "__main__":
    unittest.main()
