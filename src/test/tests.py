import unittest
import torch
from pathlib import Path
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utilities.prep_audio import process_wav_to_stack, get_audio_tensor


class TestProcessWavToStack(unittest.TestCase):
    def test_valid_tensor(self):
        audio_waveform = torch.randn(1, 64000)  # 4 seconds at 16kHz
        length = 4
        step = 1
        sample_rate = 16000
        result = process_wav_to_stack(audio_waveform, length, step, sample_rate)
        self.assertEqual(result.shape, (1, 1, 64000))

    def test_invalid_tensor(self):
        audio_waveform = torch.randn(64000)  # 1D tensor
        length = 4
        step = 1
        sample_rate = 16000
        with self.assertRaises(ValueError):
            process_wav_to_stack(audio_waveform, length, step, sample_rate)

    def test_too_short_tensor(self):
        audio_waveform = torch.randn(1, 32000)  # 2 seconds at 16kHz
        length = 4
        step = 1
        sample_rate = 16000
        with self.assertRaises(ValueError):
            process_wav_to_stack(audio_waveform, length, step, sample_rate)

    def test_length_2_seconds(self):
        audio_waveform = torch.randn(1, 32000)  # 2 seconds at 16kHz
        length = 2
        step = 1
        sample_rate = 16000
        result = process_wav_to_stack(audio_waveform, length, step, sample_rate)
        self.assertEqual(result.shape, (1, 1, 32000))

    def test_length_6_seconds(self):
        audio_waveform = torch.randn(1, 96000)  # 6 seconds at 16kHz
        length = 6
        step = 1
        sample_rate = 16000
        result = process_wav_to_stack(audio_waveform, length, step, sample_rate)
        self.assertEqual(result.shape, (1, 1, 96000))

    def test_length_0_seconds(self):
        audio_waveform = torch.randn(1, 64000)  # 4 seconds at 16kHz
        length = 0
        step = 1
        sample_rate = 16000
        with self.assertRaises(ValueError):
            process_wav_to_stack(audio_waveform, length, step, sample_rate)

    def test_step_0_seconds(self):
        audio_waveform = torch.randn(1, 64000)
        length = 4
        step = 0
        sample_rate = 16000
        with self.assertRaises(ValueError):
            process_wav_to_stack(audio_waveform, length, step, sample_rate)


class TestGetAudioTensor(unittest.TestCase):
    @patch("torchaudio.load")
    def test_valid_file_path(self, mock_load):
        mock_load.return_value = (torch.randn(1, 64000), 16000)  # 4 seconds at 16kHz
        audio_path = "dummy/path.wav"
        length = 4
        step = 1
        sample_rate = 16000
        result = get_audio_tensor(audio_path, length, step, sample_rate)
        self.assertEqual(result.shape, (1, 1, 64000))

    def test_valid_tensor(self):
        audio_waveform = torch.randn(1, 64000)  # 4 seconds at 16kHz
        length = 4
        step = 1
        sample_rate = 16000
        result = get_audio_tensor(audio_waveform, length, step, sample_rate)
        self.assertEqual(result.shape, (1, 1, 64000))

    def test_invalid_input(self):
        audio_waveform = [1, 2, 3]  # Invalid input type
        length = 4
        step = 1
        sample_rate = 16000
        with self.assertRaises(ValueError):
            get_audio_tensor(audio_waveform, length, step, sample_rate)


if __name__ == "__main__":
    unittest.main()
