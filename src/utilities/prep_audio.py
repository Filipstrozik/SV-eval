from typing import Union
import torch
from pathlib import Path
import torchaudio


def process_wav_to_stack(
    audio_waveform: torch.Tensor,
    length: int = 4,
    step: int = 1,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """
    This function takes a waveform tensor and returns a stack of audio tensors of length `length` seconds, with a step of `step` seconds.

    Args:
    - audio_waveform: torch.Tensor of shape (1,num_samples)
    - length: int
    - step: int
    - sample_rate: int
    """

    # check if input tensor is 2D (1, num_samples)
    if len(audio_waveform.shape) != 2:
        raise ValueError(
            f"Expected a 2D tensor, got a {len(audio_waveform.shape)}D tensor"
        )

    if length <= 0:
        raise ValueError("The segment length must be greater than 0")

    if step <= 0:
        raise ValueError("The step length must be greater than 0")

    num_samples = audio_waveform.shape[1]
    segment_length = length * sample_rate
    step_length = step * sample_rate

    if num_samples < segment_length:
        raise ValueError(
            f"The audio waveform is too short to create even one segment of length {num_samples}, expected at least {segment_length}"
        )

    audio_segments = [
        audio_waveform[:, i * step_length : i * step_length + segment_length]
        for i in range((num_samples - segment_length) // step_length + 1)
    ]
    # return stack of tensors should be of shape (num_segments, 1, segment_length)
    audio_tensor = torch.stack(audio_segments)
    return audio_tensor


def get_audio_tensor(
    audio: Union[str, Path, torch.Tensor],
    length: int = 4,
    step: int = 1,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """
    This function takes a path to a wav file or a torch tensor and returns a stack of audio tensors of length `length` seconds, with a step of `step` seconds.

    Args:
    - audio: Union[str, Path, torch.Tensor]
    - length: int
    - step: int
    - sample_rate: int
    """
    if isinstance(audio, (str, Path)):
        audio_waveform, sample_rate = torchaudio.load(audio)
    elif isinstance(audio, torch.Tensor):
        audio_waveform = audio
    else:
        raise ValueError("Expected a path to a wav file or a torch tensor")

    audio_tensor = process_wav_to_stack(audio_waveform, length, step, sample_rate)
    return audio_tensor
