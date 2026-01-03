"""
Audio Benchmark: Speech Commands Dataset

Supports three input representations:
1. Raw waveform (1D) - 16,000 samples
2. Mel spectrogram (2D) - 128 mels × ~100 frames
3. Complex STFT - magnitude + phase channels
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR.

    Applies frequency and time masking to spectrograms.
    Reference: https://arxiv.org/abs/1904.08779

    Args:
        freq_mask_param: Maximum frequency mask width (F in paper)
        time_mask_param: Maximum time mask width (T in paper)
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 10,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input tensor.

        Args:
            x: Input tensor of shape (channels, freq, time) or (freq, time)

        Returns:
            Augmented tensor with same shape
        """
        x = x.clone()

        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            freq_dim, time_dim = x.shape
            is_2d = True
            x = x.unsqueeze(0)  # Add channel dim
        else:
            _, freq_dim, time_dim = x.shape
            is_2d = False

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, min(self.freq_mask_param, freq_dim), (1,)).item()
            f0 = torch.randint(0, freq_dim - f + 1, (1,)).item()
            x[:, f0 : f0 + f, :] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param, time_dim), (1,)).item()
            t0 = torch.randint(0, time_dim - t + 1, (1,)).item()
            x[:, :, t0 : t0 + t] = 0

        if is_2d:
            x = x.squeeze(0)

        return x


# Speech Commands labels (v2, 35 classes)
SPEECH_COMMANDS_LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(SPEECH_COMMANDS_LABELS)}


@dataclass
class AudioConfig:
    """Configuration for audio preprocessing."""

    sample_rate: int = 16000
    duration_sec: float = 1.0
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 160
    representation: Literal["waveform", "melspec", "stft"] = "waveform"


class SpeechCommandsDataset(Dataset):
    """
    Speech Commands dataset with multiple representation options.

    Args:
        root: Root directory for dataset
        subset: "training", "validation", or "testing"
        config: AudioConfig for preprocessing
        download: Whether to download if not present
        augment: Optional augmentation transform (e.g., SpecAugment)
    """

    def __init__(
        self,
        root: str = "./data",
        subset: str = "training",
        config: AudioConfig | None = None,
        download: bool = True,
        augment: SpecAugment | None = None,
    ):
        self.config = config or AudioConfig()
        self.target_length = int(self.config.sample_rate * self.config.duration_sec)
        self.augment = augment

        # Load Speech Commands dataset
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=download,
            subset=subset,
        )

        # Filter to only include the 35 main classes
        self.indices = [
            i for i in range(len(self.dataset)) if self.dataset[i][2] in SPEECH_COMMANDS_LABELS
        ]

        # Setup transforms based on representation
        if self.config.representation == "melspec":
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
            )
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        waveform, sample_rate, label, *_ = self.dataset[actual_idx]

        # Resample if necessary
        if sample_rate != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.config.sample_rate)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to target length
        waveform = self._pad_or_trim(waveform)

        # Convert to target representation
        if self.config.representation == "waveform":
            # Shape: (1, 16000) -> squeeze to (16000,) for sequence processing
            x = waveform.squeeze(0)
        elif self.config.representation == "melspec":
            # Shape: (1, n_mels, time_frames)
            mel = self.mel_transform(waveform)
            x = self.amplitude_to_db(mel).squeeze(0)  # (n_mels, time_frames)
        elif self.config.representation == "stft":
            # Complex STFT - return magnitude and phase
            stft = torch.stft(
                waveform.squeeze(0),
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                return_complex=True,
            )
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            x = torch.stack([magnitude, phase], dim=0)  # (2, freq_bins, time_frames)
        else:
            raise ValueError(f"Unknown representation: {self.config.representation}")

        # Apply augmentation if provided (only for spectrogram-like representations)
        if self.augment is not None and self.config.representation in ["melspec", "stft"]:
            x = self.augment(x)

        label_idx = LABEL_TO_IDX[label]
        return x, label_idx

    def _pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or trim waveform to target length."""
        length = waveform.shape[1]
        if length < self.target_length:
            # Pad with zeros
            padding = self.target_length - length
            waveform = F.pad(waveform, (0, padding))
        elif length > self.target_length:
            # Trim
            waveform = waveform[:, : self.target_length]
        return waveform


def get_audio_dataloaders(
    config: AudioConfig | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    root: str = "./data",
    augment: SpecAugment | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders for Speech Commands.

    Args:
        config: AudioConfig for preprocessing
        batch_size: Batch size
        num_workers: Number of data loading workers
        root: Root directory for dataset
        augment: Optional augmentation for training data (e.g., SpecAugment)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = config or AudioConfig()

    # Only apply augmentation to training data
    train_dataset = SpeechCommandsDataset(root, "training", config, augment=augment)
    val_dataset = SpeechCommandsDataset(root, "validation", config)
    test_dataset = SpeechCommandsDataset(root, "testing", config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_input_shape(config: AudioConfig) -> tuple:
    """Get the input shape for a given configuration."""
    if config.representation == "waveform":
        # (sequence_length,)
        return (int(config.sample_rate * config.duration_sec),)
    elif config.representation == "melspec":
        # (n_mels, time_frames)
        time_frames = int(config.sample_rate * config.duration_sec / config.hop_length) + 1
        return (config.n_mels, time_frames)
    elif config.representation == "stft":
        # (2, freq_bins, time_frames) - magnitude and phase
        freq_bins = config.n_fft // 2 + 1
        time_frames = int(config.sample_rate * config.duration_sec / config.hop_length) + 1
        return (2, freq_bins, time_frames)
    else:
        raise ValueError(f"Unknown representation: {config.representation}")


if __name__ == "__main__":
    # Test the dataset
    print("Testing Speech Commands dataset loading...")

    for rep in ["waveform", "melspec", "stft"]:
        print(f"\n{rep.upper()} representation:")
        config = AudioConfig(representation=rep)
        dataset = SpeechCommandsDataset(config=config)
        x, y = dataset[0]
        print(f"  Shape: {x.shape}")
        print(f"  Label: {SPEECH_COMMANDS_LABELS[y]}")
        print(f"  Expected shape: {get_input_shape(config)}")
