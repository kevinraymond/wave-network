"""Tests for audio benchmark utilities."""

import pytest
import torch

from benchmarks.audio import (
    LABEL_TO_IDX,
    SPEECH_COMMANDS_LABELS,
    AudioConfig,
    SpecAugment,
    get_input_shape,
)


class TestSpeechCommandsLabels:
    """Tests for label constants."""

    def test_label_count(self):
        """Test we have 35 labels."""
        assert len(SPEECH_COMMANDS_LABELS) == 35

    def test_label_to_idx_mapping(self):
        """Test label to index mapping is correct."""
        assert len(LABEL_TO_IDX) == 35
        for idx, label in enumerate(SPEECH_COMMANDS_LABELS):
            assert LABEL_TO_IDX[label] == idx

    def test_common_labels_present(self):
        """Test common speech commands are present."""
        common = ["yes", "no", "stop", "go", "left", "right", "up", "down"]
        for label in common:
            assert label in SPEECH_COMMANDS_LABELS


class TestAudioConfig:
    """Tests for AudioConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.duration_sec == 1.0
        assert config.n_mels == 128
        assert config.n_fft == 400
        assert config.hop_length == 160
        assert config.representation == "waveform"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AudioConfig(
            sample_rate=22050,
            duration_sec=2.0,
            n_mels=80,
            n_fft=512,
            hop_length=256,
            representation="melspec",
        )
        assert config.sample_rate == 22050
        assert config.duration_sec == 2.0
        assert config.n_mels == 80
        assert config.n_fft == 512
        assert config.hop_length == 256
        assert config.representation == "melspec"

    def test_representations(self):
        """Test all representation types are valid."""
        for rep in ["waveform", "melspec", "stft"]:
            config = AudioConfig(representation=rep)
            assert config.representation == rep


class TestGetInputShape:
    """Tests for get_input_shape function."""

    def test_waveform_shape(self):
        """Test waveform input shape."""
        config = AudioConfig(representation="waveform", sample_rate=16000, duration_sec=1.0)
        shape = get_input_shape(config)
        assert shape == (16000,)

    def test_waveform_shape_custom_duration(self):
        """Test waveform shape with custom duration."""
        config = AudioConfig(representation="waveform", sample_rate=16000, duration_sec=2.0)
        shape = get_input_shape(config)
        assert shape == (32000,)

    def test_melspec_shape(self):
        """Test mel spectrogram input shape."""
        config = AudioConfig(
            representation="melspec",
            sample_rate=16000,
            duration_sec=1.0,
            n_mels=128,
            hop_length=160,
        )
        shape = get_input_shape(config)
        assert shape[0] == 128  # n_mels
        assert shape[1] == 101  # time_frames (16000 / 160 + 1)

    def test_stft_shape(self):
        """Test STFT input shape."""
        config = AudioConfig(
            representation="stft",
            sample_rate=16000,
            duration_sec=1.0,
            n_fft=400,
            hop_length=160,
        )
        shape = get_input_shape(config)
        assert shape[0] == 2  # magnitude and phase channels
        assert shape[1] == 201  # freq_bins (n_fft // 2 + 1)
        assert shape[2] == 101  # time_frames

    def test_invalid_representation(self):
        """Test invalid representation raises error."""
        config = AudioConfig()
        # Manually set invalid representation
        config.representation = "invalid"
        with pytest.raises(ValueError, match="Unknown representation"):
            get_input_shape(config)


class TestSpecAugment:
    """Tests for SpecAugment augmentation."""

    def test_2d_input(self):
        """Test SpecAugment with 2D input (freq, time)."""
        augment = SpecAugment(
            freq_mask_param=10, time_mask_param=5, num_freq_masks=1, num_time_masks=1
        )
        x = torch.ones(64, 100)  # (freq, time)
        out = augment(x)
        assert out.shape == x.shape
        # Should have some zeros from masking
        assert (out == 0).any()

    def test_3d_input(self):
        """Test SpecAugment with 3D input (channels, freq, time)."""
        augment = SpecAugment(
            freq_mask_param=10, time_mask_param=5, num_freq_masks=1, num_time_masks=1
        )
        x = torch.ones(2, 64, 100)  # (channels, freq, time)
        out = augment(x)
        assert out.shape == x.shape
        # Should have some zeros from masking
        assert (out == 0).any()

    def test_preserves_non_masked_regions(self):
        """Test that non-masked regions are preserved."""
        augment = SpecAugment(
            freq_mask_param=5, time_mask_param=5, num_freq_masks=1, num_time_masks=1
        )
        x = torch.ones(64, 100) * 5.0
        out = augment(x)
        # Non-zero values should still be 5.0
        assert torch.allclose(out[out != 0], torch.tensor(5.0))

    def test_no_mask_when_params_zero(self):
        """Test no masking occurs with zero mask count."""
        augment = SpecAugment(
            freq_mask_param=10, time_mask_param=10, num_freq_masks=0, num_time_masks=0
        )
        x = torch.ones(64, 100)
        out = augment(x)
        assert torch.allclose(out, x)

    def test_multiple_masks(self):
        """Test multiple frequency and time masks."""
        augment = SpecAugment(
            freq_mask_param=5, time_mask_param=5, num_freq_masks=3, num_time_masks=3
        )
        x = torch.ones(64, 100)
        out = augment(x)
        # Should have zeros from multiple masks
        zero_count = (out == 0).sum().item()
        assert zero_count > 0

    def test_does_not_modify_input(self):
        """Test SpecAugment doesn't modify input tensor."""
        augment = SpecAugment()
        x = torch.ones(64, 100)
        x_original = x.clone()
        _ = augment(x)
        assert torch.allclose(x, x_original)

    def test_deterministic_with_seed(self):
        """Test SpecAugment is deterministic with same seed."""
        augment = SpecAugment(freq_mask_param=10, time_mask_param=10)
        x = torch.ones(64, 100)

        torch.manual_seed(42)
        out1 = augment(x.clone())

        torch.manual_seed(42)
        out2 = augment(x.clone())

        assert torch.allclose(out1, out2)

    def test_default_params(self):
        """Test default SpecAugment parameters."""
        augment = SpecAugment()
        assert augment.freq_mask_param == 27
        assert augment.time_mask_param == 10
        assert augment.num_freq_masks == 2
        assert augment.num_time_masks == 2
