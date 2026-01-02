"""Pytest configuration and shared fixtures for Wave Network tests."""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(params=["modulation", "interference"])
def wave_mode(request):
    """Parametrized fixture for wave operation modes."""
    return request.param


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config():
    """Small model configuration for fast tests."""
    return {
        "vocab_size": 1000,
        "embedding_dim": 64,
        "num_classes": 4,
        "batch_size": 2,
        "seq_len": 8,
    }


@pytest.fixture
def standard_config():
    """Standard model configuration matching BERT tokenizer."""
    return {
        "vocab_size": 30522,
        "embedding_dim": 768,
        "num_classes": 4,
        "batch_size": 4,
        "seq_len": 128,
    }


@pytest.fixture
def sample_input(small_config):
    """Generate sample input tensors for testing."""
    return torch.randint(
        0, small_config["vocab_size"], (small_config["batch_size"], small_config["seq_len"])
    )


@pytest.fixture
def sample_batch(small_config):
    """Generate a complete sample batch with inputs, mask, and labels."""
    batch_size = small_config["batch_size"]
    seq_len = small_config["seq_len"]
    vocab_size = small_config["vocab_size"]
    num_classes = small_config["num_classes"]

    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, num_classes, (batch_size,)),
    }


@pytest.fixture
def partial_mask(small_config):
    """Create a partial attention mask (first half active)."""
    batch_size = small_config["batch_size"]
    seq_len = small_config["seq_len"]
    mask = torch.ones(batch_size, seq_len)
    mask[:, seq_len // 2 :] = 0
    return mask


@pytest.fixture
def wave_network_model(small_config):
    """Create a small WaveNetwork for testing."""
    from wave_network import WaveNetwork

    return WaveNetwork(
        vocab_size=small_config["vocab_size"],
        embedding_dim=small_config["embedding_dim"],
        num_classes=small_config["num_classes"],
    )


@pytest.fixture
def deep_wave_network_model(small_config):
    """Create a small DeepWaveNetwork for testing."""
    from wave_network_deep import DeepWaveNetwork

    return DeepWaveNetwork(
        vocab_size=small_config["vocab_size"],
        embedding_dim=small_config["embedding_dim"],
        num_classes=small_config["num_classes"],
        num_layers=3,
    )


@pytest.fixture
def wave_attention_model(small_config):
    """Create a WaveAttention module for testing."""
    from wave_attention import WaveAttention

    return WaveAttention(
        embedding_dim=small_config["embedding_dim"],
        num_heads=8,
    )


# Vision model fixtures
@pytest.fixture
def vision_config():
    """Small vision model configuration for fast tests."""
    return {
        "image_size": 32,
        "patch_size": 4,
        "in_channels": 3,
        "embedding_dim": 64,
        "num_classes": 10,
        "num_layers": 2,
        "batch_size": 2,
    }


@pytest.fixture
def sample_images(vision_config):
    """Generate sample image tensors for testing."""
    return torch.randn(
        vision_config["batch_size"],
        vision_config["in_channels"],
        vision_config["image_size"],
        vision_config["image_size"],
    )


@pytest.fixture
def wave_vision_model(vision_config):
    """Create a small WaveVisionNetwork for testing."""
    from models.wave_vision import WaveVisionNetwork

    return WaveVisionNetwork(
        image_size=vision_config["image_size"],
        patch_size=vision_config["patch_size"],
        in_channels=vision_config["in_channels"],
        embedding_dim=vision_config["embedding_dim"],
        num_classes=vision_config["num_classes"],
        num_layers=vision_config["num_layers"],
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
