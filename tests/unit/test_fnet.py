"""Unit tests for FNet implementation."""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.fnet import FNet, FNetLite, FNetEncoderBlock, FourierMixing


class TestFourierMixing:
    """Test suite for FourierMixing layer."""

    def test_forward_shape(self):
        """Test that Fourier mixing preserves shape."""
        mixing = FourierMixing()
        x = torch.randn(2, 8, 64)
        output = mixing(x)

        assert output.shape == x.shape

    def test_forward_real_output(self):
        """Test that output is real (not complex)."""
        mixing = FourierMixing()
        x = torch.randn(2, 8, 64)
        output = mixing(x)

        assert output.dtype in [torch.float32, torch.float64]
        assert not output.is_complex()

    def test_no_learnable_params(self):
        """Test that FourierMixing has no learnable parameters."""
        mixing = FourierMixing()
        params = list(mixing.parameters())

        assert len(params) == 0


class TestFNetEncoderBlock:
    """Test suite for FNet encoder block."""

    def test_forward_shape(self):
        """Test that encoder block preserves shape."""
        block = FNetEncoderBlock(hidden_dim=64)
        x = torch.randn(2, 8, 64)
        output = block(x)

        assert output.shape == x.shape

    def test_forward_no_nan(self):
        """Test that forward pass produces no NaN values."""
        block = FNetEncoderBlock(hidden_dim=64)
        x = torch.randn(2, 8, 64)
        output = block(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_residual_connection(self):
        """Test that output differs from input (residual + processing)."""
        block = FNetEncoderBlock(hidden_dim=64)
        x = torch.randn(2, 8, 64)
        output = block(x)

        assert not torch.allclose(output, x)

    def test_mask_parameter_accepted(self):
        """Test that attention_mask parameter is accepted (for API compatibility)."""
        block = FNetEncoderBlock(hidden_dim=64)
        x = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8)

        # Should not raise
        output = block(x, attention_mask=mask)
        assert output.shape == x.shape


class TestFNet:
    """Test suite for FNet model."""

    @pytest.fixture
    def config(self):
        """Small config for fast tests."""
        return {
            "vocab_size": 1000,
            "embedding_dim": 64,
            "num_classes": 4,
            "num_layers": 2,
            "batch_size": 2,
            "seq_len": 8,
        }

    def test_forward_shape(self, config):
        """Test that FNet produces correct output shape."""
        model = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )
        input_ids = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))
        output = model(input_ids)

        expected_shape = (config["batch_size"], config["num_classes"])
        assert output.shape == expected_shape

    def test_forward_no_nan(self, config):
        """Test that forward pass produces no NaN values."""
        model = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )
        input_ids = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))
        output = model(input_ids)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_with_attention_mask(self, config):
        """Test that attention mask affects output."""
        model = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )
        input_ids = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))

        # Full mask
        mask_full = torch.ones(config["batch_size"], config["seq_len"])
        output_full = model(input_ids, attention_mask=mask_full)

        # Partial mask
        mask_partial = torch.ones(config["batch_size"], config["seq_len"])
        mask_partial[:, config["seq_len"] // 2:] = 0
        output_partial = model(input_ids, attention_mask=mask_partial)

        # Outputs should differ due to different pooling
        assert not torch.allclose(output_full, output_partial)

    def test_backward_pass(self, config):
        """Test that gradients flow correctly."""
        model = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )
        input_ids = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))

        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        # Check embeddings have gradients
        assert model.token_embedding.weight.grad is not None
        assert not torch.isnan(model.token_embedding.weight.grad).any()

    def test_get_num_params(self, config):
        """Test parameter counting."""
        model = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )
        num_params = model.get_num_params()

        # Should be positive
        assert num_params > 0

        # Manually verify
        expected = sum(p.numel() for p in model.parameters())
        assert num_params == expected


class TestFNetLite:
    """Test suite for FNetLite model."""

    @pytest.fixture
    def config(self):
        """Small config for fast tests."""
        return {
            "vocab_size": 1000,
            "embedding_dim": 64,
            "num_classes": 4,
            "num_layers": 2,
            "batch_size": 2,
            "seq_len": 8,
        }

    def test_forward_shape(self, config):
        """Test that FNetLite produces correct output shape."""
        model = FNetLite(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )
        input_ids = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))
        output = model(input_ids)

        expected_shape = (config["batch_size"], config["num_classes"])
        assert output.shape == expected_shape

    def test_fewer_params_than_fnet(self, config):
        """Test that FNetLite has fewer parameters than FNet."""
        fnet = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )
        fnet_lite = FNetLite(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=config["num_layers"],
        )

        fnet_params = sum(p.numel() for p in fnet.parameters())
        fnet_lite_params = sum(p.numel() for p in fnet_lite.parameters())

        assert fnet_lite_params < fnet_params


class TestFNetVsWaveNetwork:
    """Comparison tests between FNet and Wave Network."""

    @pytest.fixture
    def config(self):
        """Shared config for comparison."""
        return {
            "vocab_size": 1000,
            "embedding_dim": 64,
            "num_classes": 4,
            "batch_size": 2,
            "seq_len": 8,
        }

    def test_same_interface(self, config):
        """Test that FNet and WaveNetwork have the same interface."""
        from wave_network import WaveNetwork

        fnet = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=2,
        )
        wave = WaveNetwork(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
        )

        input_ids = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))
        mask = torch.ones(config["batch_size"], config["seq_len"])

        # Both should accept same inputs
        fnet_out = fnet(input_ids, attention_mask=mask)
        wave_out = wave(input_ids, attention_mask=mask)

        # Both should produce same output shape
        assert fnet_out.shape == wave_out.shape

    def test_different_outputs(self, config):
        """Test that FNet and WaveNetwork produce different outputs."""
        from wave_network import WaveNetwork

        # Use same random seed for embeddings
        torch.manual_seed(42)
        fnet = FNet(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            num_layers=2,
        )

        torch.manual_seed(42)
        wave = WaveNetwork(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
        )

        input_ids = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))

        fnet_out = fnet(input_ids)
        wave_out = wave(input_ids)

        # Should produce different outputs (different architectures)
        assert not torch.allclose(fnet_out, wave_out)
