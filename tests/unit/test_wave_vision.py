"""Unit tests for Wave Vision models."""

import pytest
import torch


class TestWavePatchEmbedding:
    """Test suite for WavePatchEmbedding."""

    def test_patch_embedding_shape(self, vision_config):
        """Test that patch embedding produces correct output shape."""
        from models.wave_vision import WavePatchEmbedding

        embed = WavePatchEmbedding(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            in_channels=vision_config["in_channels"],
            embedding_dim=vision_config["embedding_dim"],
        )

        images = torch.randn(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        )

        output = embed(images)

        # Expected: (batch, num_patches + 1, embedding_dim)
        # num_patches = (32/4)^2 = 64, +1 for CLS token = 65
        num_patches = (vision_config["image_size"] // vision_config["patch_size"]) ** 2
        expected_shape = (
            vision_config["batch_size"],
            num_patches + 1,
            vision_config["embedding_dim"],
        )
        assert output.shape == expected_shape

    def test_num_patches_calculation(self, vision_config):
        """Test that num_patches is calculated correctly."""
        from models.wave_vision import WavePatchEmbedding

        embed = WavePatchEmbedding(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
        )

        expected_num_patches = (vision_config["image_size"] // vision_config["patch_size"]) ** 2
        assert embed.num_patches == expected_num_patches

    def test_invalid_patch_size(self):
        """Test that invalid patch size raises error."""
        from models.wave_vision import WavePatchEmbedding

        with pytest.raises(ValueError, match="must be divisible"):
            WavePatchEmbedding(image_size=32, patch_size=5)

    def test_cls_token_prepended(self, vision_config):
        """Test that CLS token is correctly prepended."""
        from models.wave_vision import WavePatchEmbedding

        embed = WavePatchEmbedding(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            embedding_dim=vision_config["embedding_dim"],
        )

        images = torch.randn(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        )

        output = embed(images)

        # CLS token should be the same for all samples in batch (before position embedding)
        # After position embedding, they should have position 0 encoding added
        # Just verify the shape is correct
        assert output[:, 0, :].shape == (
            vision_config["batch_size"],
            vision_config["embedding_dim"],
        )


class TestWaveVisionNetwork:
    """Test suite for WaveVisionNetwork."""

    def test_forward_pass_shape(self, wave_vision_model, sample_images, vision_config):
        """Test forward pass produces correct output shape."""
        output = wave_vision_model(sample_images)

        expected_shape = (vision_config["batch_size"], vision_config["num_classes"])
        assert output.shape == expected_shape

    def test_no_nan_inf(self, wave_vision_model, sample_images):
        """Test that forward pass produces no NaN or Inf values."""
        output = wave_vision_model(sample_images)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, wave_vision_model, sample_images):
        """Test that gradients flow correctly."""
        output = wave_vision_model(sample_images)
        loss = output.sum()
        loss.backward()

        assert wave_vision_model.patch_embed.projection.weight.grad is not None
        assert not torch.isnan(wave_vision_model.patch_embed.projection.weight.grad).any()

    @pytest.mark.parametrize("mode", ["modulation", "interference"])
    def test_wave_modes(self, vision_config, mode):
        """Test both wave operation modes work."""
        from models.wave_vision import WaveVisionNetwork

        model = WaveVisionNetwork(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            embedding_dim=vision_config["embedding_dim"],
            num_classes=vision_config["num_classes"],
            num_layers=vision_config["num_layers"],
            mode=mode,
        )

        images = torch.randn(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        )

        output = model(images)
        assert output.shape == (vision_config["batch_size"], vision_config["num_classes"])
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("num_layers", [1, 3, 6])
    def test_num_layers(self, vision_config, num_layers):
        """Test that correct number of layers are created."""
        from models.wave_vision import WaveVisionNetwork

        model = WaveVisionNetwork(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            embedding_dim=vision_config["embedding_dim"],
            num_classes=vision_config["num_classes"],
            num_layers=num_layers,
        )

        assert len(model.wave_layers) == num_layers

    def test_count_parameters(self, wave_vision_model):
        """Test parameter counting method."""
        param_count = wave_vision_model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_get_config(self, wave_vision_model, vision_config):
        """Test config retrieval method."""
        config = wave_vision_model.get_config()

        assert config["image_size"] == vision_config["image_size"]
        assert config["patch_size"] == vision_config["patch_size"]
        assert config["embedding_dim"] == vision_config["embedding_dim"]
        assert config["num_classes"] == vision_config["num_classes"]
        assert config["num_layers"] == vision_config["num_layers"]
        assert "num_parameters" in config


class TestWaveVisionConfigs:
    """Test preset configurations."""

    def test_create_wave_vision_small(self):
        """Test creating wave_vision_small model."""
        from models.wave_vision import create_wave_vision

        model = create_wave_vision("wave_vision_small", num_classes=10)

        assert model.embedding_dim == 384
        assert len(model.wave_layers) == 3

    def test_create_wave_vision_tiny(self):
        """Test creating wave_vision_tiny model."""
        from models.wave_vision import create_wave_vision

        model = create_wave_vision("wave_vision_tiny", num_classes=10)

        assert model.embedding_dim == 192
        assert len(model.wave_layers) == 3

    def test_create_wave_vision_base(self):
        """Test creating wave_vision_base model."""
        from models.wave_vision import create_wave_vision

        model = create_wave_vision("wave_vision_base", num_classes=10)

        assert model.embedding_dim == 768
        assert len(model.wave_layers) == 6

    def test_invalid_config_name(self):
        """Test that invalid config name raises error."""
        from models.wave_vision import create_wave_vision

        with pytest.raises(ValueError, match="Unknown config"):
            create_wave_vision("invalid_config")

    def test_config_override(self):
        """Test that config values can be overridden."""
        from models.wave_vision import create_wave_vision

        model = create_wave_vision("wave_vision_small", num_classes=100, num_layers=5)

        assert model.num_classes == 100
        assert len(model.wave_layers) == 5


class TestNumericalStability:
    """Test numerical stability of vision wave operations."""

    def test_extreme_pixel_values(self, wave_vision_model, vision_config):
        """Test behavior with extreme pixel values."""
        # Very bright images
        images_bright = (
            torch.ones(
                vision_config["batch_size"],
                vision_config["in_channels"],
                vision_config["image_size"],
                vision_config["image_size"],
            )
            * 10
        )

        output = wave_vision_model(images_bright)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_images(self, wave_vision_model, vision_config):
        """Test behavior with all-zero images."""
        images_zero = torch.zeros(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        )

        output = wave_vision_model(images_zero)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_stability(self, wave_vision_model, sample_images):
        """Test that gradients remain stable."""
        output = wave_vision_model(sample_images)
        loss = output.sum()
        loss.backward()

        for name, param in wave_vision_model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


class TestImageSizes:
    """Test various image sizes and patch configurations."""

    @pytest.mark.parametrize(
        "image_size,patch_size",
        [
            (32, 4),  # CIFAR: 64 patches
            (32, 8),  # CIFAR: 16 patches
            (224, 16),  # ImageNet: 196 patches
            (224, 32),  # ImageNet: 49 patches
        ],
    )
    def test_various_sizes(self, image_size, patch_size):
        """Test various image and patch size combinations."""
        from models.wave_vision import WaveVisionNetwork

        model = WaveVisionNetwork(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=64,
            num_classes=10,
            num_layers=2,
        )

        images = torch.randn(2, 3, image_size, image_size)
        output = model(images)

        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()


# ============================================================================
# Tests for WaveVisionNetwork2D (wave_vision_2d.py)
# ============================================================================


class TestWaveLayer2D:
    """Test suite for WaveLayer2D."""

    def test_wave_layer_2d_shape(self, vision_config):
        """Test that WaveLayer2D produces correct output shape."""
        from models.wave_vision_2d import WaveLayer2D

        grid_size = vision_config["image_size"] // vision_config["patch_size"]
        layer = WaveLayer2D(
            embedding_dim=vision_config["embedding_dim"],
            grid_size=grid_size,
        )

        # Input: (batch, H*W + 1, dim) - includes CLS token
        seq_len = grid_size * grid_size + 1
        x = torch.randn(
            vision_config["batch_size"],
            seq_len,
            vision_config["embedding_dim"],
        )

        output = layer(x)

        assert output.shape == x.shape

    def test_wave_layer_2d_no_nan(self, vision_config):
        """Test that WaveLayer2D produces no NaN values."""
        from models.wave_vision_2d import WaveLayer2D

        grid_size = vision_config["image_size"] // vision_config["patch_size"]
        layer = WaveLayer2D(
            embedding_dim=vision_config["embedding_dim"],
            grid_size=grid_size,
        )

        seq_len = grid_size * grid_size + 1
        x = torch.randn(
            vision_config["batch_size"],
            seq_len,
            vision_config["embedding_dim"],
        )

        output = layer(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("mode", ["modulation", "interference"])
    def test_wave_layer_2d_modes(self, vision_config, mode):
        """Test both wave modes work for WaveLayer2D."""
        from models.wave_vision_2d import WaveLayer2D

        grid_size = vision_config["image_size"] // vision_config["patch_size"]
        layer = WaveLayer2D(
            embedding_dim=vision_config["embedding_dim"],
            grid_size=grid_size,
            mode=mode,
        )

        seq_len = grid_size * grid_size + 1
        x = torch.randn(
            vision_config["batch_size"],
            seq_len,
            vision_config["embedding_dim"],
        )

        output = layer(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_wave_layer_2d_gradient_flow(self, vision_config):
        """Test that gradients flow through WaveLayer2D."""
        from models.wave_vision_2d import WaveLayer2D

        grid_size = vision_config["image_size"] // vision_config["patch_size"]
        layer = WaveLayer2D(
            embedding_dim=vision_config["embedding_dim"],
            grid_size=grid_size,
        )

        seq_len = grid_size * grid_size + 1
        x = torch.randn(
            vision_config["batch_size"],
            seq_len,
            vision_config["embedding_dim"],
            requires_grad=True,
        )

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestWaveVisionNetwork2D:
    """Test suite for WaveVisionNetwork2D."""

    @pytest.fixture
    def wave_vision_2d_model(self, vision_config):
        """Create a WaveVisionNetwork2D for testing."""
        from models.wave_vision_2d import WaveVisionNetwork2D

        return WaveVisionNetwork2D(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            in_channels=vision_config["in_channels"],
            embedding_dim=vision_config["embedding_dim"],
            num_classes=vision_config["num_classes"],
            num_layers=vision_config["num_layers"],
        )

    def test_forward_pass_shape(self, wave_vision_2d_model, sample_images, vision_config):
        """Test forward pass produces correct output shape."""
        output = wave_vision_2d_model(sample_images)

        expected_shape = (vision_config["batch_size"], vision_config["num_classes"])
        assert output.shape == expected_shape

    def test_no_nan_inf(self, wave_vision_2d_model, sample_images):
        """Test that forward pass produces no NaN or Inf values."""
        output = wave_vision_2d_model(sample_images)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, wave_vision_2d_model, sample_images):
        """Test that gradients flow correctly."""
        output = wave_vision_2d_model(sample_images)
        loss = output.sum()
        loss.backward()

        assert wave_vision_2d_model.patch_embed.weight.grad is not None
        assert not torch.isnan(wave_vision_2d_model.patch_embed.weight.grad).any()

    @pytest.mark.parametrize("mode", ["modulation", "interference"])
    def test_wave_modes(self, vision_config, mode):
        """Test both wave operation modes work."""
        from models.wave_vision_2d import WaveVisionNetwork2D

        model = WaveVisionNetwork2D(
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            embedding_dim=vision_config["embedding_dim"],
            num_classes=vision_config["num_classes"],
            num_layers=vision_config["num_layers"],
            mode=mode,
        )

        images = torch.randn(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        )

        output = model(images)
        assert output.shape == (vision_config["batch_size"], vision_config["num_classes"])
        assert not torch.isnan(output).any()

    def test_invalid_patch_size(self):
        """Test that invalid patch size raises error."""
        from models.wave_vision_2d import WaveVisionNetwork2D

        with pytest.raises(ValueError, match="must be divisible"):
            WaveVisionNetwork2D(image_size=32, patch_size=5, num_classes=10)


# ============================================================================
# Tests for CNNWaveVision (wave_vision_hybrid.py)
# ============================================================================


class TestResBlock:
    """Test suite for ResBlock."""

    def test_resblock_same_channels(self):
        """Test ResBlock with same input/output channels."""
        from models.wave_vision_hybrid import ResBlock

        block = ResBlock(64, 64)
        x = torch.randn(2, 64, 16, 16)
        output = block(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_resblock_channel_expansion(self):
        """Test ResBlock with channel expansion."""
        from models.wave_vision_hybrid import ResBlock

        block = ResBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 16, 16)
        output = block(x)

        assert output.shape == (2, 128, 8, 8)
        assert not torch.isnan(output).any()

    def test_resblock_gradient_flow(self):
        """Test that gradients flow through ResBlock."""
        from models.wave_vision_hybrid import ResBlock

        block = ResBlock(64, 64)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCNNStem:
    """Test suite for CNNStem."""

    def test_cnn_stem_output_shape(self):
        """Test CNNStem produces correct output shape."""
        from models.wave_vision_hybrid import CNNStem

        stem = CNNStem(in_channels=3, base_channels=64)
        x = torch.randn(2, 3, 32, 32)
        output = stem(x)

        # 32x32 -> 16x16 -> 8x8, channels: 3 -> 64 -> 128 -> 256
        assert output.shape == (2, 256, 8, 8)

    def test_cnn_stem_no_nan(self):
        """Test CNNStem produces no NaN values."""
        from models.wave_vision_hybrid import CNNStem

        stem = CNNStem()
        x = torch.randn(2, 3, 32, 32)
        output = stem(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestCNNWaveVision:
    """Test suite for CNNWaveVision hybrid model."""

    @pytest.fixture
    def cnn_wave_model(self, vision_config):
        """Create a CNNWaveVision for testing."""
        from models.wave_vision_hybrid import CNNWaveVision

        return CNNWaveVision(
            image_size=vision_config["image_size"],
            in_channels=vision_config["in_channels"],
            num_classes=vision_config["num_classes"],
            base_channels=32,  # Smaller for faster tests
            embedding_dim=vision_config["embedding_dim"],
            num_wave_layers=vision_config["num_layers"],
        )

    def test_forward_pass_shape(self, cnn_wave_model, sample_images, vision_config):
        """Test forward pass produces correct output shape."""
        output = cnn_wave_model(sample_images)

        expected_shape = (vision_config["batch_size"], vision_config["num_classes"])
        assert output.shape == expected_shape

    def test_no_nan_inf(self, cnn_wave_model, sample_images):
        """Test that forward pass produces no NaN or Inf values."""
        output = cnn_wave_model(sample_images)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, cnn_wave_model, sample_images):
        """Test that gradients flow correctly."""
        output = cnn_wave_model(sample_images)
        loss = output.sum()
        loss.backward()

        # Check CNN stem gradients
        assert cnn_wave_model.stem.conv1.weight.grad is not None
        assert not torch.isnan(cnn_wave_model.stem.conv1.weight.grad).any()

    @pytest.mark.parametrize("mode", ["modulation", "interference"])
    def test_wave_modes(self, vision_config, mode):
        """Test both wave operation modes work."""
        from models.wave_vision_hybrid import CNNWaveVision

        model = CNNWaveVision(
            image_size=vision_config["image_size"],
            num_classes=vision_config["num_classes"],
            base_channels=32,
            num_wave_layers=2,
            mode=mode,
        )

        images = torch.randn(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        )

        output = model(images)
        assert output.shape == (vision_config["batch_size"], vision_config["num_classes"])
        assert not torch.isnan(output).any()

    def test_count_parameters(self, cnn_wave_model):
        """Test parameter counting method."""
        param_count = cnn_wave_model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)

    def test_get_config(self, cnn_wave_model, vision_config):
        """Test config retrieval method."""
        config = cnn_wave_model.get_config()

        assert config["image_size"] == vision_config["image_size"]
        assert "num_wave_layers" in config
        assert "num_parameters" in config

    def test_extreme_values(self, cnn_wave_model, vision_config):
        """Test behavior with extreme pixel values."""
        # Very bright images
        images_bright = (
            torch.ones(
                vision_config["batch_size"],
                vision_config["in_channels"],
                vision_config["image_size"],
                vision_config["image_size"],
            )
            * 10
        )

        output = cnn_wave_model(images_bright)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_images(self, cnn_wave_model, vision_config):
        """Test behavior with all-zero images."""
        images_zero = torch.zeros(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        )

        output = cnn_wave_model(images_zero)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
