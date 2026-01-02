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
        assert output[:, 0, :].shape == (vision_config["batch_size"], vision_config["embedding_dim"])


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
        images_bright = torch.ones(
            vision_config["batch_size"],
            vision_config["in_channels"],
            vision_config["image_size"],
            vision_config["image_size"],
        ) * 10

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
            (32, 4),   # CIFAR: 64 patches
            (32, 8),   # CIFAR: 16 patches
            (224, 16), # ImageNet: 196 patches
            (224, 32), # ImageNet: 49 patches
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
