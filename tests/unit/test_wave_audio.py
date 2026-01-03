"""Tests for Wave Audio models."""

import pytest
import torch

from models.wave_audio import (
    WaveAudio1D,
    WaveAudio2D,
    WaveAudioSTFT,
    WaveAudioSTFTExport,
    WaveLayer1D,
    WaveLayerComplex,
    WaveLayerReal,
    create_wave_audio_model,
)


class TestWaveLayer1D:
    """Tests for WaveLayer1D."""

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        layer = WaveLayer1D(embedding_dim=64, mode="modulation", dropout=0.1, eps=1e-8)
        x = torch.randn(2, 10, 64)  # (batch, seq, embed)
        out = layer(x)
        assert out.shape == x.shape

    def test_modes(self):
        """Test both wave modes work."""
        for mode in ["modulation", "interference"]:
            layer = WaveLayer1D(embedding_dim=64, mode=mode, dropout=0.1, eps=1e-8)
            x = torch.randn(2, 10, 64)
            out = layer(x)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_no_nan_output(self):
        """Test layer doesn't produce NaN."""
        layer = WaveLayer1D(embedding_dim=64, mode="modulation", dropout=0.1, eps=1e-8)
        layer.eval()
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestWaveLayerComplex:
    """Tests for WaveLayerComplex."""

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        layer = WaveLayerComplex(embedding_dim=64, dropout=0.1, eps=1e-8, mode="modulation")
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_modes(self):
        """Test both wave modes work."""
        for mode in ["modulation", "interference"]:
            layer = WaveLayerComplex(embedding_dim=64, dropout=0.1, eps=1e-8, mode=mode)
            x = torch.randn(2, 10, 64)
            out = layer(x)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_no_nan_output(self):
        """Test layer doesn't produce NaN."""
        layer = WaveLayerComplex(embedding_dim=64, dropout=0.1, eps=1e-8)
        layer.eval()
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestWaveLayerReal:
    """Tests for WaveLayerReal (ONNX-compatible)."""

    def test_forward_shape(self):
        """Test output shape matches input shape."""
        layer = WaveLayerReal(embedding_dim=64, dropout=0.1, eps=1e-8, mode="modulation")
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_modes(self):
        """Test both wave modes work."""
        for mode in ["modulation", "interference"]:
            layer = WaveLayerReal(embedding_dim=64, dropout=0.1, eps=1e-8, mode=mode)
            x = torch.randn(2, 10, 64)
            out = layer(x)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_equivalent_to_complex(self):
        """Test WaveLayerReal produces similar results to WaveLayerComplex."""
        torch.manual_seed(42)

        # Create layers with same weights
        complex_layer = WaveLayerComplex(embedding_dim=64, dropout=0.0, eps=1e-8, mode="modulation")
        real_layer = WaveLayerReal(embedding_dim=64, dropout=0.0, eps=1e-8, mode="modulation")

        # Copy weights
        real_layer.load_state_dict(complex_layer.state_dict())

        complex_layer.eval()
        real_layer.eval()

        x = torch.randn(2, 10, 64)

        with torch.no_grad():
            complex_out = complex_layer(x)
            real_out = real_layer(x)

        # Should be very close (not exact due to float precision)
        assert torch.allclose(complex_out, real_out, rtol=1e-4, atol=1e-5)


class TestWaveAudio1D:
    """Tests for WaveAudio1D model."""

    def test_forward_shape(self):
        """Test output shape."""
        model = WaveAudio1D(num_classes=35, input_length=16000, embedding_dim=64, num_layers=2)
        x = torch.randn(2, 16000)
        out = model(x)
        assert out.shape == (2, 35)

    def test_forward_3d_input(self):
        """Test with 3D input (batch, 1, length)."""
        model = WaveAudio1D(num_classes=35, input_length=16000, embedding_dim=64, num_layers=2)
        x = torch.randn(2, 1, 16000)
        out = model(x)
        assert out.shape == (2, 35)

    def test_no_nan_output(self):
        """Test model doesn't produce NaN."""
        model = WaveAudio1D(num_classes=35, embedding_dim=64, num_layers=2)
        model.eval()
        x = torch.randn(2, 16000)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_backward_pass(self):
        """Test gradients flow properly."""
        model = WaveAudio1D(num_classes=35, embedding_dim=64, num_layers=2)
        x = torch.randn(2, 16000)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestWaveAudio2D:
    """Tests for WaveAudio2D model."""

    def test_forward_shape(self):
        """Test output shape."""
        model = WaveAudio2D(
            num_classes=35, freq_bins=128, time_frames=100, embedding_dim=64, num_layers=2
        )
        x = torch.randn(2, 128, 100)
        out = model(x)
        assert out.shape == (2, 35)

    def test_forward_4d_input(self):
        """Test with 4D input (batch, channels, freq, time)."""
        model = WaveAudio2D(
            num_classes=35,
            input_channels=1,
            freq_bins=128,
            time_frames=100,
            embedding_dim=64,
            num_layers=2,
        )
        x = torch.randn(2, 1, 128, 100)
        out = model(x)
        assert out.shape == (2, 35)

    def test_no_nan_output(self):
        """Test model doesn't produce NaN."""
        model = WaveAudio2D(num_classes=35, freq_bins=128, time_frames=100, embedding_dim=64)
        model.eval()
        x = torch.randn(2, 128, 100)
        out = model(x)
        assert not torch.isnan(out).any()


class TestWaveAudioSTFT:
    """Tests for WaveAudioSTFT model."""

    def test_forward_shape(self):
        """Test output shape."""
        model = WaveAudioSTFT(
            num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64, num_layers=2
        )
        x = torch.randn(2, 2, 201, 101)  # (batch, 2, freq, time) - mag and phase
        out = model(x)
        assert out.shape == (2, 35)

    def test_no_nan_output(self):
        """Test model doesn't produce NaN."""
        model = WaveAudioSTFT(num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64)
        model.eval()
        x = torch.randn(2, 2, 201, 101)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_modes(self):
        """Test both wave modes work."""
        for mode in ["modulation", "interference"]:
            model = WaveAudioSTFT(
                num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64, mode=mode
            )
            x = torch.randn(2, 2, 201, 101)
            out = model(x)
            assert out.shape == (2, 35)
            assert not torch.isnan(out).any()

    def test_backward_pass(self):
        """Test gradients flow properly."""
        model = WaveAudioSTFT(num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64)
        x = torch.randn(2, 2, 201, 101)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestWaveAudioSTFTExport:
    """Tests for WaveAudioSTFTExport (ONNX-compatible) model."""

    def test_forward_shape(self):
        """Test output shape."""
        model = WaveAudioSTFTExport(
            num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64, num_layers=2
        )
        x = torch.randn(2, 2, 201, 101)
        out = model(x)
        assert out.shape == (2, 35)

    def test_equivalent_to_stft(self):
        """Test WaveAudioSTFTExport produces same results as WaveAudioSTFT."""
        torch.manual_seed(42)

        # Create models
        stft_model = WaveAudioSTFT(
            num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64, num_layers=2
        )
        export_model = WaveAudioSTFTExport(
            num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64, num_layers=2
        )

        # Copy weights
        export_model.load_state_dict(stft_model.state_dict())

        stft_model.eval()
        export_model.eval()

        x = torch.randn(1, 2, 201, 101)

        with torch.no_grad():
            stft_out = stft_model(x)
            export_out = export_model(x)

        # Should be very close
        assert torch.allclose(stft_out, export_out, rtol=1e-4, atol=1e-5)

    def test_no_complex_ops(self):
        """Verify no complex tensor operations (for ONNX compatibility)."""
        model = WaveAudioSTFTExport(
            num_classes=35, freq_bins=201, time_frames=101, embedding_dim=64, num_layers=2
        )
        model.eval()
        x = torch.randn(1, 2, 201, 101)

        # Trace the model - this would fail if complex ops are used
        try:
            traced = torch.jit.trace(model, x)
            assert traced is not None
        except Exception as e:
            pytest.fail(f"Model contains operations incompatible with tracing: {e}")


class TestCreateWaveAudioModel:
    """Tests for the factory function."""

    def test_waveform_model(self):
        """Test creating waveform model."""
        model = create_wave_audio_model("waveform", num_classes=35, embedding_dim=64)
        assert isinstance(model, WaveAudio1D)

    def test_melspec_model(self):
        """Test creating melspec model."""
        model = create_wave_audio_model(
            "melspec", num_classes=35, embedding_dim=64, freq_bins=128, time_frames=100
        )
        assert isinstance(model, WaveAudio2D)

    def test_stft_model(self):
        """Test creating STFT model."""
        model = create_wave_audio_model(
            "stft", num_classes=35, embedding_dim=64, freq_bins=201, time_frames=101
        )
        assert isinstance(model, WaveAudioSTFT)

    def test_invalid_representation(self):
        """Test invalid representation raises error."""
        with pytest.raises(ValueError, match="Unknown representation"):
            create_wave_audio_model("invalid")

    def test_custom_params(self):
        """Test custom parameters are passed through."""
        model = create_wave_audio_model(
            "waveform",
            num_classes=10,
            embedding_dim=128,
            num_layers=6,
            mode="interference",
            dropout=0.2,
        )
        assert model.embedding_dim == 128
        assert model.num_layers == 6
        assert model.mode == "interference"
