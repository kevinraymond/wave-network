"""
Wave Audio Network: Wave-based models for audio classification.

Supports three input representations:
1. Raw waveform (1D) - processes 16k samples as sequence
2. Mel spectrogram (2D) - processes time-frequency representation
3. Complex STFT - processes magnitude and phase separately
"""

import torch
import torch.nn as nn


class WaveAudio1D(nn.Module):
    """
    Wave Network for raw waveform audio classification.

    Uses 1D convolutions to create patch embeddings from raw audio,
    then applies wave-based processing.

    Args:
        num_classes: Number of output classes (default: 35 for Speech Commands)
        input_length: Length of input waveform (default: 16000 for 1 sec @ 16kHz)
        patch_size: Size of each audio patch (default: 160 = 10ms @ 16kHz)
        embedding_dim: Dimension of embeddings (default: 256)
        num_layers: Number of wave processing layers (default: 4)
        mode: Wave operation mode - "modulation" or "interference"
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        num_classes: int = 35,
        input_length: int = 16000,
        patch_size: int = 160,
        embedding_dim: int = 256,
        num_layers: int = 4,
        mode: str = "modulation",
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.eps = eps

        # Number of patches
        self.num_patches = input_length // patch_size

        # Patch embedding: 1D conv to convert raw audio to embeddings
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim) * 0.02)

        # Wave processing layers
        self.layers = nn.ModuleList(
            [WaveLayer1D(embedding_dim, mode, dropout, eps) for _ in range(num_layers)]
        )

        # Classification head
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input waveform of shape (batch, length) or (batch, 1, length)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Ensure (batch, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Patch embedding: (batch, 1, length) -> (batch, embed_dim, num_patches)
        x = self.patch_embed(x)

        # Transpose to (batch, num_patches, embed_dim)
        x = x.transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply wave layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classify
        x = self.norm(x)
        return self.classifier(x)


class WaveLayer1D(nn.Module):
    """Single wave processing layer for 1D sequences."""

    def __init__(self, embedding_dim: int, mode: str, dropout: float, eps: float):
        super().__init__()
        self.mode = mode
        self.eps = eps

        # Linear projections for wave operation
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

        # Initialize orthogonally
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)

    def get_global_semantics(self, x):
        """Global semantics per dimension."""
        squared = x * x
        return torch.sqrt(torch.sum(squared, dim=1, keepdim=True) + self.eps)

    def to_complex(self, x, g):
        """Convert to complex representation."""
        ratio = x / (g + self.eps)
        ratio = torch.clamp(ratio, -1 + self.eps, 1 - self.eps)
        sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=self.eps))
        phase = torch.atan2(sqrt_term, ratio)
        return g * torch.exp(1j * phase)

    def wave_op(self, z1, z2):
        """Apply wave operation."""
        g1 = self.get_global_semantics(z1)
        g2 = self.get_global_semantics(z2)
        c1 = self.to_complex(z1, g1)
        c2 = self.to_complex(z2, g2)

        if self.mode == "interference":
            result = c1 + c2
        else:  # modulation
            result = c1 * c2

        return torch.abs(result)

    def forward(self, x):
        # Wave sublayer
        residual = x
        x = self.norm(x)
        z1 = self.linear1(x)
        z2 = self.linear2(x)
        x = self.wave_op(z1, z2)
        x = self.dropout(x)
        x = residual + x

        # FFN sublayer
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class WaveAudio2D(nn.Module):
    """
    Wave Network for spectrogram-based audio classification.

    Processes mel spectrograms or STFT magnitude using 2D wave operations.

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (1 for melspec, 2 for STFT mag+phase)
        freq_bins: Number of frequency bins
        time_frames: Number of time frames
        patch_size: Size of patches (freq, time)
        embedding_dim: Dimension of embeddings
        num_layers: Number of wave processing layers
        mode: Wave operation mode
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_classes: int = 35,
        input_channels: int = 1,
        freq_bins: int = 128,
        time_frames: int = 101,
        patch_size: tuple = (16, 4),
        embedding_dim: int = 256,
        num_layers: int = 4,
        mode: str = "modulation",
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.eps = eps

        # Calculate grid size
        self.grid_h = freq_bins // patch_size[0]
        self.grid_w = time_frames // patch_size[1]
        self.num_patches = self.grid_h * self.grid_w

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim) * 0.02)

        # Wave processing layers
        self.layers = nn.ModuleList(
            [WaveLayer1D(embedding_dim, mode, dropout, eps) for _ in range(num_layers)]
        )

        # Classification head
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Spectrogram of shape (batch, channels, freq, time)
               or (batch, freq, time) for single channel

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Ensure 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, grid_h, grid_w)

        # Flatten spatial dims
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply wave layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classify
        x = self.norm(x)
        return self.classifier(x)


class WaveAudioSTFT(nn.Module):
    """
    Wave Network that processes STFT magnitude and phase separately.

    This architecture is most aligned with wave theory - it explicitly
    processes the phase information that wave networks are designed for.

    Args:
        num_classes: Number of output classes
        freq_bins: Number of frequency bins
        time_frames: Number of time frames
        patch_size: Size of patches (freq, time)
        embedding_dim: Dimension of embeddings
        num_layers: Number of wave processing layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_classes: int = 35,
        freq_bins: int = 201,
        time_frames: int = 101,
        patch_size: tuple = (8, 4),
        embedding_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps

        # Grid dimensions
        self.grid_h = freq_bins // patch_size[0]
        self.grid_w = time_frames // patch_size[1]
        self.num_patches = self.grid_h * self.grid_w

        # Separate embeddings for magnitude and phase
        self.mag_embed = nn.Conv2d(1, embedding_dim // 2, patch_size, stride=patch_size)
        self.phase_embed = nn.Conv2d(1, embedding_dim // 2, patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim) * 0.02)

        # Wave layers - uses actual complex numbers with the embedded mag/phase
        self.layers = nn.ModuleList(
            [WaveLayerComplex(embedding_dim, dropout, eps) for _ in range(num_layers)]
        )

        # Classification head
        self.norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: STFT output of shape (batch, 2, freq, time)
               where channel 0 is magnitude, channel 1 is phase

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Split magnitude and phase
        mag = x[:, 0:1, :, :]  # (batch, 1, freq, time)
        phase = x[:, 1:2, :, :]

        # Embed separately
        mag_emb = self.mag_embed(mag)  # (batch, embed_dim/2, grid_h, grid_w)
        phase_emb = self.phase_embed(phase)

        # Concatenate
        x = torch.cat([mag_emb, phase_emb], dim=1)  # (batch, embed_dim, grid_h, grid_w)

        # Flatten
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply wave layers
        for layer in self.layers:
            x = layer(x)

        # Pool and classify
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.classifier(x)


class WaveLayerComplex(nn.Module):
    """
    Wave layer that works with magnitude/phase embedded representations.

    Treats the first half of embedding as magnitude-derived,
    second half as phase-derived, and combines them.
    """

    def __init__(self, embedding_dim: int, dropout: float, eps: float):
        super().__init__()
        self.eps = eps
        self.half_dim = embedding_dim // 2

        # Projections
        self.mag_proj = nn.Linear(embedding_dim, self.half_dim)
        self.phase_proj = nn.Linear(embedding_dim, self.half_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Wave sublayer with explicit mag/phase
        residual = x
        x = self.norm(x)

        # Project to mag and phase components
        mag = torch.abs(self.mag_proj(x)) + self.eps
        phase = self.phase_proj(x)

        # Create complex representation and multiply (wave modulation)
        # Use learned magnitude and phase
        c = mag * torch.exp(1j * phase)

        # Global pooling in complex domain for interaction
        c_global = c.mean(dim=1, keepdim=True)
        c_modulated = c * c_global.conj()

        # Back to real
        output = torch.cat([c_modulated.real, c_modulated.imag], dim=-1)
        output = self.out_proj(output)
        output = self.dropout(output)
        x = residual + output

        # FFN sublayer
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


def create_wave_audio_model(
    representation: str,
    num_classes: int = 35,
    embedding_dim: int = 256,
    num_layers: int = 4,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create appropriate wave audio model.

    Args:
        representation: "waveform", "melspec", or "stft"
        num_classes: Number of output classes
        embedding_dim: Embedding dimension
        num_layers: Number of layers
        **kwargs: Additional arguments for specific model

    Returns:
        Wave audio model
    """
    if representation == "waveform":
        return WaveAudio1D(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            **kwargs,
        )
    elif representation == "melspec":
        return WaveAudio2D(
            num_classes=num_classes,
            input_channels=1,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            **kwargs,
        )
    elif representation == "stft":
        return WaveAudioSTFT(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")


if __name__ == "__main__":
    # Test models
    print("Testing Wave Audio models...")

    # Test 1D waveform model
    print("\n1D Waveform model:")
    model_1d = WaveAudio1D(num_classes=35)
    x = torch.randn(4, 16000)
    out = model_1d(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_1d.parameters()):,}")

    # Test 2D melspec model
    print("\n2D Melspec model:")
    model_2d = WaveAudio2D(num_classes=35, freq_bins=128, time_frames=101)
    x = torch.randn(4, 128, 101)
    out = model_2d(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_2d.parameters()):,}")

    # Test STFT model
    print("\nSTFT model:")
    model_stft = WaveAudioSTFT(num_classes=35, freq_bins=201, time_frames=101)
    x = torch.randn(4, 2, 201, 101)
    out = model_stft(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_stft.parameters()):,}")
