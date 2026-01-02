"""
Hybrid CNN-Wave Vision Network

Combines a CNN stem for local feature extraction with Wave layers
for global processing. The CNN provides strong inductive bias for
local patterns (edges, textures) while Wave layers handle global
relationships efficiently.

Architecture:
- CNN Stem: Conv + 2 ResBlocks (32x32 -> 8x8)
- Transition: Flatten + project + CLS token + position embeddings
- Wave Layers: 2x WaveLayer2D for global 2D-aware processing
- Classification Head: LayerNorm + Dropout + Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wave_vision_2d import WaveLayer2D


class ResBlock(nn.Module):
    """
    Residual block with 2 conv layers and skip connection.

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        stride: Stride for first conv (for downsampling)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out)

        return out


class CNNStem(nn.Module):
    """
    CNN stem for initial local feature extraction.

    Takes 32x32 input and produces 8x8 feature maps through
    initial conv + 2 ResBlocks with stride-2 downsampling.

    Args:
        in_channels: Input image channels (default: 3)
        base_channels: Initial channel count (default: 64)
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Initial conv: 32x32x3 -> 32x32x64
        self.conv1 = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)

        # ResBlock 1: 32x32x64 -> 16x16x128
        self.block1 = ResBlock(base_channels, base_channels * 2, stride=2)

        # ResBlock 2: 16x16x128 -> 8x8x256
        self.block2 = ResBlock(base_channels * 2, base_channels * 4, stride=2)

        self.out_channels = base_channels * 4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.block1(x)
        x = self.block2(x)

        return x


class CNNToWave(nn.Module):
    """
    Transition layer from CNN feature maps to Wave sequence format.

    Flattens spatial dimensions, projects to embedding dim,
    prepends CLS token, and adds position embeddings.

    Args:
        cnn_channels: Number of CNN output channels
        embedding_dim: Target embedding dimension
        grid_size: Spatial grid size (e.g., 8 for 8x8)
    """

    def __init__(self, cnn_channels: int, embedding_dim: int, grid_size: int):
        super().__init__()

        self.grid_size = grid_size
        self.num_patches = grid_size**2

        # Project CNN features to embedding dim
        self.projection = nn.Linear(cnn_channels, embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.uniform_(self.cls_token, -0.1, 0.1)

        # Position embeddings (+1 for CLS token)
        self.position_embedding = nn.Embedding(self.num_patches + 1, embedding_dim)
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, H, W) CNN feature maps

        Returns:
            (batch, num_patches + 1, embedding_dim) sequence
        """
        batch_size = x.shape[0]

        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # Project to embedding dim
        x = self.projection(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(positions)

        return x


class CNNWaveVision(nn.Module):
    """
    Hybrid CNN-Wave Vision Network.

    Combines CNN stem for local features with Wave layers for global
    processing. Achieves strong inductive bias from CNN while
    leveraging Wave's efficient global operations.

    Args:
        image_size: Input image size (default: 32)
        in_channels: Input image channels (default: 3)
        num_classes: Number of output classes (default: 10)
        base_channels: CNN stem initial channels (default: 64)
        embedding_dim: Wave layer embedding dim (default: 256)
        num_wave_layers: Number of WaveLayer2D modules (default: 2)
        mode: Wave operation mode (default: "modulation")
        dropout: Dropout rate in classifier (default: 0.2)
        wave_dropout: Dropout between wave layers (default: 0.1)
    """

    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        embedding_dim: int = 256,
        num_wave_layers: int = 2,
        mode: str = "modulation",
        dropout: float = 0.2,
        wave_dropout: float = 0.1,
        patch_size: int = 4,  # Ignored, kept for compatibility with training script
    ):
        super().__init__()

        self.image_size = image_size

        # CNN stem: 32x32 -> 8x8
        self.stem = CNNStem(in_channels=in_channels, base_channels=base_channels)

        # Grid size after CNN stem (2 stride-2 blocks = /4)
        self.grid_size = image_size // 4

        # Transition from CNN to Wave
        self.transition = CNNToWave(
            cnn_channels=self.stem.out_channels,
            embedding_dim=embedding_dim,
            grid_size=self.grid_size,
        )

        # Wave layers for global processing
        self.wave_layers = nn.ModuleList(
            [WaveLayer2D(embedding_dim, self.grid_size, mode=mode) for _ in range(num_wave_layers)]
        )

        # Dropout between wave layers
        self.wave_dropout = nn.Dropout(wave_dropout)

        # Classification head
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, channels, height, width)

        Returns:
            (batch, num_classes) logits
        """
        # CNN stem for local features
        x = self.stem(x)

        # Transition to Wave format
        x = self.transition(x)

        # Wave layers for global processing
        for i, layer in enumerate(self.wave_layers):
            x = layer(x)
            if i < len(self.wave_layers) - 1:
                x = self.wave_dropout(x)

        # Classify from CLS token
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "image_size": self.image_size,
            "grid_size": self.grid_size,
            "num_wave_layers": len(self.wave_layers),
            "num_parameters": self.count_parameters(),
        }
