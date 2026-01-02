"""
Wave Network for Vision (Image Classification)

Extends the Wave Network architecture to process images by treating
image patches as a sequence, similar to Vision Transformer (ViT).

Key differences from text Wave Network:
- Patch embedding instead of token embedding
- 2D learnable position embeddings
- CLS token for classification

References:
- Wave Network: https://arxiv.org/abs/2411.02674
- Vision Transformer: https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn

# Import WaveLayer from the deep wave network module
import sys
from pathlib import Path

# Add parent directory to path for importing wave_network_deep
sys.path.insert(0, str(Path(__file__).parent.parent))
from wave_network_deep import WaveLayer


class WavePatchEmbedding(nn.Module):
    """
    Convert image patches to wave-compatible embeddings.

    Similar to ViT's linear projection but with orthogonal initialization
    compatible with Wave Network's frequency-domain processing.

    Args:
        image_size: Input image size (assumes square images)
        patch_size: Size of each patch (assumes square patches)
        in_channels: Number of input channels (3 for RGB)
        embedding_dim: Dimension of output embeddings
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embedding_dim: int = 384,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"Image size {image_size} must be divisible by patch size {patch_size}"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.embedding_dim = embedding_dim

        # Linear projection with orthogonal init (wave-friendly)
        self.projection = nn.Linear(self.patch_dim, embedding_dim)
        nn.init.orthogonal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Learnable position embeddings (+1 for CLS token)
        self.position_embedding = nn.Embedding(self.num_patches + 1, embedding_dim)
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.uniform_(self.cls_token, -0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch embeddings.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Patch embeddings of shape (batch, num_patches + 1, embedding_dim)
            where +1 is for the CLS token
        """
        batch_size = x.shape[0]

        # Extract patches: (batch, channels, H, W) -> (batch, num_patches, patch_dim)
        # Reshape to (batch, channels, grid_h, patch_h, grid_w, patch_w)
        x = x.reshape(
            batch_size,
            x.shape[1],
            self.image_size // self.patch_size,
            self.patch_size,
            self.image_size // self.patch_size,
            self.patch_size,
        )
        # Permute to (batch, grid_h, grid_w, channels, patch_h, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # Flatten patches: (batch, num_patches, patch_dim)
        x = x.reshape(batch_size, self.num_patches, self.patch_dim)

        # Project to embedding dimension
        x = self.projection(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        positions = torch.arange(self.num_patches + 1, device=x.device)
        x = x + self.position_embedding(positions)

        return x


class WaveVisionNetwork(nn.Module):
    """
    Wave Network for image classification.

    Uses wave-based signal processing to classify images by:
    1. Splitting images into patches
    2. Projecting patches to embeddings
    3. Processing through wave layers (modulation/interference)
    4. Classifying based on CLS token representation

    Args:
        image_size: Input image size (default: 32 for CIFAR)
        patch_size: Size of each patch (default: 4)
        in_channels: Number of input channels (default: 3 for RGB)
        embedding_dim: Dimension of embeddings (default: 384)
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        num_layers: Number of wave layers (default: 3)
        mode: Wave operation mode - "modulation" or "interference"
        eps: Small constant for numerical stability
        dropout: Dropout rate for classification head

    Example:
        >>> model = WaveVisionNetwork(num_classes=10)
        >>> images = torch.randn(8, 3, 32, 32)
        >>> logits = model(images)  # (8, 10)
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embedding_dim: int = 384,
        num_classes: int = 10,
        num_layers: int = 3,
        mode: str = "modulation",
        eps: float = 1e-8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        # Patch embedding
        self.patch_embed = WavePatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )

        # Stack of wave layers (reusing WaveLayer from wave_network_deep.py)
        self.wave_layers = nn.ModuleList(
            [WaveLayer(embedding_dim, mode=mode, eps=eps) for _ in range(num_layers)]
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Layer norm before classifier (like BERT)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision wave network.

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Convert to patch embeddings: (batch, num_patches + 1, embedding_dim)
        x = self.patch_embed(x)

        # Process through wave layers
        for layer in self.wave_layers:
            x = layer(x)

        # Get CLS token representation (first token)
        cls_output = x[:, 0]

        # Normalize and classify
        cls_output = self.norm(cls_output)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get model configuration as a dictionary."""
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "num_patches": self.patch_embed.num_patches,
            "num_parameters": self.count_parameters(),
        }


# Preset configurations for common use cases
WAVE_VISION_CONFIGS = {
    "wave_vision_tiny": {
        "embedding_dim": 192,
        "num_layers": 3,
    },
    "wave_vision_small": {
        "embedding_dim": 384,
        "num_layers": 3,
    },
    "wave_vision_base": {
        "embedding_dim": 768,
        "num_layers": 6,
    },
}


def create_wave_vision(
    config_name: str = "wave_vision_small",
    image_size: int = 32,
    patch_size: int = 4,
    num_classes: int = 10,
    **kwargs,
) -> WaveVisionNetwork:
    """
    Create a Wave Vision model from a preset configuration.

    Args:
        config_name: One of "wave_vision_tiny", "wave_vision_small", "wave_vision_base"
        image_size: Input image size
        patch_size: Size of each patch
        num_classes: Number of output classes
        **kwargs: Additional arguments to override config

    Returns:
        WaveVisionNetwork model
    """
    if config_name not in WAVE_VISION_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(WAVE_VISION_CONFIGS.keys())}"
        )

    config = WAVE_VISION_CONFIGS[config_name].copy()
    config.update(kwargs)

    return WaveVisionNetwork(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        **config,
    )
