"""
2D Spatial-Aware Wave Network for Vision

Extends Wave Network with 2D spatial awareness by computing
wave operations along rows, columns, and local neighborhoods
instead of treating patches as a flat sequence.

Key innovations:
1. Row-wise and column-wise global semantics
2. Local 3x3 neighborhood wave operations
3. Multi-scale fusion of global and local features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveLayer2D(nn.Module):
    """
    2D-aware wave processing layer.

    Instead of computing global semantics across all patches,
    computes separately for:
    - Rows (horizontal relationships)
    - Columns (vertical relationships)
    - Local 3x3 neighborhoods (local texture)

    Args:
        embedding_dim: Dimension of embeddings
        grid_size: Size of the patch grid (e.g., 8 for 8x8 grid)
        mode: Wave operation mode ("modulation" or "interference")
        eps: Numerical stability constant
    """

    def __init__(
        self,
        embedding_dim: int,
        grid_size: int,
        mode: str = "modulation",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.mode = mode
        self.eps = eps

        # Projections for different scales
        self.proj_row = nn.Linear(embedding_dim, embedding_dim)
        self.proj_col = nn.Linear(embedding_dim, embedding_dim)
        self.proj_local = nn.Linear(embedding_dim, embedding_dim)

        # Initialize with orthogonal weights
        nn.init.orthogonal_(self.proj_row.weight)
        nn.init.orthogonal_(self.proj_col.weight)
        nn.init.orthogonal_(self.proj_local.weight)
        nn.init.zeros_(self.proj_row.bias)
        nn.init.zeros_(self.proj_col.bias)
        nn.init.zeros_(self.proj_local.bias)

        # Learnable fusion weights for multi-scale combination
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def get_global_semantics(self, x, dim):
        """Calculate global semantics along a specific dimension."""
        squared = x * x
        return torch.sqrt(torch.sum(squared, dim=dim, keepdim=True) + self.eps)

    def to_complex_repr(self, x, global_semantics):
        """Convert to complex polar representation."""
        ratio = x / (global_semantics + self.eps)
        # Clamp to valid range for sqrt
        ratio = torch.clamp(ratio, -1 + self.eps, 1 - self.eps)
        sqrt_term = torch.sqrt(1 - ratio**2)
        phase = torch.atan2(sqrt_term, ratio)
        return global_semantics * torch.exp(1j * phase)

    def wave_op(self, c1, c2):
        """Apply wave operation (modulation or interference)."""
        if self.mode == "modulation":
            return c1 * c2
        else:
            return c1 + c2

    def process_rows(self, x):
        """
        Process patches row-wise.

        For an 8x8 grid, computes global semantics within each row,
        capturing horizontal relationships.

        Args:
            x: (batch, H*W, dim)

        Returns:
            (batch, H*W, dim) with row-wise wave processing
        """
        batch_size = x.shape[0]
        h, w = self.grid_size, self.grid_size

        # Reshape to (batch, H, W, dim)
        x_2d = x.view(batch_size, h, w, -1)

        # Project
        z = self.proj_row(x_2d)

        # Compute row-wise global semantics: (batch, H, 1, dim)
        g = self.get_global_semantics(z, dim=2)

        # Convert to complex and apply wave operation
        c = self.to_complex_repr(z, g)

        # Within each row, modulate adjacent patches
        c_shifted = torch.roll(c, shifts=1, dims=2)
        output = self.wave_op(c, c_shifted)

        # Convert back to real
        output = torch.abs(output)

        # Reshape back to (batch, H*W, dim)
        return output.view(batch_size, h * w, -1)

    def process_cols(self, x):
        """
        Process patches column-wise.

        For an 8x8 grid, computes global semantics within each column,
        capturing vertical relationships.

        Args:
            x: (batch, H*W, dim)

        Returns:
            (batch, H*W, dim) with column-wise wave processing
        """
        batch_size = x.shape[0]
        h, w = self.grid_size, self.grid_size

        # Reshape to (batch, H, W, dim)
        x_2d = x.view(batch_size, h, w, -1)

        # Project
        z = self.proj_col(x_2d)

        # Compute column-wise global semantics: (batch, 1, W, dim)
        g = self.get_global_semantics(z, dim=1)

        # Convert to complex and apply wave operation
        c = self.to_complex_repr(z, g)

        # Within each column, modulate adjacent patches
        c_shifted = torch.roll(c, shifts=1, dims=1)
        output = self.wave_op(c, c_shifted)

        # Convert back to real
        output = torch.abs(output)

        # Reshape back to (batch, H*W, dim)
        return output.view(batch_size, h * w, -1)

    def process_local(self, x):
        """
        Process patches with local 3x3 neighborhoods.

        Uses a depthwise convolution-like operation in the wave domain.

        Args:
            x: (batch, H*W, dim)

        Returns:
            (batch, H*W, dim) with local wave processing
        """
        batch_size = x.shape[0]
        h, w = self.grid_size, self.grid_size
        dim = x.shape[-1]

        # Reshape to (batch, dim, H, W) for spatial operations
        x_2d = x.view(batch_size, h, w, dim).permute(0, 3, 1, 2)

        # Project
        z = self.proj_local(x.view(batch_size, h, w, dim))
        z_2d = z.permute(0, 3, 1, 2)  # (batch, dim, H, W)

        # Pad for 3x3 neighborhood
        z_padded = F.pad(z_2d, (1, 1, 1, 1), mode='replicate')

        # Compute local global semantics using avg pool
        # This gives us the "local magnitude" for each 3x3 region
        local_squared = z_padded ** 2
        local_sum = F.avg_pool2d(local_squared, kernel_size=3, stride=1, padding=0)
        local_g = torch.sqrt(local_sum * 9 + self.eps)  # *9 to undo avg

        # Convert center to complex
        ratio = z_2d / (local_g + self.eps)
        ratio = torch.clamp(ratio, -1 + self.eps, 1 - self.eps)
        sqrt_term = torch.sqrt(1 - ratio**2)
        phase = torch.atan2(sqrt_term, ratio)
        c_center = local_g * torch.exp(1j * phase)

        # Get shifted version (diagonal neighbor)
        z_shifted = torch.roll(torch.roll(z_2d, 1, dims=2), 1, dims=3)
        ratio_shifted = z_shifted / (local_g + self.eps)
        ratio_shifted = torch.clamp(ratio_shifted, -1 + self.eps, 1 - self.eps)
        sqrt_shifted = torch.sqrt(1 - ratio_shifted**2)
        phase_shifted = torch.atan2(sqrt_shifted, ratio_shifted)
        c_neighbor = local_g * torch.exp(1j * phase_shifted)

        # Wave operation
        output = self.wave_op(c_center, c_neighbor)
        output = torch.abs(output)

        # Reshape back to (batch, H*W, dim)
        output = output.permute(0, 2, 3, 1).reshape(batch_size, h * w, dim)
        return output

    def forward(self, x):
        """
        Forward pass with multi-scale 2D wave processing.

        Args:
            x: (batch, seq_len, dim) where seq_len = grid_size^2 + 1 (CLS token)

        Returns:
            (batch, seq_len, dim) with 2D-aware wave processing
        """
        # Separate CLS token from patch tokens
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # Multi-scale processing
        row_out = self.process_rows(patch_tokens)
        col_out = self.process_cols(patch_tokens)
        local_out = self.process_local(patch_tokens)

        # Learnable fusion of scales
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * row_out + weights[1] * col_out + weights[2] * local_out

        # Residual connection for patches
        patch_out = self.layer_norm(fused + patch_tokens)

        # Update CLS token by attending to processed patches (global pooling)
        cls_update = patch_out.mean(dim=1, keepdim=True)
        cls_out = self.layer_norm(cls_token + cls_update)

        # Recombine
        output = torch.cat([cls_out, patch_out], dim=1)

        return output


class WaveVisionNetwork2D(nn.Module):
    """
    Wave Network for vision with 2D spatial awareness.

    Uses WaveLayer2D to process image patches while preserving
    spatial structure through row, column, and local operations.

    Args:
        image_size: Input image size
        patch_size: Size of each patch
        in_channels: Number of input channels
        embedding_dim: Dimension of embeddings
        num_classes: Number of output classes
        num_layers: Number of wave layers
        mode: Wave operation mode
        dropout: Dropout rate
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

        if image_size % patch_size != 0:
            raise ValueError(
                f"Image size {image_size} must be divisible by patch size {patch_size}"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.embedding_dim = embedding_dim

        # Patch embedding (same as before)
        patch_dim = patch_size * patch_size * in_channels
        self.patch_embed = nn.Linear(patch_dim, embedding_dim)
        nn.init.orthogonal_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

        # Position embeddings
        self.position_embedding = nn.Embedding(self.num_patches + 1, embedding_dim)
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.uniform_(self.cls_token, -0.1, 0.1)

        # 2D-aware wave layers
        self.wave_layers = nn.ModuleList([
            WaveLayer2D(embedding_dim, self.grid_size, mode=mode, eps=eps)
            for _ in range(num_layers)
        ])

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
        batch_size = x.shape[0]

        # Extract patches
        x = x.reshape(
            batch_size,
            x.shape[1],
            self.grid_size, self.patch_size,
            self.grid_size, self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(batch_size, self.num_patches, -1)

        # Project to embedding dim
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        positions = torch.arange(self.num_patches + 1, device=x.device)
        x = x + self.position_embedding(positions)

        # 2D-aware wave processing
        for layer in self.wave_layers:
            x = layer(x)

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
            "patch_size": self.patch_size,
            "grid_size": self.grid_size,
            "embedding_dim": self.embedding_dim,
            "num_layers": len(self.wave_layers),
            "num_parameters": self.count_parameters(),
        }
