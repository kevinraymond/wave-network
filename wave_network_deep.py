import torch
import torch.nn as nn

from models import wave_ops


class WaveLayer(nn.Module):
    """Single wave processing layer.

    Args:
        embedding_dim (int): Feature dimension
        mode (str): "modulation" or "interference" (default: "modulation")
        eps (float): Numerical stability constant (default: 1e-8)
        output_mode (str): Complex-to-real conversion - "real_imag", "real", or
            "abs" (default: "real_imag"; see WaveNetwork for the trade-offs)
    """

    def __init__(self, embedding_dim, mode="modulation", eps=1e-8, output_mode="real_imag"):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.embedding_dim = embedding_dim
        self.output_mode = output_mode

        # Two linear projections
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

        # Orthogonal initialization
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

        if output_mode == "real_imag":
            self.out_proj = nn.Linear(2 * embedding_dim, embedding_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def get_global_semantics_per_dim(self, x):
        """Calculate global semantics per dimension"""
        return wave_ops.global_semantics(x, dim=1, eps=self.eps)

    def get_phase_components(self, x, global_semantics):
        """Calculate phase components"""
        return wave_ops.phase_components(x, global_semantics, self.eps)

    def to_complex_repr(self, x, global_semantics):
        """Convert to complex representation"""
        return wave_ops.to_complex(x, global_semantics, self.eps)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through wave layer.

        Args:
            x: Input tensor of shape (batch, seq, dim)
            attention_mask: Optional mask of shape (batch, seq)

        Returns:
            Output tensor of shape (batch, seq, dim)
        """
        # Project
        z1 = self.linear1(x)
        z2 = self.linear2(x)

        # Mask after projection so the linear biases don't leak padded
        # positions into the global semantics norms
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            z1 = z1 * mask
            z2 = z2 * mask

        # Wave operation
        g1 = self.get_global_semantics_per_dim(z1)
        g2 = self.get_global_semantics_per_dim(z2)
        c1 = self.to_complex_repr(z1, g1)
        c2 = self.to_complex_repr(z2, g2)
        output = wave_ops.wave_superpose(c1, c2, self.mode)

        # Convert to real
        output = wave_ops.from_complex(output, self.output_mode)
        if self.output_mode == "real_imag":
            output = self.out_proj(output)

        # Residual connection + layer norm
        output = self.layer_norm(output + x)

        return output


class DeepWaveNetwork(nn.Module):
    """
    Multi-layer Wave Network for text classification.

    Extends the basic Wave Network with multiple stacked wave processing layers
    for increased representational capacity.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of token embeddings (default: 768)
        num_classes (int): Number of output classes (default: 4)
        num_layers (int): Number of wave layers to stack (default: 3)
        mode (str): Wave operation mode - "modulation" or "interference" (default: "modulation")
        eps (float): Small constant for numerical stability (default: 1e-8)
        max_seq_len (int): Maximum sequence length for positional encoding (default: 512)
        output_mode (str): Complex-to-real conversion - "real_imag", "real", or
            "abs" (default: "real_imag"; see WaveNetwork for the trade-offs)

    Example:
        >>> model = DeepWaveNetwork(vocab_size=30522, num_classes=2, num_layers=4)
        >>> input_ids = torch.randint(0, 30522, (8, 128))
        >>> attention_mask = torch.ones(8, 128)
        >>> output = model(input_ids, attention_mask)  # (8, 2)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=768,
        num_classes=4,
        num_layers=3,
        mode="modulation",
        eps=1e-8,
        max_seq_len=512,
        output_mode="real_imag",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Positional encoding (as per paper Figure 6a)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim) * 0.02)

        # Stack of wave layers
        self.wave_layers = nn.ModuleList(
            [
                WaveLayer(embedding_dim, mode=mode, eps=eps, output_mode=output_mode)
                for _ in range(num_layers)
            ]
        )

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.eps = eps

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the deep wave network.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask of shape (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Embed
        x = self.embedding(input_ids)

        # Add positional encoding (as per paper Figure 6a)
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]

        # Process through wave layers
        for layer in self.wave_layers:
            x = layer(x, attention_mask)

        # Pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(x * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        else:
            pooled = torch.mean(x, dim=1)

        # Classify
        return self.classifier(pooled)
