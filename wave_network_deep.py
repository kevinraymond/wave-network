import torch
import torch.nn as nn


class WaveLayer(nn.Module):
    """Single wave processing layer"""

    def __init__(self, embedding_dim, mode="modulation", eps=1e-8):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.embedding_dim = embedding_dim

        # Two linear projections
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

        # Orthogonal initialization
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def get_global_semantics_per_dim(self, x):
        """Calculate global semantics per dimension"""
        squared = x * x
        return torch.sqrt(torch.sum(squared, dim=1, keepdim=True))

    def get_phase_components(self, x, global_semantics):
        """Calculate phase components"""
        ratio = x / (global_semantics + self.eps)
        sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=0, max=1))
        return sqrt_term, ratio

    def to_complex_repr(self, x, global_semantics):
        """Convert to complex representation"""
        sqrt_term, ratio = self.get_phase_components(x, global_semantics)
        phase = torch.atan2(sqrt_term, ratio)
        return global_semantics * torch.exp(1j * phase)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through wave layer.

        Args:
            x: Input tensor of shape (batch, seq, dim)
            attention_mask: Optional mask of shape (batch, seq)

        Returns:
            Output tensor of shape (batch, seq, dim)
        """
        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask

        # Project
        z1 = self.linear1(x)
        z2 = self.linear2(x)

        # Wave operation
        g1 = self.get_global_semantics_per_dim(z1)
        g2 = self.get_global_semantics_per_dim(z2)
        c1 = self.to_complex_repr(z1, g1)
        c2 = self.to_complex_repr(z2, g2)

        if self.mode == "modulation":
            output = c1 * c2
        else:  # interference
            output = c1 + c2

        # Convert to real
        output = torch.abs(output)

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
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Stack of wave layers
        self.wave_layers = nn.ModuleList(
            [WaveLayer(embedding_dim, mode=mode, eps=eps) for _ in range(num_layers)]
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
