"""
FNet: Mixing Tokens with Fourier Transforms

Implementation of FNet (arXiv:2105.03824) as a baseline for comparison
with Wave Network. FNet replaces self-attention with unparameterized
2D Fourier transforms for token mixing.

Key insights from the paper:
- 2D FFT achieves 92-97% of BERT accuracy on GLUE
- Trains 80% faster on GPUs, 70% faster on TPUs
- No learnable parameters in the mixing layer
- Particularly efficient at smaller model sizes
"""

import torch
import torch.fft
import torch.nn as nn


class FourierMixing(nn.Module):
    """
    Fourier-based token mixing layer.

    Replaces self-attention with 2D Fourier transforms:
    - First FFT along sequence dimension (token mixing)
    - Second FFT along hidden dimension (channel mixing)
    - Take real part of the result

    This has no learnable parameters - mixing is purely based on FFT.
    """

    def forward(self, x):
        """
        Apply 2D Fourier transform for token mixing.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of same shape with mixed representations
        """
        # Apply 2D FFT and take real part
        # fft2 applies FFT along last two dimensions (seq_len, hidden_dim)
        return torch.fft.fft2(x).real


class FeedForward(nn.Module):
    """
    Feed-forward network used after Fourier mixing.

    Standard transformer-style FFN with:
    - Expansion layer (hidden_dim -> ffn_dim)
    - GELU activation
    - Projection layer (ffn_dim -> hidden_dim)
    - Dropout
    """

    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.1):
        """
        Args:
            hidden_dim: Input and output dimension
            ffn_dim: Intermediate dimension (default: 4 * hidden_dim)
            dropout: Dropout rate
        """
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim

        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass through FFN."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class FNetEncoderBlock(nn.Module):
    """
    Single FNet encoder block.

    Structure:
    1. Fourier sublayer with pre-norm and residual connection
    2. Feed-forward sublayer with pre-norm and residual connection

    Uses pre-normalization (norm before sublayer) as in the original paper.
    """

    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.1):
        """
        Args:
            hidden_dim: Hidden dimension
            ffn_dim: FFN intermediate dimension (default: 4 * hidden_dim)
            dropout: Dropout rate
        """
        super().__init__()

        self.fourier_mixing = FourierMixing()
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through encoder block.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional mask (not used in Fourier mixing, kept for API compatibility)

        Returns:
            Output tensor of same shape
        """
        # Fourier sublayer with pre-norm and residual
        residual = x
        x = self.norm1(x)
        x = self.fourier_mixing(x)
        x = self.dropout(x)
        x = residual + x

        # FFN sublayer with pre-norm and residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class FNet(nn.Module):
    """
    FNet model for text classification.

    Architecture:
    - Token embedding layer
    - Position embedding (learned)
    - Stack of FNet encoder blocks
    - Global average pooling (with mask support)
    - Classification head

    This implementation matches the Wave Network API for fair comparison:
    - Same input format: (input_ids, attention_mask)
    - Same output format: (batch, num_classes) logits

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of token embeddings (default: 768)
        num_classes: Number of output classes (default: 4)
        num_layers: Number of encoder blocks (default: 6)
        ffn_dim: FFN intermediate dimension (default: 4 * embedding_dim)
        max_seq_len: Maximum sequence length for position embeddings (default: 512)
        dropout: Dropout rate (default: 0.1)
        eps: Small constant for numerical stability (default: 1e-8)

    Example:
        >>> model = FNet(vocab_size=30522, num_classes=4)
        >>> input_ids = torch.randint(0, 30522, (8, 128))
        >>> attention_mask = torch.ones(8, 128)
        >>> output = model(input_ids, attention_mask)  # (8, 4)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=768,
        num_classes=4,
        num_layers=6,
        ffn_dim=None,
        max_seq_len=512,
        dropout=0.1,
        eps=1e-8,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.eps = eps

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Initialize embeddings
        nn.init.uniform_(self.token_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)

        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)

        # Encoder layers
        if ffn_dim is None:
            ffn_dim = 4 * embedding_dim

        self.encoder_layers = nn.ModuleList(
            [FNetEncoderBlock(embedding_dim, ffn_dim, dropout) for _ in range(num_layers)]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through FNet.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask of shape (batch, seq_len)
                           Values: 1 for real tokens, 0 for padding

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        batch_size, seq_len = input_ids.shape

        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = token_embeds + position_embeds
        x = self.embedding_dropout(x)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # Final norm
        x = self.final_norm(x)

        # Pooling with mask support
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(x * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        else:
            pooled = torch.mean(x, dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def get_num_params(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class FNetLite(nn.Module):
    """
    Lightweight FNet variant for faster experimentation.

    Differences from full FNet:
    - Fewer default layers (3 vs 6)
    - Smaller FFN multiplier (2x vs 4x)
    - No position embeddings (simpler)

    Good for quick ablations and memory-constrained settings.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=768,
        num_classes=4,
        num_layers=3,
        ffn_multiplier=2,
        dropout=0.1,
        eps=1e-8,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.eps = eps

        # Simple embedding (no position)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Encoder layers with smaller FFN
        ffn_dim = ffn_multiplier * embedding_dim
        self.encoder_layers = nn.ModuleList(
            [FNetEncoderBlock(embedding_dim, ffn_dim, dropout) for _ in range(num_layers)]
        )

        # Final norm
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through FNetLite."""
        # Embed
        x = self.embedding(input_ids)

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # Final norm
        x = self.final_norm(x)

        # Pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(x * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        else:
            pooled = torch.mean(x, dim=1)

        # Classification
        return self.classifier(pooled)
