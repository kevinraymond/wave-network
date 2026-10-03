import torch
import torch.nn as nn

from models import wave_ops


class WaveNetwork(nn.Module):
    """
    Wave Network for text classification using wave-based signal processing.

    Based on the paper: "Wave Network: An Ultra-Small Language Model"
    (https://arxiv.org/abs/2411.02674)

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of token embeddings (default: 768)
        num_classes (int): Number of output classes (default: 4)
        mode (str): Wave operation mode - "modulation" or "interference" (default: "modulation")
        eps (float): Small constant for numerical stability (default: 1e-8)
        learnable_mode (bool): Enable learnable mixing between modulation and interference (default: False)
        max_seq_len (int): Maximum sequence length for positional encoding (default: 512)
        output_mode (str): How the complex wave output maps back to real features -
            "real_imag", "real", or "abs" (default: "real_imag").

            "abs" reproduces the pre-v0.3.0 behavior but is degenerate under
            modulation: |c1*c2| = G1*G2 is identical for every token, so all
            token-level information is discarded. "real" is the paper's local
            semantics reading but is linear under interference. "real_imag"
            preserves both components in both modes (mirrors WaveLayerSTFT in
            models/wave_audio.py) at the cost of one extra linear projection.

    Attributes:
        embedding: Token embedding layer
        linear1, linear2: Linear transformations for wave processing
        layer_norm: Layer normalization
        classifier: Final classification layer

    Example:
        >>> model = WaveNetwork(vocab_size=30522, num_classes=2)
        >>> input_ids = torch.randint(0, 30522, (8, 128))  # batch=8, seq_len=128
        >>> attention_mask = torch.ones(8, 128)
        >>> output = model(input_ids, attention_mask)  # (8, 2)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=768,
        num_classes=4,
        mode="modulation",
        eps=1e-8,
        learnable_mode=False,
        max_seq_len=512,
        output_mode="real_imag",
    ):
        super().__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.learnable_mode = learnable_mode
        self.max_seq_len = max_seq_len
        self.output_mode = output_mode

        # Token embedding in frequency domain
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional encoding (as per paper Figure 6a)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim) * 0.02)

        # Frequency domain transformations
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

        # Orthogonal initialization for frequency preservation
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

        # real_imag doubles the feature dimension; project back down
        if output_mode == "real_imag":
            self.out_proj = nn.Linear(2 * embedding_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Learnable mixing between modulation and interference
        if learnable_mode:
            self.mode_weight = nn.Parameter(torch.tensor(0.5))

        # Initialize embeddings in frequency domain
        self.initialize_frequency_embeddings()

    def initialize_frequency_embeddings(self):
        """Initialize embeddings with uniform distribution (like BERT)"""
        with torch.no_grad():
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def get_global_semantics_per_dim(self, x):
        """Calculate global semantics per dimension as per paper section 3.1.1"""
        return wave_ops.global_semantics(x, dim=1, eps=self.eps)

    def get_phase_components(self, x, global_semantics):
        """Calculate phase components as per section 3.1.2"""
        return wave_ops.phase_components(x, global_semantics, self.eps)

    def to_complex_repr(self, x, global_semantics):
        """Convert to complex representation in polar form"""
        return wave_ops.to_complex(x, global_semantics, self.eps)

    def _superpose(self, z1, z2, mode):
        """Convert both inputs to complex representations and combine them."""
        g1 = self.get_global_semantics_per_dim(z1)
        g2 = self.get_global_semantics_per_dim(z2)
        c1 = self.to_complex_repr(z1, g1)
        c2 = self.to_complex_repr(z2, g2)
        return wave_ops.wave_superpose(c1, c2, mode)

    def wave_modulation(self, z1, z2):
        """
        Wave modulation in frequency domain.

        Multiplies two complex wave representations. In complex multiplication,
        phases add and magnitudes multiply naturally - this is the correct
        wave modulation behavior.
        """
        return self._superpose(z1, z2, "modulation")

    def wave_interference(self, z1, z2):
        """
        Wave interference in frequency domain.

        Adds two complex wave representations. Wave interference naturally
        occurs through complex addition where phases interact constructively
        or destructively.
        """
        return self._superpose(z1, z2, "interference")

    def _from_complex(self, c):
        """Map the complex wave output back to real features."""
        output = wave_ops.from_complex(c, self.output_mode)
        if self.output_mode == "real_imag":
            output = self.out_proj(output)
        return output

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the Wave Network.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask of shape (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Get embeddings
        embeddings = self.embedding(input_ids)  # (batch, seq, emb)

        # Add positional encoding (as per paper Figure 6a)
        seq_len = embeddings.size(1)
        embeddings = embeddings + self.pos_embed[:, :seq_len, :]

        # Transform in frequency domain
        z1 = self.linear1(embeddings)
        z2 = self.linear2(embeddings)

        # Mask after projection: the linear biases would otherwise let padded
        # positions contribute to the global semantics norms
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            z1 = z1 * mask
            z2 = z2 * mask

        # Apply wave operation
        if self.learnable_mode:
            # Compute both operations
            modulated = self.wave_modulation(z1, z2)
            interfered = self.wave_interference(z1, z2)

            # Learnable mixing
            alpha = torch.sigmoid(self.mode_weight)
            output = alpha * modulated + (1 - alpha) * interfered
        else:
            output = self._superpose(z1, z2, self.mode)

        # Convert back to real features
        output = self._from_complex(output)
        output = self.layer_norm(output)

        # Global average pooling - only average over non-masked tokens
        if attention_mask is not None:
            # Sum over sequence, then divide by actual lengths
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            pooled = torch.sum(output * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        else:
            pooled = torch.mean(output, dim=1)

        return self.classifier(pooled)
