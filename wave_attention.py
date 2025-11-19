import torch
import torch.nn as nn


class WaveAttention(nn.Module):
    """
    Attention mechanism using wave operations.

    This implements a novel attention mechanism where attention scores are
    computed using wave interference between query and key representations
    in the complex domain.

    Args:
        embedding_dim (int): Dimension of embeddings
        num_heads (int): Number of attention heads (default: 8)
        eps (float): Small constant for numerical stability (default: 1e-8)

    Example:
        >>> attention = WaveAttention(embedding_dim=768, num_heads=8)
        >>> x = torch.randn(2, 128, 768)  # (batch, seq, dim)
        >>> mask = torch.ones(2, 128)
        >>> output = attention(x, mask)  # (2, 128, 768)
    """
    def __init__(self, embedding_dim, num_heads=8, eps=1e-8):
        super(WaveAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.eps = eps

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.out = nn.Linear(embedding_dim, embedding_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def get_global_semantics(self, x):
        """Compute global semantics per dimension"""
        squared = x * x
        return torch.sqrt(torch.sum(squared, dim=-2, keepdim=True))

    def to_complex(self, x, g):
        """Convert to complex representation"""
        eps = self.eps
        ratio = x / (g + eps)
        sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=0, max=1))
        phase = torch.atan2(sqrt_term, ratio)
        return g * torch.exp(1j * phase)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through wave attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)
            attention_mask: Optional mask of shape (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, seq, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Wave-based attention scores
        # Use wave interference between Q and K
        g_q = self.get_global_semantics(Q)
        g_k = self.get_global_semantics(K.transpose(-2, -1))

        c_q = self.to_complex(Q, g_q)
        c_k = self.to_complex(K, g_k)

        # Compute attention as wave interference
        # Shape: (batch, heads, seq_q, seq_k)
        attention_complex = torch.matmul(c_q, c_k.transpose(-2, -1).conj())
        attention_scores = torch.abs(attention_complex)

        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        # Softmax normalization
        attention_probs = torch.softmax(attention_scores / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        ), dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_probs, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        output = self.out(context)

        # Residual connection + layer norm
        output = self.layer_norm(output + x)

        return output


class WaveAttentionNetwork(nn.Module):
    """
    Text classification network using wave attention layers.

    Combines wave attention with traditional embeddings and pooling
    for text classification tasks.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of token embeddings (default: 768)
        num_classes (int): Number of output classes (default: 4)
        num_layers (int): Number of wave attention layers (default: 2)
        num_heads (int): Number of attention heads (default: 8)
        eps (float): Small constant for numerical stability (default: 1e-8)

    Example:
        >>> model = WaveAttentionNetwork(vocab_size=30522, num_classes=2)
        >>> input_ids = torch.randint(0, 30522, (8, 128))
        >>> attention_mask = torch.ones(8, 128)
        >>> output = model(input_ids, attention_mask)  # (8, 2)
    """
    def __init__(self, vocab_size, embedding_dim=768, num_classes=4,
                 num_layers=2, num_heads=8, eps=1e-8):
        super(WaveAttentionNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Stack of wave attention layers
        self.attention_layers = nn.ModuleList([
            WaveAttention(embedding_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.eps = eps

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through wave attention network.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask of shape (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # Embed
        x = self.embedding(input_ids)

        # Process through attention layers
        for layer in self.attention_layers:
            x = layer(x, attention_mask)

        # Pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(x * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        else:
            pooled = torch.mean(x, dim=1)

        # Classify
        return self.classifier(pooled)
