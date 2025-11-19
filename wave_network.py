import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, vocab_size, embedding_dim=768, num_classes=4, mode="modulation",
                 eps=1e-8, learnable_mode=False):
        super(WaveNetwork, self).__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.learnable_mode = learnable_mode
        
        # Token embedding in frequency domain
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Frequency domain transformations
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        
        # Orthogonal initialization for frequency preservation
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.orthogonal_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        
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
        # Square each element (signal energy E = w²j,k)
        squared = x * x
        
        # Sum squared values per dimension (Gk = ||w:,k||2)
        return torch.sqrt(torch.sum(squared, dim=1, keepdim=True))

    def get_phase_components(self, x, global_semantics):
        """Calculate phase components as per section 3.1.2"""
        # Calculate ratio wj,k/Gk
        ratio = x / (global_semantics + self.eps)
        
        # Calculate sqrt(1-(wj,k/Gk)²) term
        sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=0, max=1))
        
        # Return both components for arctan2
        return sqrt_term, ratio

    def to_complex_repr(self, x, global_semantics):
        """Convert to complex representation in polar form"""
        sqrt_term, ratio = self.get_phase_components(x, global_semantics)
        phase = torch.atan2(sqrt_term, ratio)
        return global_semantics * torch.exp(1j * phase)

    def wave_modulation(self, z1, z2):
        """
        Wave modulation in frequency domain.

        Multiplies two complex wave representations. In complex multiplication,
        phases add and magnitudes multiply naturally - this is the correct
        wave modulation behavior.
        """
        # Get global semantics per dimension
        g1 = self.get_global_semantics_per_dim(z1)
        g2 = self.get_global_semantics_per_dim(z2)

        # Convert to complex representations
        c1 = self.to_complex_repr(z1, g1)
        c2 = self.to_complex_repr(z2, g2)

        # Simple multiplication - phase adds, magnitude multiplies
        # This is the natural wave modulation
        modulated = c1 * c2

        return modulated

    def wave_interference(self, z1, z2):
        """
        Wave interference in frequency domain.

        Adds two complex wave representations. Wave interference naturally
        occurs through complex addition where phases interact constructively
        or destructively.
        """
        # Get global semantics per dimension
        g1 = self.get_global_semantics_per_dim(z1)
        g2 = self.get_global_semantics_per_dim(z2)

        # Convert to complex representations
        c1 = self.to_complex_repr(z1, g1)
        c2 = self.to_complex_repr(z2, g2)

        # Simple addition - natural wave interference
        interfered = c1 + c2

        return interfered

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

        # Apply attention mask to embeddings before processing
        if attention_mask is not None:
            # Expand mask to embedding dimension
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            embeddings = embeddings * mask

        # Transform in frequency domain
        z1 = self.linear1(embeddings)
        z2 = self.linear2(embeddings)

        # Apply wave operation
        if self.learnable_mode:
            # Compute both operations
            modulated = self.wave_modulation(z1, z2)
            interfered = self.wave_interference(z1, z2)

            # Learnable mixing
            alpha = torch.sigmoid(self.mode_weight)
            output = alpha * modulated + (1 - alpha) * interfered
        else:
            # Fixed mode
            if self.mode == "interference":
                output = self.wave_interference(z1, z2)
            else:  # modulation
                output = self.wave_modulation(z1, z2)

        # Convert back to magnitude
        output = torch.abs(output)
        output = self.layer_norm(output)

        # Global average pooling - only average over non-masked tokens
        if attention_mask is not None:
            # Sum over sequence, then divide by actual lengths
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            pooled = torch.sum(output * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        else:
            pooled = torch.mean(output, dim=1)

        return self.classifier(pooled)