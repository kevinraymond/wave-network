import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, num_classes=4, mode="modulation"):
        super(WaveNetwork, self).__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        
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
        
        # Initialize embeddings in frequency domain
        self.initialize_frequency_embeddings()

    def initialize_frequency_embeddings(self):
        """Initialize embeddings for frequency domain representation"""
        with torch.no_grad():
            # Initialize with uniform distribution
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            
            # Convert to polar form (treating as frequency components)
            magnitude = torch.abs(self.embedding.weight)
            phase = torch.angle(self.embedding.weight.type(torch.complex64))
            
            # Normalize magnitude to unit energy
            magnitude = magnitude / torch.sqrt(torch.sum(magnitude**2, dim=1, keepdim=True))
            
            # Convert back to Cartesian form
            self.embedding.weight.data = magnitude * torch.cos(phase)

    def get_global_semantics_per_dim(self, x):
        """Calculate global semantics per dimension as per paper section 3.1.1"""
        # Square each element (signal energy E = w²j,k)
        squared = x * x
        
        # Sum squared values per dimension (Gk = ||w:,k||2)
        return torch.sqrt(torch.sum(squared, dim=1, keepdim=True))

    def get_phase_components(self, x, global_semantics):
        """Calculate phase components as per section 3.1.2"""
        eps = 1e-8
        # Calculate ratio wj,k/Gk
        ratio = x / (global_semantics + eps)
        
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
        """Wave modulation in frequency domain"""
        # Get global semantics per dimension
        g1 = self.get_global_semantics_per_dim(z1)
        g2 = self.get_global_semantics_per_dim(z2)
        
        # Convert to complex representations
        c1 = self.to_complex_repr(z1, g1)
        c2 = self.to_complex_repr(z2, g2)
        
        # Calculate total signal energy
        total_energy1 = torch.sum(torch.abs(c1)**2, dim=1, keepdim=True)
        total_energy2 = torch.sum(torch.abs(c2)**2, dim=1, keepdim=True)
        
        # Signal modulation in frequency domain
        modulated = c1 * c2
        
        # Energy conservation
        modulated = modulated * torch.sqrt(total_energy1 * total_energy2) / (torch.abs(modulated) + 1e-8)
        
        return modulated

    def wave_interference(self, z1, z2):
        """Wave interference in frequency domain"""
        # Get global semantics per dimension
        g1 = self.get_global_semantics_per_dim(z1)
        g2 = self.get_global_semantics_per_dim(z2)
        
        # Convert to complex representations
        c1 = self.to_complex_repr(z1, g1)
        c2 = self.to_complex_repr(z2, g2)
        
        # Calculate total signal energy
        total_energy = torch.sum(torch.abs(c1)**2 + torch.abs(c2)**2, dim=1, keepdim=True)
        
        # Signal interference in frequency domain
        interfered = c1 + c2
        
        # Energy conservation
        interfered = interfered * torch.sqrt(total_energy) / (torch.abs(interfered) + 1e-8)
        
        return interfered

    def forward(self, input_ids):
        # Get frequency domain embeddings
        embeddings = self.embedding(input_ids)
        
        # Transform in frequency domain
        z1 = self.linear1(embeddings)
        z2 = self.linear2(embeddings)
        
        # Apply wave operation
        if self.mode == "interference":
            output = self.wave_interference(z1, z2)
        else:  # modulation
            output = self.wave_modulation(z1, z2)
        
        # Convert back to magnitude
        output = torch.abs(output)
        output = self.layer_norm(output)
        
        # Global average pooling
        pooled = torch.mean(output, dim=1)
        return self.classifier(pooled)