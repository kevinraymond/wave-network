import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, num_classes=4):
        super(WaveNetwork, self).__init__()

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Linear layers to generate two variants of complex vector representations
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def get_global_semantics(self, embeddings):
        # Calculate magnitude (global semantics)
        return torch.norm(embeddings, dim=1, keepdim=True)

    def get_phase(self, embeddings, global_semantics):
        # Calculate phase using arctan2
        return torch.atan2(
            torch.sqrt(1 - (embeddings / global_semantics) ** 2),
            embeddings / global_semantics,
        )

    def wave_modulation(self, z1, z2):
        # Perform wave modulation (complex multiplication)
        magnitude = torch.abs(z1) * torch.abs(z2)
        phase = torch.angle(z1) + torch.angle(z2)
        return magnitude * torch.exp(1j * phase)

    def forward(self, input_ids):
        # Get initial embeddings
        embeddings = self.embedding(input_ids)

        # Generate two variants of complex representations
        z1 = self.linear1(embeddings)
        z2 = self.linear2(embeddings)

        # Convert to complex representations
        g1 = self.get_global_semantics(z1)
        g2 = self.get_global_semantics(z2)
        p1 = self.get_phase(z1, g1)
        p2 = self.get_phase(z2, g2)

        # Convert to complex numbers
        c1 = g1 * torch.exp(1j * p1)
        c2 = g2 * torch.exp(1j * p2)

        # Perform wave modulation
        output = self.wave_modulation(c1, c2)

        # Convert back to real domain and normalize
        output = torch.abs(output)
        output = self.layer_norm(output)

        # Pool and classify
        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits
