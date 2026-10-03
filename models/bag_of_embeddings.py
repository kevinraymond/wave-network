"""Bag-of-embeddings baseline for text classification.

The control for Wave Network text results: the same token embedding, layer
norm, masked mean pooling and linear classifier as ``WaveNetwork``, with the
wave operation (and positional encoding) removed. Order-insensitive models of
this kind (e.g. fastText, https://arxiv.org/abs/1607.01759) are already strong
on topic and sentiment classification, so a wave result only shows the wave
operation helps if it beats this model at the same embedding size.
"""

import torch
import torch.nn as nn


class BagOfEmbeddings(nn.Module):
    """Mean-pooled token embeddings followed by a linear classifier.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of token embeddings (default: 768)
        num_classes (int): Number of output classes (default: 4)
        eps (float): Small constant for numerical stability (default: 1e-8)
    """

    def __init__(self, vocab_size, embedding_dim=768, num_classes=4, eps=1e-8):
        super().__init__()
        self.eps = eps

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len)
            attention_mask: Optional attention mask of shape (batch, seq_len)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        output = self.layer_norm(self.embedding(input_ids))

        # Global average pooling - only average over non-masked tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            pooled = torch.sum(output * mask, dim=1) / (torch.sum(mask, dim=1) + self.eps)
        else:
            pooled = torch.mean(output, dim=1)

        return self.classifier(pooled)
