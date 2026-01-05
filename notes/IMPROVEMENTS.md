# Wave Network Improvements & Fixes

This document contains specific, actionable improvements for the Wave Network implementation.

---

## Priority 1: Critical Fixes

### Fix 1: Energy Conservation in Wave Operations

**Problem:** Current energy normalization uses total sequence energy but normalizes per-element, causing mathematical inconsistency.

**File:** `wave_network.py`

**Current Code (lines 73-93):**
```python
def wave_modulation(self, z1, z2):
    g1 = self.get_global_semantics_per_dim(z1)
    g2 = self.get_global_semantics_per_dim(z2)
    c1 = self.to_complex_repr(z1, g1)
    c2 = self.to_complex_repr(z2, g2)

    # ISSUE: This computes total energy across sequence
    total_energy1 = torch.sum(torch.abs(c1)**2, dim=1, keepdim=True)
    total_energy2 = torch.sum(torch.abs(c2)**2, dim=1, keepdim=True)

    modulated = c1 * c2

    # But normalizes per-element (magnitude is per-element)
    modulated = modulated * torch.sqrt(total_energy1 * total_energy2) / (torch.abs(modulated) + 1e-8)

    return modulated
```

**Recommended Fix (Option A - Per-element conservation):**
```python
def wave_modulation(self, z1, z2):
    g1 = self.get_global_semantics_per_dim(z1)
    g2 = self.get_global_semantics_per_dim(z2)
    c1 = self.to_complex_repr(z1, g1)
    c2 = self.to_complex_repr(z2, g2)

    # Multiply complex vectors
    modulated = c1 * c2

    # Per-element magnitude conservation
    # Magnitude of product should be product of magnitudes
    target_magnitude = torch.abs(c1) * torch.abs(c2)
    current_magnitude = torch.abs(modulated)
    modulated = modulated * target_magnitude / (current_magnitude + 1e-8)

    return modulated
```

**Recommended Fix (Option B - Simplified, let backprop handle):**
```python
def wave_modulation(self, z1, z2):
    g1 = self.get_global_semantics_per_dim(z1)
    g2 = self.get_global_semantics_per_dim(z2)
    c1 = self.to_complex_repr(z1, g1)
    c2 = self.to_complex_repr(z2, g2)

    # Simple multiplication - phase adds, magnitude multiplies
    # This is the natural wave modulation
    modulated = c1 * c2

    return modulated
```

**Apply same fix to `wave_interference`:**
```python
def wave_interference(self, z1, z2):
    g1 = self.get_global_semantics_per_dim(z1)
    g2 = self.get_global_semantics_per_dim(z2)
    c1 = self.to_complex_repr(z1, g1)
    c2 = self.to_complex_repr(z2, g2)

    # Simple addition - natural wave interference
    interfered = c1 + c2

    return interfered
```

---

### Fix 2: Use Attention Mask

**Problem:** Padding tokens are included in global semantics calculation, affecting results.

**File:** `wave_network.py`

**Current Code:**
```python
def forward(self, input_ids):
    embeddings = self.embedding(input_ids)
    z1 = self.linear1(embeddings)
    z2 = self.linear2(embeddings)
    # ... wave operations
```

**Fixed Code:**
```python
def forward(self, input_ids, attention_mask=None):
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
        pooled = torch.sum(output * mask, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
    else:
        pooled = torch.mean(output, dim=1)

    return self.classifier(pooled)
```

---

### Fix 3: Simplify Embedding Initialization

**Problem:** Current initialization treats real embeddings as complex, which is conceptually odd.

**File:** `wave_network.py`

**Current Code (lines 31-45):**
```python
def initialize_frequency_embeddings(self):
    with torch.no_grad():
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        magnitude = torch.abs(self.embedding.weight)
        phase = torch.angle(self.embedding.weight.type(torch.complex64))
        magnitude = magnitude / torch.sqrt(torch.sum(magnitude**2, dim=1, keepdim=True))
        self.embedding.weight.data = magnitude * torch.cos(phase)
```

**Simplified Code:**
```python
def initialize_frequency_embeddings(self):
    """Initialize embeddings with uniform distribution (like BERT)"""
    with torch.no_grad():
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
```

---

## Priority 2: Numerical Stability

### Improvement 1: Add Gradient Monitoring

**File:** `train.py`

**Add to Trainer class:**
```python
def check_gradients(self):
    """Check for NaN/Inf in gradients"""
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"WARNING: NaN gradient in {name}")
                return False
            if torch.isinf(param.grad).any():
                print(f"WARNING: Inf gradient in {name}")
                return False
    return True
```

**Add to training loop (after loss.backward()):**
```python
# Backward pass
loss.backward()

# Check gradients
if not self.check_gradients():
    print(f"Skipping update due to invalid gradients")
    self.optimizer.zero_grad()
    continue

# Clip gradients
torch.nn.utils.clip_grad_norm_(...)
```

---

### Improvement 2: Configurable Epsilon

**File:** `wave_network.py`

**Add to __init__:**
```python
def __init__(self, vocab_size, embedding_dim=768, num_classes=4,
             mode="modulation", eps=1e-8):
    super(WaveNetwork, self).__init__()
    self.mode = mode
    self.embedding_dim = embedding_dim
    self.eps = eps  # Make epsilon configurable
    # ...
```

**Update usages:**
```python
def get_phase_components(self, x, global_semantics):
    ratio = x / (global_semantics + self.eps)  # Use self.eps
    sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=0, max=1))
    return sqrt_term, ratio
```

---

## Priority 3: Enhanced Features

### Enhancement 1: Multi-Layer Wave Network

**New File:** `wave_network_deep.py`

```python
import torch
import torch.nn as nn
from wave_network import WaveNetwork


class WaveLayer(nn.Module):
    """Single wave processing layer"""
    def __init__(self, embedding_dim, mode="modulation", eps=1e-8):
        super(WaveLayer, self).__init__()
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
        squared = x * x
        return torch.sqrt(torch.sum(squared, dim=1, keepdim=True))

    def get_phase_components(self, x, global_semantics):
        ratio = x / (global_semantics + self.eps)
        sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=0, max=1))
        return sqrt_term, ratio

    def to_complex_repr(self, x, global_semantics):
        sqrt_term, ratio = self.get_phase_components(x, global_semantics)
        phase = torch.atan2(sqrt_term, ratio)
        return global_semantics * torch.exp(1j * phase)

    def forward(self, x, attention_mask=None):
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
    """Multi-layer Wave Network"""
    def __init__(self, vocab_size, embedding_dim=768, num_classes=4,
                 num_layers=3, mode="modulation", eps=1e-8):
        super(DeepWaveNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Stack of wave layers
        self.wave_layers = nn.ModuleList([
            WaveLayer(embedding_dim, mode=mode, eps=eps)
            for _ in range(num_layers)
        ])

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # Embed
        x = self.embedding(input_ids)

        # Process through wave layers
        for layer in self.wave_layers:
            x = layer(x, attention_mask)

        # Pool
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(x * mask, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
        else:
            pooled = torch.mean(x, dim=1)

        # Classify
        return self.classifier(pooled)
```

---

### Enhancement 2: Learnable Mode Mixing

**File:** `wave_network.py`

**Add to __init__:**
```python
def __init__(self, vocab_size, embedding_dim=768, num_classes=4,
             mode="modulation", learnable_mode=False):
    # ... existing code ...

    # Learnable mixing between modulation and interference
    self.learnable_mode = learnable_mode
    if learnable_mode:
        self.mode_weight = nn.Parameter(torch.tensor(0.5))
```

**Update forward:**
```python
def forward(self, input_ids, attention_mask=None):
    # ... existing embedding code ...

    z1 = self.linear1(embeddings)
    z2 = self.linear2(embeddings)

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
        else:
            output = self.wave_modulation(z1, z2)

    # ... rest of forward pass ...
```

---

### Enhancement 3: Wave Attention Mechanism

**New File:** `wave_attention.py`

```python
import torch
import torch.nn as nn


class WaveAttention(nn.Module):
    """Attention mechanism using wave operations"""
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
```

---

## Priority 4: Testing & Validation

### Test Suite

**New File:** `test_wave_network.py`

```python
import unittest
import torch
from wave_network import WaveNetwork


class TestWaveNetwork(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.num_classes = 4
        self.batch_size = 2
        self.seq_len = 8

    def test_global_semantics_shape(self):
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        g = model.get_global_semantics_per_dim(x)

        expected_shape = (self.batch_size, 1, self.embedding_dim)
        self.assertEqual(g.shape, expected_shape)

    def test_phase_unit_circle(self):
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        g = model.get_global_semantics_per_dim(x)
        sqrt_term, ratio = model.get_phase_components(x, g)

        # Should satisfy: sqrt_term² + ratio² ≈ 1
        circle_check = sqrt_term**2 + ratio**2
        self.assertTrue(torch.allclose(circle_check, torch.ones_like(circle_check), rtol=1e-4))

    def test_forward_pass_shape(self):
        for mode in ["modulation", "interference"]:
            model = WaveNetwork(self.vocab_size, self.embedding_dim,
                              self.num_classes, mode=mode)
            input_ids = torch.randint(0, self.vocab_size,
                                     (self.batch_size, self.seq_len))
            output = model(input_ids)

            expected_shape = (self.batch_size, self.num_classes)
            self.assertEqual(output.shape, expected_shape)

    def test_no_nan_inf(self):
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))
        output = model(input_ids)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_attention_mask(self):
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))

        # Test without mask
        output1 = model(input_ids, attention_mask=None)

        # Test with full mask (all ones)
        mask_full = torch.ones(self.batch_size, self.seq_len)
        output2 = model(input_ids, attention_mask=mask_full)

        # Should be similar (not exact due to floating point)
        self.assertTrue(torch.allclose(output1, output2, rtol=1e-3))

        # Test with partial mask
        mask_partial = torch.ones(self.batch_size, self.seq_len)
        mask_partial[:, self.seq_len//2:] = 0  # Mask second half
        output3 = model(input_ids, attention_mask=mask_partial)

        # Should be different from full sequence
        self.assertFalse(torch.allclose(output1, output3, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
```

---

## Priority 5: Documentation

### Add Docstrings

**File:** `wave_network.py`

```python
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

    Attributes:
        embedding: Token embedding layer
        linear1, linear2: Linear transformations for wave processing
        layer_norm: Layer normalization
        classifier: Final classification layer

    Example:
        >>> model = WaveNetwork(vocab_size=30522, num_classes=2)
        >>> input_ids = torch.randint(0, 30522, (8, 128))  # batch=8, seq_len=128
        >>> output = model(input_ids)  # (8, 2)
    """

    def __init__(self, vocab_size, embedding_dim=768, num_classes=4,
                 mode="modulation", eps=1e-8):
        # ... existing code ...
```

---

## Summary of Changes

### Critical (Do First):
1. ✓ Fix energy conservation in wave operations
2. ✓ Add attention mask support
3. ✓ Simplify embedding initialization

### Important (Do Soon):
4. ✓ Add gradient monitoring
5. ✓ Make epsilon configurable
6. ✓ Add unit tests

### Optional (Future Work):
7. ✓ Implement multi-layer variant
8. ✓ Add learnable mode mixing
9. ✓ Experiment with wave attention
10. ✓ Add comprehensive docstrings

### Testing Priority:
- Run diagnostic.py to verify all operations
- Run test_wave_network.py for unit tests
- Re-train on AG News to verify performance maintained/improved
- Compare results before/after fixes

---

*Document created: 2025-11-19*
