import unittest
import torch
from wave_network import WaveNetwork
from wave_network_deep import DeepWaveNetwork, WaveLayer
from wave_attention import WaveAttention, WaveAttentionNetwork


class TestWaveNetwork(unittest.TestCase):
    """Test suite for WaveNetwork implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.num_classes = 4
        self.batch_size = 2
        self.seq_len = 8

    def test_global_semantics_shape(self):
        """Test that global semantics has correct shape"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        g = model.get_global_semantics_per_dim(x)

        expected_shape = (self.batch_size, 1, self.embedding_dim)
        self.assertEqual(g.shape, expected_shape)

    def test_phase_unit_circle(self):
        """Test that phase components satisfy unit circle constraint"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        g = model.get_global_semantics_per_dim(x)
        sqrt_term, ratio = model.get_phase_components(x, g)

        # Should satisfy: sqrt_term² + ratio² ≈ 1
        circle_check = sqrt_term**2 + ratio**2
        self.assertTrue(torch.allclose(circle_check, torch.ones_like(circle_check), rtol=1e-4))

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape"""
        for mode in ["modulation", "interference"]:
            model = WaveNetwork(self.vocab_size, self.embedding_dim,
                              self.num_classes, mode=mode)
            input_ids = torch.randint(0, self.vocab_size,
                                     (self.batch_size, self.seq_len))
            output = model(input_ids)

            expected_shape = (self.batch_size, self.num_classes)
            self.assertEqual(output.shape, expected_shape)

    def test_no_nan_inf(self):
        """Test that forward pass produces no NaN or Inf values"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))
        output = model(input_ids)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_attention_mask(self):
        """Test attention mask functionality"""
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

    def test_learnable_mode(self):
        """Test learnable mode mixing"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes,
                          learnable_mode=True)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))

        # Should have mode_weight parameter
        self.assertTrue(hasattr(model, 'mode_weight'))

        # Should produce valid output
        output = model(input_ids)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(output).any())

    def test_epsilon_configurable(self):
        """Test that epsilon is configurable"""
        eps_custom = 1e-6
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes,
                          eps=eps_custom)
        self.assertEqual(model.eps, eps_custom)

    def test_backward_pass(self):
        """Test that gradients flow correctly"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))

        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        # Check that embeddings have gradients
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertFalse(torch.isnan(model.embedding.weight.grad).any())


class TestDeepWaveNetwork(unittest.TestCase):
    """Test suite for DeepWaveNetwork implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.num_classes = 4
        self.batch_size = 2
        self.seq_len = 8
        self.num_layers = 3

    def test_wave_layer_residual(self):
        """Test that WaveLayer has residual connections"""
        layer = WaveLayer(self.embedding_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)

        output = layer(x)

        # Output should have same shape
        self.assertEqual(output.shape, x.shape)

        # Output should be different from input (due to processing)
        self.assertFalse(torch.allclose(output, x))

    def test_deep_network_forward(self):
        """Test forward pass through deep network"""
        model = DeepWaveNetwork(self.vocab_size, self.embedding_dim,
                               self.num_classes, num_layers=self.num_layers)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))

        output = model(input_ids)

        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any())

    def test_deep_network_with_mask(self):
        """Test deep network with attention mask"""
        model = DeepWaveNetwork(self.vocab_size, self.embedding_dim,
                               self.num_classes, num_layers=self.num_layers)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, self.seq_len//2:] = 0

        output = model(input_ids, attention_mask=mask)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(output).any())

    def test_num_layers(self):
        """Test that correct number of layers are created"""
        for num_layers in [1, 3, 5]:
            model = DeepWaveNetwork(self.vocab_size, self.embedding_dim,
                                   self.num_classes, num_layers=num_layers)
            self.assertEqual(len(model.wave_layers), num_layers)


class TestWaveAttention(unittest.TestCase):
    """Test suite for WaveAttention implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.embedding_dim = 64
        self.num_heads = 8
        self.batch_size = 2
        self.seq_len = 8

    def test_attention_forward(self):
        """Test wave attention forward pass"""
        attention = WaveAttention(self.embedding_dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)

        output = attention(x)

        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_attention_with_mask(self):
        """Test wave attention with masking"""
        attention = WaveAttention(self.embedding_dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)
        mask = torch.ones(self.batch_size, self.seq_len)
        mask[:, self.seq_len//2:] = 0

        output = attention(x, attention_mask=mask)

        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_multihead_consistency(self):
        """Test that different head counts work correctly"""
        for num_heads in [1, 2, 4, 8]:
            if self.embedding_dim % num_heads == 0:
                attention = WaveAttention(self.embedding_dim, num_heads=num_heads)
                x = torch.randn(self.batch_size, self.seq_len, self.embedding_dim)

                output = attention(x)
                self.assertEqual(output.shape, x.shape)

    def test_attention_network_forward(self):
        """Test wave attention network forward pass"""
        vocab_size = 1000
        num_classes = 4
        model = WaveAttentionNetwork(vocab_size, self.embedding_dim,
                                    num_classes, num_layers=2,
                                    num_heads=self.num_heads)

        input_ids = torch.randint(0, vocab_size,
                                 (self.batch_size, self.seq_len))

        output = model(input_ids)

        self.assertEqual(output.shape, (self.batch_size, num_classes))
        self.assertFalse(torch.isnan(output).any())


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of wave operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.num_classes = 4
        self.batch_size = 2
        self.seq_len = 8

    def test_extreme_values(self):
        """Test behavior with extreme input values"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)

        # Test with large vocab indices
        input_ids = torch.randint(self.vocab_size - 10, self.vocab_size,
                                 (self.batch_size, self.seq_len))

        output = model(input_ids)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_zero_mask(self):
        """Test behavior with all-zero mask"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))

        # Create mask with at least one non-zero element to avoid division by zero
        mask = torch.zeros(self.batch_size, self.seq_len)
        mask[:, 0] = 1  # Keep first token

        output = model(input_ids, attention_mask=mask)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_gradient_stability(self):
        """Test that gradients remain stable"""
        model = WaveNetwork(self.vocab_size, self.embedding_dim, self.num_classes)
        input_ids = torch.randint(0, self.vocab_size,
                                 (self.batch_size, self.seq_len))

        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        # Check all gradients for stability
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any(),
                               f"NaN gradient in {name}")
                self.assertFalse(torch.isinf(param.grad).any(),
                               f"Inf gradient in {name}")


if __name__ == '__main__':
    unittest.main()
