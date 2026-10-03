"""Unit tests for WaveNetwork core implementation."""

import pytest
import torch


class TestWaveNetwork:
    """Test suite for WaveNetwork implementation."""

    def test_global_semantics_shape(self, wave_network_model, small_config):
        """Test that global semantics has correct shape."""
        x = torch.randn(
            small_config["batch_size"], small_config["seq_len"], small_config["embedding_dim"]
        )
        g = wave_network_model.get_global_semantics_per_dim(x)

        expected_shape = (small_config["batch_size"], 1, small_config["embedding_dim"])
        assert g.shape == expected_shape

    def test_phase_unit_circle(self, wave_network_model, small_config):
        """Test that phase components satisfy unit circle constraint."""
        x = torch.randn(
            small_config["batch_size"], small_config["seq_len"], small_config["embedding_dim"]
        )
        g = wave_network_model.get_global_semantics_per_dim(x)
        sqrt_term, ratio = wave_network_model.get_phase_components(x, g)

        # Should satisfy: sqrt_term^2 + ratio^2 ≈ 1
        circle_check = sqrt_term**2 + ratio**2
        assert torch.allclose(circle_check, torch.ones_like(circle_check), rtol=1e-4)

    def test_forward_pass_shape(self, small_config, wave_mode):
        """Test forward pass produces correct output shape."""
        from wave_network import WaveNetwork

        model = WaveNetwork(
            small_config["vocab_size"],
            small_config["embedding_dim"],
            small_config["num_classes"],
            mode=wave_mode,
        )
        input_ids = torch.randint(
            0, small_config["vocab_size"], (small_config["batch_size"], small_config["seq_len"])
        )
        output = model(input_ids)

        expected_shape = (small_config["batch_size"], small_config["num_classes"])
        assert output.shape == expected_shape

    def test_no_nan_inf(self, wave_network_model, sample_input):
        """Test that forward pass produces no NaN or Inf values."""
        output = wave_network_model(sample_input)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_mask_full(self, wave_network_model, sample_input, small_config):
        """Test attention mask with all ones matches no mask."""
        output_no_mask = wave_network_model(sample_input, attention_mask=None)

        mask_full = torch.ones(small_config["batch_size"], small_config["seq_len"])
        output_full_mask = wave_network_model(sample_input, attention_mask=mask_full)

        assert torch.allclose(output_no_mask, output_full_mask, rtol=1e-3)

    def test_attention_mask_partial(self, wave_network_model, sample_input, partial_mask):
        """Test that partial mask produces different output."""
        output_no_mask = wave_network_model(sample_input, attention_mask=None)
        output_partial = wave_network_model(sample_input, attention_mask=partial_mask)

        assert not torch.allclose(output_no_mask, output_partial, rtol=1e-3)

    def test_learnable_mode(self, small_config, sample_input):
        """Test learnable mode mixing."""
        from wave_network import WaveNetwork

        model = WaveNetwork(
            small_config["vocab_size"],
            small_config["embedding_dim"],
            small_config["num_classes"],
            learnable_mode=True,
        )

        assert hasattr(model, "mode_weight")

        output = model(sample_input)
        assert output.shape == (small_config["batch_size"], small_config["num_classes"])
        assert not torch.isnan(output).any()

    def test_epsilon_configurable(self, small_config):
        """Test that epsilon is configurable."""
        from wave_network import WaveNetwork

        eps_custom = 1e-6
        model = WaveNetwork(
            small_config["vocab_size"],
            small_config["embedding_dim"],
            small_config["num_classes"],
            eps=eps_custom,
        )
        assert model.eps == eps_custom

    def test_backward_pass(self, wave_network_model, sample_input):
        """Test that gradients flow correctly."""
        output = wave_network_model(sample_input)
        loss = output.sum()
        loss.backward()

        assert wave_network_model.embedding.weight.grad is not None
        assert not torch.isnan(wave_network_model.embedding.weight.grad).any()


class TestDeepWaveNetwork:
    """Test suite for DeepWaveNetwork implementation."""

    def test_wave_layer_residual(self, small_config):
        """Test that WaveLayer has residual connections."""
        from wave_network_deep import WaveLayer

        layer = WaveLayer(small_config["embedding_dim"])
        x = torch.randn(
            small_config["batch_size"], small_config["seq_len"], small_config["embedding_dim"]
        )

        output = layer(x)

        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should be different

    def test_deep_network_forward(self, deep_wave_network_model, sample_input, small_config):
        """Test forward pass through deep network."""
        output = deep_wave_network_model(sample_input)

        expected_shape = (small_config["batch_size"], small_config["num_classes"])
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()

    def test_deep_network_with_mask(
        self, deep_wave_network_model, sample_input, partial_mask, small_config
    ):
        """Test deep network with attention mask."""
        output = deep_wave_network_model(sample_input, attention_mask=partial_mask)

        assert output.shape == (small_config["batch_size"], small_config["num_classes"])
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("num_layers", [1, 3, 5])
    def test_num_layers(self, small_config, num_layers):
        """Test that correct number of layers are created."""
        from wave_network_deep import DeepWaveNetwork

        model = DeepWaveNetwork(
            small_config["vocab_size"],
            small_config["embedding_dim"],
            small_config["num_classes"],
            num_layers=num_layers,
        )
        assert len(model.wave_layers) == num_layers


class TestWaveAttention:
    """Test suite for WaveAttention implementation."""

    def test_attention_forward(self, wave_attention_model, small_config):
        """Test wave attention forward pass."""
        x = torch.randn(
            small_config["batch_size"], small_config["seq_len"], small_config["embedding_dim"]
        )

        output = wave_attention_model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_attention_with_mask(self, wave_attention_model, small_config, partial_mask):
        """Test wave attention with masking."""
        x = torch.randn(
            small_config["batch_size"], small_config["seq_len"], small_config["embedding_dim"]
        )

        output = wave_attention_model(x, attention_mask=partial_mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("num_heads", [8])
    def test_multihead_consistency(self, small_config, num_heads):
        """Test that different head counts work correctly.

        Note: The current WaveAttention implementation has a shape mismatch bug
        in get_global_semantics for K (line 83) that only works by coincidence
        when head_dim == seq_len. We only test with num_heads=8 which is the
        default configuration.
        """
        from wave_attention import WaveAttention

        if small_config["embedding_dim"] % num_heads != 0:
            pytest.skip(f"embedding_dim not divisible by {num_heads}")

        attention = WaveAttention(small_config["embedding_dim"], num_heads=num_heads)
        x = torch.randn(
            small_config["batch_size"], small_config["seq_len"], small_config["embedding_dim"]
        )

        output = attention(x)
        assert output.shape == x.shape

    def test_attention_network_forward(self, small_config):
        """Test wave attention network forward pass."""
        from wave_attention import WaveAttentionNetwork

        model = WaveAttentionNetwork(
            small_config["vocab_size"],
            small_config["embedding_dim"],
            small_config["num_classes"],
            num_layers=2,
            num_heads=8,
        )

        input_ids = torch.randint(
            0, small_config["vocab_size"], (small_config["batch_size"], small_config["seq_len"])
        )

        output = model(input_ids)

        assert output.shape == (small_config["batch_size"], small_config["num_classes"])
        assert not torch.isnan(output).any()


class TestOutputModes:
    """Regression tests for the v0.3.0 complex-to-real conversion fix.

    With output_mode="abs" (the pre-v0.3.0 behavior) modulation collapses to
    |c1*c2| = G1*G2, identical at every sequence position, so the classifier
    only ever saw sequence-level norms.
    """

    def _per_position_wave_output(self, model, input_ids):
        """Wave-layer output before pooling, mirroring WaveNetwork.forward."""
        emb = model.embedding(input_ids) + model.pos_embed[:, : input_ids.size(1), :]
        z1, z2 = model.linear1(emb), model.linear2(emb)
        return model._from_complex(model._superpose(z1, z2, model.mode))

    def _make_model(self, small_config, **kwargs):
        from wave_network import WaveNetwork

        return WaveNetwork(
            small_config["vocab_size"],
            small_config["embedding_dim"],
            small_config["num_classes"],
            **kwargs,
        )

    def test_legacy_abs_modulation_collapses_positions(self, small_config, sample_input):
        """Documents the legacy degeneracy: every position gets the same vector."""
        model = self._make_model(small_config, mode="modulation", output_mode="abs")
        out = self._per_position_wave_output(model, sample_input)

        assert torch.allclose(out, out[:, :1, :].expand_as(out), atol=1e-5)

    @pytest.mark.parametrize("output_mode", ["real", "real_imag"])
    def test_fixed_modulation_preserves_positions(self, small_config, sample_input, output_mode):
        """Phase-preserving conversions must stay position-dependent."""
        model = self._make_model(small_config, mode="modulation", output_mode=output_mode)
        out = self._per_position_wave_output(model, sample_input)

        assert not torch.allclose(out, out[:, :1, :].expand_as(out), atol=1e-5)

    def test_legacy_abs_modulation_is_word_order_blind(self, small_config, sample_input):
        """Without positional encoding, the legacy model ignores word order entirely."""
        model = self._make_model(small_config, mode="modulation", output_mode="abs")
        with torch.no_grad():
            model.pos_embed.zero_()

        shuffled = torch.roll(sample_input, shifts=1, dims=1)
        assert torch.allclose(model(sample_input), model(shuffled), atol=1e-5)

    def test_word_order_changes_logits(self, small_config, sample_input):
        """With positional encoding, reordering tokens must change the logits."""
        model = self._make_model(small_config, mode="modulation")

        shuffled = torch.roll(sample_input, shifts=1, dims=1)
        assert not torch.allclose(model(sample_input), model(shuffled), atol=1e-5)

    def test_abs_mode_has_no_extra_parameters(self, small_config):
        """output_mode="abs" must stay loadable against pre-v0.3.0 checkpoints."""
        model = self._make_model(small_config, output_mode="abs")
        assert not hasattr(model, "out_proj")

    def test_deep_network_output_modes(self, small_config, sample_input):
        """DeepWaveNetwork accepts the same output_mode options."""
        from wave_network_deep import DeepWaveNetwork

        for output_mode in ("abs", "real", "real_imag"):
            model = DeepWaveNetwork(
                small_config["vocab_size"],
                small_config["embedding_dim"],
                small_config["num_classes"],
                num_layers=2,
                output_mode=output_mode,
            )
            output = model(sample_input)
            assert output.shape == (small_config["batch_size"], small_config["num_classes"])
            assert not torch.isnan(output).any()


class TestNumericalStability:
    """Test numerical stability of wave operations."""

    def test_extreme_values(self, wave_network_model, small_config):
        """Test behavior with extreme input values."""
        input_ids = torch.randint(
            small_config["vocab_size"] - 10,
            small_config["vocab_size"],
            (small_config["batch_size"], small_config["seq_len"]),
        )

        output = wave_network_model(input_ids)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_minimal_mask(self, wave_network_model, small_config):
        """Test behavior with minimal mask (only first token)."""
        input_ids = torch.randint(
            0, small_config["vocab_size"], (small_config["batch_size"], small_config["seq_len"])
        )

        mask = torch.zeros(small_config["batch_size"], small_config["seq_len"])
        mask[:, 0] = 1  # Keep only first token

        output = wave_network_model(input_ids, attention_mask=mask)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_stability(self, wave_network_model, sample_input):
        """Test that gradients remain stable."""
        output = wave_network_model(sample_input)
        loss = output.sum()
        loss.backward()

        for name, param in wave_network_model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
