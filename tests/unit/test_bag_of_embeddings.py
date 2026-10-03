"""Unit tests for the bag-of-embeddings baseline."""

import torch

from models.bag_of_embeddings import BagOfEmbeddings


def _make_model(small_config):
    return BagOfEmbeddings(
        vocab_size=small_config["vocab_size"],
        embedding_dim=small_config["embedding_dim"],
        num_classes=small_config["num_classes"],
    )


class TestBagOfEmbeddings:
    """Tests for BagOfEmbeddings."""

    def test_forward_shape(self, small_config, sample_input):
        model = _make_model(small_config)
        output = model(sample_input)
        assert output.shape == (small_config["batch_size"], small_config["num_classes"])
        assert torch.isfinite(output).all()

    def test_word_order_invariant(self, small_config, sample_input):
        """The baseline is a bag of tokens: reordering must not change the logits."""
        model = _make_model(small_config)
        shuffled = torch.roll(sample_input, shifts=3, dims=1)
        assert torch.allclose(model(sample_input), model(shuffled), atol=1e-5)

    def test_padding_is_ignored(self, small_config, sample_input, partial_mask):
        """Tokens under the mask must not affect the logits."""
        model = _make_model(small_config)
        changed = sample_input.clone()
        half = small_config["seq_len"] // 2
        changed[:, half:] = (changed[:, half:] + 1) % small_config["vocab_size"]

        out = model(sample_input, attention_mask=partial_mask)
        out_changed = model(changed, attention_mask=partial_mask)
        assert torch.allclose(out, out_changed, atol=1e-5)

    def test_backward(self, small_config, sample_input):
        model = _make_model(small_config)
        model(sample_input).sum().backward()
        assert model.embedding.weight.grad is not None
        assert torch.isfinite(model.embedding.weight.grad).all()

    def test_matches_wave_network_minus_wave_op(self, small_config):
        """Same parameters as WaveNetwork apart from the wave-specific ones."""
        from wave_network import WaveNetwork

        bag = dict(_make_model(small_config).named_parameters())
        wave = dict(
            WaveNetwork(
                small_config["vocab_size"],
                small_config["embedding_dim"],
                small_config["num_classes"],
            ).named_parameters()
        )
        for name, param in bag.items():
            assert wave[name].shape == param.shape
