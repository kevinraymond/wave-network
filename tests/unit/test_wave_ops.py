"""Unit tests for the shared wave-representation math in models/wave_ops.py."""

import pytest
import torch

from models import wave_ops


class TestGlobalSemantics:
    """Test global semantics computation."""

    def test_shape(self):
        """Reduced dim is kept so the result broadcasts against the input."""
        x = torch.randn(2, 8, 16)
        g = wave_ops.global_semantics(x, dim=1)
        assert g.shape == (2, 1, 16)

    def test_value_matches_norm(self):
        """G_k must equal the L2 norm along the reduced dimension."""
        x = torch.randn(2, 8, 16)
        g = wave_ops.global_semantics(x, dim=1)
        expected = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        assert torch.allclose(g, expected, atol=1e-6)

    def test_zero_input_gradient_finite(self):
        """eps inside the sqrt keeps gradients finite for all-zero slices."""
        x = torch.zeros(1, 4, 3, requires_grad=True)
        g = wave_ops.global_semantics(x, dim=1, eps=1e-8)
        g.sum().backward()
        assert torch.isfinite(x.grad).all()


class TestToComplex:
    """Test the polar complex representation."""

    def test_unit_circle(self):
        """sin^2 + cos^2 = 1 for the phase components."""
        x = torch.randn(2, 8, 16)
        g = wave_ops.global_semantics(x, dim=1)
        sqrt_term, ratio = wave_ops.phase_components(x, g)
        circle = sqrt_term**2 + ratio**2
        assert torch.allclose(circle, torch.ones_like(circle), rtol=1e-4)

    def test_real_part_recovers_input(self):
        """Paper Figure 4: Re(Z_jk) = x_jk by construction."""
        x = torch.randn(2, 8, 16)
        g = wave_ops.global_semantics(x, dim=1)
        c = wave_ops.to_complex(x, g)
        assert torch.allclose(c.real, x, atol=1e-4)

    def test_magnitude_is_global_semantics(self):
        """|Z_jk| = G_k for every position j."""
        x = torch.randn(2, 8, 16)
        g = wave_ops.global_semantics(x, dim=1)
        c = wave_ops.to_complex(x, g)
        assert torch.allclose(torch.abs(c), g.expand_as(c.real), rtol=1e-4)

    def test_gradient_finite_when_position_carries_full_norm(self):
        """Regression: ratio = ±1 used to hit sqrt(0), whose gradient is inf.

        When a single position holds the entire norm of a dimension the
        unclamped formulation produced NaN gradients (the NaN issues noted in
        notes/benchmarks.md).
        """
        x = torch.zeros(1, 4, 2)
        x[0, 2, 0] = 3.0  # position 2 carries the full norm of dim 0
        x[0, 1, 1] = -2.0  # ratio = -1 case
        x.requires_grad_(True)

        g = wave_ops.global_semantics(x, dim=1, eps=1e-8)
        c = wave_ops.to_complex(x, g)
        wave_ops.from_complex(c, "real_imag").sum().backward()

        assert torch.isfinite(x.grad).all()


class TestWaveSuperpose:
    """Test interference and modulation."""

    def test_interference_is_addition(self):
        c1 = torch.randn(2, 4, 8, dtype=torch.complex64)
        c2 = torch.randn(2, 4, 8, dtype=torch.complex64)
        assert torch.equal(wave_ops.wave_superpose(c1, c2, "interference"), c1 + c2)

    def test_modulation_is_multiplication(self):
        c1 = torch.randn(2, 4, 8, dtype=torch.complex64)
        c2 = torch.randn(2, 4, 8, dtype=torch.complex64)
        assert torch.equal(wave_ops.wave_superpose(c1, c2, "modulation"), c1 * c2)

    def test_unknown_mode_raises(self):
        c = torch.randn(2, 4, 8, dtype=torch.complex64)
        with pytest.raises(ValueError, match="Unknown wave mode"):
            wave_ops.wave_superpose(c, c, "diffraction")


class TestFromComplex:
    """Test complex-to-real conversion modes."""

    @pytest.fixture
    def c(self):
        return torch.randn(2, 4, 8, dtype=torch.complex64)

    def test_abs(self, c):
        assert torch.equal(wave_ops.from_complex(c, "abs"), torch.abs(c))

    def test_real(self, c):
        assert torch.equal(wave_ops.from_complex(c, "real"), c.real)

    def test_real_imag_doubles_last_dim(self, c):
        out = wave_ops.from_complex(c, "real_imag")
        assert out.shape == (2, 4, 16)
        assert torch.equal(out[..., :8], c.real)
        assert torch.equal(out[..., 8:], c.imag)

    def test_unknown_mode_raises(self, c):
        with pytest.raises(ValueError, match="Unknown output_mode"):
            wave_ops.from_complex(c, "imag")


class TestDegeneracies:
    """Document the degenerate mode/output_mode combinations.

    These are the mathematical facts that motivated the v0.3.0 output_mode
    change; if either ever fails, the representation has changed.
    """

    def test_abs_of_modulation_is_position_independent(self):
        """|c1*c2| = G1*G2 — identical for every sequence position."""
        z1, z2 = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
        g1 = wave_ops.global_semantics(z1, dim=1)
        g2 = wave_ops.global_semantics(z2, dim=1)
        c1 = wave_ops.to_complex(z1, g1)
        c2 = wave_ops.to_complex(z2, g2)

        out = wave_ops.from_complex(wave_ops.wave_superpose(c1, c2, "modulation"), "abs")
        assert torch.allclose(out, out[:, :1, :].expand_as(out), atol=1e-4)

    def test_real_of_interference_is_linear(self):
        """Re(c1+c2) = z1 + z2 — the wave representation cancels out."""
        z1, z2 = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
        g1 = wave_ops.global_semantics(z1, dim=1)
        g2 = wave_ops.global_semantics(z2, dim=1)
        c1 = wave_ops.to_complex(z1, g1)
        c2 = wave_ops.to_complex(z2, g2)

        out = wave_ops.from_complex(wave_ops.wave_superpose(c1, c2, "interference"), "real")
        assert torch.allclose(out, z1 + z2, atol=1e-3)

    def test_real_imag_preserves_positions_in_both_modes(self):
        """real_imag output must stay position-dependent for both wave modes."""
        z1, z2 = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
        g1 = wave_ops.global_semantics(z1, dim=1)
        g2 = wave_ops.global_semantics(z2, dim=1)
        c1 = wave_ops.to_complex(z1, g1)
        c2 = wave_ops.to_complex(z2, g2)

        for mode in ("modulation", "interference"):
            out = wave_ops.from_complex(wave_ops.wave_superpose(c1, c2, mode), "real_imag")
            assert not torch.allclose(out, out[:, :1, :].expand_as(out), atol=1e-4)


class TestParts:
    """The trig-free (real, imag) path must match the complex-tensor path."""

    def _inputs(self):
        z1, z2 = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
        g1 = wave_ops.global_semantics(z1, dim=1, eps=1e-8)
        g2 = wave_ops.global_semantics(z2, dim=1, eps=1e-8)
        return z1, g1, z2, g2

    def test_to_parts_matches_polar_form(self):
        """real + i*imag equals G * exp(i * atan2(sqrt_term, ratio))."""
        z, g, _, _ = self._inputs()
        real, imag = wave_ops.to_parts(z, g)

        sqrt_term, ratio = wave_ops.phase_components(z, g)
        polar = g * torch.exp(1j * torch.atan2(sqrt_term, ratio))

        assert torch.allclose(real, polar.real, atol=1e-5)
        assert torch.allclose(imag, polar.imag, atol=1e-5)

    def test_imag_is_context(self):
        """imag = sqrt(G^2 - x^2), the part of the norm carried by other tokens."""
        z, g, _, _ = self._inputs()
        _, imag = wave_ops.to_parts(z, g)
        assert torch.allclose(imag, torch.sqrt(torch.clamp(g**2 - z**2, min=0)), atol=1e-3)

    @pytest.mark.parametrize("mode", ["modulation", "interference"])
    @pytest.mark.parametrize("output_mode", ["abs", "real", "real_imag"])
    def test_wave_features_matches_complex_path(self, mode, output_mode):
        z1, g1, z2, g2 = self._inputs()

        c = wave_ops.wave_superpose(wave_ops.to_complex(z1, g1), wave_ops.to_complex(z2, g2), mode)
        expected = wave_ops.from_complex(c, output_mode)

        out = wave_ops.wave_features(z1, g1, z2, g2, mode, output_mode)
        assert not out.is_complex()
        assert torch.allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_unknown_mode_raises(self):
        p = (torch.randn(2, 4), torch.randn(2, 4))
        with pytest.raises(ValueError, match="Unknown wave mode"):
            wave_ops.superpose_parts(p, p, "diffraction")

    def test_unknown_output_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown output_mode"):
            wave_ops.from_parts(torch.randn(2, 4), torch.randn(2, 4), "imag")

    @pytest.mark.parametrize("output_mode", ["abs", "real", "real_imag"])
    def test_gradient_finite_for_zero_input(self, output_mode):
        """All-zero slices (e.g. fully padded dimensions) must not produce NaN gradients."""
        z1 = torch.zeros(1, 4, 3, requires_grad=True)
        z2 = torch.zeros(1, 4, 3, requires_grad=True)
        g1 = wave_ops.global_semantics(z1, dim=1, eps=1e-8)
        g2 = wave_ops.global_semantics(z2, dim=1, eps=1e-8)

        wave_ops.wave_features(z1, g1, z2, g2, "modulation", output_mode).sum().backward()

        assert torch.isfinite(z1.grad).all()
        assert torch.isfinite(z2.grad).all()
