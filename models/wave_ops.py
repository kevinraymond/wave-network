"""Shared wave-representation math used by all Wave Network variants.

Implements the complex vector token representation from "Wave Network: An
Ultra-Small Language Model" (https://arxiv.org/abs/2411.02674), section 3:

    Z_jk = G_k * exp(i * alpha_jk)

where G_k = ||x_:,k||_2 is the global semantics of dimension k (shared by every
token) and alpha_jk = atan2(sqrt(1 - (x_jk/G_k)^2), x_jk/G_k) is the per-token
phase. By construction the real part recovers the input (Re(Z_jk) = x_jk, paper
Figure 4) and the imaginary part is the "context" sqrt(G_k^2 - x_jk^2).

Conversion back to real values (``from_complex``) is where implementations can
silently degenerate:

- ``abs`` after **modulation** is position-independent: |c1 * c2| = G1_k * G2_k
  for every token, so all token-level information is discarded.
- ``real`` after **interference** is linear: Re(c1 + c2) = z1 + z2, bypassing
  the wave representation entirely.
- ``real_imag`` preserves everything in both modes; the caller projects the
  doubled feature dimension back down (see WaveLayerSTFT in wave_audio.py).
"""

import torch


def global_semantics(x: torch.Tensor, dim: int = 1, eps: float = 0.0) -> torch.Tensor:
    """Global semantics G_k = ||x||_2 along ``dim`` (paper section 3.1.1).

    Keeps ``dim`` so the result broadcasts against ``x``. ``eps`` inside the
    sqrt guards the infinite gradient of sqrt at 0 for all-zero slices.
    """
    return torch.sqrt(torch.sum(x * x, dim=dim, keepdim=True) + eps)


def phase_components(
    x: torch.Tensor, g: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sqrt_term, ratio) for the phase alpha = atan2(sqrt_term, ratio).

    ratio = x/G is cos(alpha); sqrt_term = sqrt(1 - ratio^2) is sin(alpha).
    Both clamps keep sqrt away from 0, whose gradient is infinite — when a
    single position carries the entire norm of a dimension (ratio = ±1) an
    unclamped sqrt(0) produces NaN gradients.
    """
    ratio = x / (g + eps)
    ratio = torch.clamp(ratio, -1 + eps, 1 - eps)
    sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=eps))
    return sqrt_term, ratio


def to_complex(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert to the polar complex representation Z = G * exp(i * alpha)."""
    sqrt_term, ratio = phase_components(x, g, eps)
    phase = torch.atan2(sqrt_term, ratio)
    return g * torch.exp(1j * phase)


def wave_superpose(c1: torch.Tensor, c2: torch.Tensor, mode: str = "modulation") -> torch.Tensor:
    """Combine two complex representations (paper section 3.2).

    Interference is complex addition (phases interact constructively or
    destructively); modulation is complex multiplication (magnitudes multiply,
    phases add).
    """
    if mode == "interference":
        return c1 + c2
    if mode == "modulation":
        return c1 * c2
    raise ValueError(f"Unknown wave mode: {mode!r} (expected 'modulation' or 'interference')")


def from_complex(c: torch.Tensor, output_mode: str = "real_imag") -> torch.Tensor:
    """Convert a complex representation back to real features.

    ``real_imag`` doubles the last dimension (caller projects back down);
    ``abs`` and ``real`` preserve shape but each degenerates for one wave mode
    (see module docstring).
    """
    if output_mode == "abs":
        return torch.abs(c)
    if output_mode == "real":
        return c.real
    if output_mode == "real_imag":
        return torch.cat([c.real, c.imag], dim=-1)
    raise ValueError(
        f"Unknown output_mode: {output_mode!r} (expected 'abs', 'real', or 'real_imag')"
    )
