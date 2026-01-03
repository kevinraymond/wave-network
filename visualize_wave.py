"""
Wave Network Visualization: Understanding What Wave Layers Learn

This script provides tools to visualize:
1. Phase distributions - how tokens are encoded in phase space
2. Global semantics (magnitude) patterns per dimension
3. Modulation vs interference behavior
4. How complex representations evolve through the network
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# Use non-interactive backend for saving
plt.switch_backend("Agg")


def get_phase_and_magnitude(x, eps=1e-8):
    """
    Extract phase and magnitude from real tensor using wave network method.

    Args:
        x: Real tensor of shape (batch, seq_len, embed_dim)
        eps: Small constant for numerical stability

    Returns:
        magnitude: Global semantics per dimension (batch, 1, embed_dim)
        phase: Phase angles per token (batch, seq_len, embed_dim)
        complex_repr: Complex representation (batch, seq_len, embed_dim)
    """
    # Global semantics: L2 norm across sequence for each dimension
    squared = x * x
    magnitude = torch.sqrt(torch.sum(squared, dim=1, keepdim=True) + eps)

    # Phase calculation
    ratio = x / (magnitude + eps)
    ratio = torch.clamp(ratio, -1 + eps, 1 - eps)
    sqrt_term = torch.sqrt(torch.clamp(1 - ratio**2, min=eps))
    phase = torch.atan2(sqrt_term, ratio)

    # Complex representation
    complex_repr = magnitude * torch.exp(1j * phase)

    return magnitude, phase, complex_repr


def visualize_phase_distribution(embeddings, title="Phase Distribution", save_path=None):
    """
    Visualize phase distribution across tokens and dimensions.

    Args:
        embeddings: Tensor of shape (batch, seq_len, embed_dim)
        title: Plot title
        save_path: Path to save figure
    """
    with torch.no_grad():
        magnitude, phase, _ = get_phase_and_magnitude(embeddings)

        # Take first batch item
        phase_np = phase[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Phase heatmap (first 64 dimensions)
        ax1 = axes[0]
        im1 = ax1.imshow(phase_np[:, :64], aspect="auto", cmap="hsv", vmin=0, vmax=np.pi)
        ax1.set_xlabel("Embedding Dimension")
        ax1.set_ylabel("Token Position")
        ax1.set_title("Phase Values (first 64 dims)")
        plt.colorbar(im1, ax=ax1, label="Phase (radians)")

        # 2. Phase histogram
        ax2 = axes[1]
        ax2.hist(phase_np.flatten(), bins=50, density=True, alpha=0.7, color="blue")
        ax2.axvline(x=np.pi / 2, color="red", linestyle="--", label="π/2")
        ax2.set_xlabel("Phase (radians)")
        ax2.set_ylabel("Density")
        ax2.set_title("Phase Distribution")
        ax2.legend()

        # 3. Mean phase per position
        ax3 = axes[2]
        mean_phase = phase_np.mean(axis=1)
        std_phase = phase_np.std(axis=1)
        positions = np.arange(len(mean_phase))
        ax3.plot(positions, mean_phase, "b-", label="Mean")
        ax3.fill_between(
            positions, mean_phase - std_phase, mean_phase + std_phase, alpha=0.3, label="±1 std"
        )
        ax3.set_xlabel("Token Position")
        ax3.set_ylabel("Phase (radians)")
        ax3.set_title("Phase by Position")
        ax3.legend()

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close()


def visualize_global_semantics(embeddings, title="Global Semantics", save_path=None):
    """
    Visualize global semantics (magnitude) patterns.

    Args:
        embeddings: Tensor of shape (batch, seq_len, embed_dim)
        title: Plot title
        save_path: Path to save figure
    """
    with torch.no_grad():
        magnitude, _, _ = get_phase_and_magnitude(embeddings)

        # Take first batch item, squeeze sequence dim
        mag_np = magnitude[0, 0].cpu().numpy()  # (embed_dim,)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Magnitude per dimension
        ax1 = axes[0]
        ax1.bar(range(min(64, len(mag_np))), mag_np[:64], alpha=0.7)
        ax1.set_xlabel("Embedding Dimension")
        ax1.set_ylabel("Global Semantics (G_k)")
        ax1.set_title("Magnitude per Dimension (first 64)")

        # 2. Magnitude distribution
        ax2 = axes[1]
        ax2.hist(mag_np, bins=30, density=True, alpha=0.7, color="green")
        ax2.axvline(
            x=mag_np.mean(), color="red", linestyle="--", label=f"Mean: {mag_np.mean():.3f}"
        )
        ax2.set_xlabel("Global Semantics Value")
        ax2.set_ylabel("Density")
        ax2.set_title("Magnitude Distribution")
        ax2.legend()

        # 3. Sorted magnitudes (to see structure)
        ax3 = axes[2]
        sorted_mag = np.sort(mag_np)[::-1]
        ax3.plot(sorted_mag, "b-")
        ax3.set_xlabel("Rank")
        ax3.set_ylabel("Global Semantics (G_k)")
        ax3.set_title("Sorted Magnitudes")
        ax3.set_yscale("log")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close()


def visualize_modulation_vs_interference(z1, z2, save_path=None):
    """
    Compare modulation and interference operations.

    Args:
        z1, z2: Tensors of shape (batch, seq_len, embed_dim)
        save_path: Path to save figure
    """
    with torch.no_grad():
        # Get complex representations
        mag1, phase1, c1 = get_phase_and_magnitude(z1)
        mag2, phase2, c2 = get_phase_and_magnitude(z2)

        # Modulation: c1 * c2
        modulated = c1 * c2
        mod_mag = torch.abs(modulated)
        mod_phase = torch.angle(modulated)

        # Interference: c1 + c2
        interfered = c1 + c2
        int_mag = torch.abs(interfered)
        int_phase = torch.angle(interfered)

        # Take first batch item
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Row 1: Magnitudes
        axes[0, 0].imshow(mag1[0, 0:1].cpu().numpy().T, aspect="auto", cmap="viridis")
        axes[0, 0].set_title("Input 1 Magnitude")

        axes[0, 1].imshow(mag2[0, 0:1].cpu().numpy().T, aspect="auto", cmap="viridis")
        axes[0, 1].set_title("Input 2 Magnitude")

        axes[0, 2].imshow(mod_mag[0].cpu().numpy()[:, :64].T, aspect="auto", cmap="viridis")
        axes[0, 2].set_title("Modulation Output Magnitude")
        axes[0, 2].set_xlabel("Token")

        axes[0, 3].imshow(int_mag[0].cpu().numpy()[:, :64].T, aspect="auto", cmap="viridis")
        axes[0, 3].set_title("Interference Output Magnitude")
        axes[0, 3].set_xlabel("Token")

        # Row 2: Phases
        axes[1, 0].imshow(phase1[0].cpu().numpy()[:, :64].T, aspect="auto", cmap="hsv")
        axes[1, 0].set_title("Input 1 Phase")

        axes[1, 1].imshow(phase2[0].cpu().numpy()[:, :64].T, aspect="auto", cmap="hsv")
        axes[1, 1].set_title("Input 2 Phase")

        axes[1, 2].imshow(mod_phase[0].cpu().numpy()[:, :64].T, aspect="auto", cmap="hsv")
        axes[1, 2].set_title("Modulation Output Phase")
        axes[1, 2].set_xlabel("Token")

        axes[1, 3].imshow(int_phase[0].cpu().numpy()[:, :64].T, aspect="auto", cmap="hsv")
        axes[1, 3].set_title("Interference Output Phase")
        axes[1, 3].set_xlabel("Token")

        plt.suptitle("Modulation (multiply) vs Interference (add) Comparison")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close()

        # Print statistics
        print("\n=== Modulation vs Interference Statistics ===")
        print(f"Modulation magnitude - mean: {mod_mag.mean():.4f}, std: {mod_mag.std():.4f}")
        print(f"Interference magnitude - mean: {int_mag.mean():.4f}, std: {int_mag.std():.4f}")
        print(f"Modulation phase range: [{mod_phase.min():.4f}, {mod_phase.max():.4f}]")
        print(f"Interference phase range: [{int_phase.min():.4f}, {int_phase.max():.4f}]")


def visualize_complex_plane(embeddings, title="Complex Plane View", save_path=None):
    """
    Visualize embeddings in the complex plane.

    Args:
        embeddings: Tensor of shape (batch, seq_len, embed_dim)
        title: Plot title
        save_path: Path to save figure
    """
    with torch.no_grad():
        _, _, complex_repr = get_phase_and_magnitude(embeddings)

        # Take first batch, sample some dimensions
        c = complex_repr[0].cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Show 8 different embedding dimensions
        dims_to_show = [0, 16, 32, 64, 128, 192, 224, min(255, c.shape[1] - 1)]

        for idx, dim in enumerate(dims_to_show):
            ax = axes[idx // 4, idx % 4]

            # Extract real and imaginary parts for this dimension
            real = c[:, dim].real
            imag = c[:, dim].imag

            # Color by position
            colors = np.arange(len(real))
            ax.scatter(real, imag, c=colors, cmap="viridis", s=30, alpha=0.7)

            # Draw unit circle for reference
            theta = np.linspace(0, 2 * np.pi, 100)
            max_r = max(np.abs(real).max(), np.abs(imag).max()) * 1.1
            ax.plot(max_r * np.cos(theta), max_r * np.sin(theta), "k--", alpha=0.3, linewidth=0.5)

            ax.set_xlim(-max_r, max_r)
            ax.set_ylim(-max_r, max_r)
            ax.set_aspect("equal")
            ax.set_xlabel("Real")
            ax.set_ylabel("Imaginary")
            ax.set_title(f"Dim {dim}")
            ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

        plt.suptitle(title + "\n(color = token position)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close()


def trace_forward_pass(model, input_ids, attention_mask=None, save_dir="data/results/wave_viz"):
    """
    Trace through forward pass and visualize intermediate states.

    Args:
        model: WaveNetwork model
        input_ids: Token IDs tensor
        attention_mask: Optional attention mask
        save_dir: Directory to save visualizations
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Step 1: Embeddings
        embeddings = model.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = embeddings * mask

        print("Step 1: Embeddings")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")

        visualize_phase_distribution(
            embeddings,
            title="Step 1: Input Embeddings",
            save_path=f"{save_dir}/step1_embeddings_phase.png",
        )
        visualize_global_semantics(
            embeddings,
            title="Step 1: Input Embeddings",
            save_path=f"{save_dir}/step1_embeddings_magnitude.png",
        )
        visualize_complex_plane(
            embeddings,
            title="Step 1: Input Embeddings",
            save_path=f"{save_dir}/step1_embeddings_complex.png",
        )

        # Step 2: Linear transformations
        z1 = model.linear1(embeddings)
        z2 = model.linear2(embeddings)

        print("\nStep 2: Linear Transformations")
        print(f"  z1 range: [{z1.min():.4f}, {z1.max():.4f}]")
        print(f"  z2 range: [{z2.min():.4f}, {z2.max():.4f}]")

        visualize_phase_distribution(
            z1, title="Step 2a: After Linear1", save_path=f"{save_dir}/step2_z1_phase.png"
        )
        visualize_phase_distribution(
            z2, title="Step 2b: After Linear2", save_path=f"{save_dir}/step2_z2_phase.png"
        )

        # Step 3: Wave operation comparison
        visualize_modulation_vs_interference(
            z1, z2, save_path=f"{save_dir}/step3_modulation_vs_interference.png"
        )

        # Step 4: Final output (run full forward)
        output = model(input_ids, attention_mask)
        print(f"\nFinal output shape: {output.shape}")
        print(f"Output logits: {output[0].cpu().numpy()}")


def main():
    """Run visualization demos."""
    from pathlib import Path

    save_dir = Path("data/results/wave_viz")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wave Network Visualization")
    print("=" * 60)

    # Demo 1: Random embeddings to understand the mechanism
    print("\n--- Demo 1: Understanding Phase/Magnitude from Random Data ---")

    batch_size, seq_len, embed_dim = 4, 32, 256
    random_embeddings = torch.randn(batch_size, seq_len, embed_dim)

    visualize_phase_distribution(
        random_embeddings,
        title="Random Embeddings - Phase Distribution",
        save_path=str(save_dir / "demo1_random_phase.png"),
    )
    visualize_global_semantics(
        random_embeddings,
        title="Random Embeddings - Global Semantics",
        save_path=str(save_dir / "demo1_random_magnitude.png"),
    )
    visualize_complex_plane(
        random_embeddings,
        title="Random Embeddings - Complex Plane",
        save_path=str(save_dir / "demo1_random_complex.png"),
    )

    # Demo 2: Modulation vs Interference
    print("\n--- Demo 2: Modulation vs Interference ---")

    z1 = torch.randn(batch_size, seq_len, embed_dim)
    z2 = torch.randn(batch_size, seq_len, embed_dim)

    visualize_modulation_vs_interference(z1, z2, save_path=str(save_dir / "demo2_mod_vs_int.png"))

    # Demo 3: Structured input (sinusoidal pattern)
    print("\n--- Demo 3: Structured (Sinusoidal) Input ---")

    positions = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1)
    freqs = torch.arange(embed_dim).float().unsqueeze(0).unsqueeze(0) / embed_dim * 10
    sinusoidal = torch.sin(positions * freqs) * 0.5
    sinusoidal = sinusoidal.expand(batch_size, -1, -1)

    visualize_phase_distribution(
        sinusoidal,
        title="Sinusoidal Pattern - Phase Distribution",
        save_path=str(save_dir / "demo3_sinusoidal_phase.png"),
    )
    visualize_complex_plane(
        sinusoidal,
        title="Sinusoidal Pattern - Complex Plane",
        save_path=str(save_dir / "demo3_sinusoidal_complex.png"),
    )

    # Demo 4: Trace actual model forward pass
    print("\n--- Demo 4: Trace WaveNetwork Forward Pass ---")

    try:
        from wave_network import WaveNetwork

        # Create small model
        model = WaveNetwork(vocab_size=1000, embedding_dim=256, num_classes=4, mode="modulation")

        # Random input
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones_like(input_ids)

        trace_forward_pass(
            model, input_ids, attention_mask, save_dir=str(save_dir / "demo4_forward_trace")
        )

    except ImportError as e:
        print(f"Could not import WaveNetwork: {e}")
        print("Skipping forward pass trace.")

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
