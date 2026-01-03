"""
Visualize learned representations from trained Wave Audio STFT model.

This script loads the best-performing audio model and visualizes:
1. What the magnitude/phase embeddings learn
2. How the WaveLayerComplex processes audio
3. Class-discriminative patterns
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.audio import SPEECH_COMMANDS_LABELS, AudioConfig, SpeechCommandsDataset
from models.wave_audio import WaveAudioSTFT

matplotlib.use("Agg")


def load_sample_data(n_samples=8):
    """Load sample audio data."""
    config = AudioConfig(representation="stft")
    dataset = SpeechCommandsDataset(config=config, subset="testing")

    # Get diverse samples (different classes)
    samples = []
    labels = []
    seen_classes = set()

    for i in range(len(dataset)):
        x, y = dataset[i]
        if y not in seen_classes:
            samples.append(x)
            labels.append(y)
            seen_classes.add(y)
            if len(samples) >= n_samples:
                break

    return torch.stack(samples), torch.tensor(labels)


def visualize_stft_input(x, label_idx, save_path=None):
    """Visualize STFT input (magnitude and phase channels)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    mag = x[0].cpu().numpy()  # (freq, time)
    phase = x[1].cpu().numpy()

    # Magnitude
    im1 = axes[0].imshow(mag, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Frequency Bin")
    axes[0].set_title("STFT Magnitude")
    plt.colorbar(im1, ax=axes[0])

    # Phase
    im2 = axes[1].imshow(phase, aspect="auto", origin="lower", cmap="hsv", vmin=-np.pi, vmax=np.pi)
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Frequency Bin")
    axes[1].set_title("STFT Phase")
    plt.colorbar(im2, ax=axes[1])

    # Combined (magnitude * cos(phase) for visualization)
    combined = mag * np.cos(phase)
    im3 = axes[2].imshow(combined, aspect="auto", origin="lower", cmap="RdBu")
    axes[2].set_xlabel("Time Frame")
    axes[2].set_ylabel("Frequency Bin")
    axes[2].set_title("Magnitude × cos(Phase)")
    plt.colorbar(im3, ax=axes[2])

    label_name = SPEECH_COMMANDS_LABELS[label_idx]
    plt.suptitle(f'STFT Input for "{label_name}"')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def visualize_embeddings(model, x, save_path=None):
    """Visualize what the magnitude and phase embeddings learn."""
    model.eval()
    with torch.no_grad():
        # Split input
        mag = x[:, 0:1, :, :]  # (batch, 1, freq, time)
        phase = x[:, 1:2, :, :]

        # Get embeddings
        mag_emb = model.mag_embed(mag)  # (batch, dim/2, grid_h, grid_w)
        phase_emb = model.phase_embed(phase)

        # Take first sample
        mag_emb_np = mag_emb[0].cpu().numpy()  # (dim/2, grid_h, grid_w)
        phase_emb_np = phase_emb[0].cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Show some magnitude embedding channels
        for i, dim in enumerate([0, 16, 32, 64]):
            if dim < mag_emb_np.shape[0]:
                axes[0, i].imshow(mag_emb_np[dim], aspect="auto", cmap="viridis")
                axes[0, i].set_title(f"Mag Embed Ch {dim}")
                axes[0, i].set_xlabel("Time")
                axes[0, i].set_ylabel("Freq")

        # Show some phase embedding channels
        for i, dim in enumerate([0, 16, 32, 64]):
            if dim < phase_emb_np.shape[0]:
                axes[1, i].imshow(phase_emb_np[dim], aspect="auto", cmap="hsv")
                axes[1, i].set_title(f"Phase Embed Ch {dim}")
                axes[1, i].set_xlabel("Time")
                axes[1, i].set_ylabel("Freq")

        plt.suptitle("Learned Magnitude and Phase Embeddings")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close()

        return mag_emb, phase_emb


def visualize_layer_activations(model, x, save_path=None):
    """Visualize activations through the wave layers."""
    model.eval()

    # Hook to capture intermediate activations
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    # Register hooks on each wave layer
    hooks = []
    for layer in model.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)  # Run forward to trigger hooks

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Visualize activations
    n_layers = len(activations)
    fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8))

    for i, act in enumerate(activations):
        act_np = act[0].cpu().numpy()  # (seq, dim)

        # Activation magnitude heatmap
        axes[0, i].imshow(act_np[:, :64].T, aspect="auto", cmap="viridis")
        axes[0, i].set_title(f"Layer {i+1} Activations")
        axes[0, i].set_xlabel("Patch")
        axes[0, i].set_ylabel("Dim")

        # Activation statistics
        axes[1, i].hist(act_np.flatten(), bins=50, density=True, alpha=0.7)
        axes[1, i].set_xlabel("Activation Value")
        axes[1, i].set_ylabel("Density")
        axes[1, i].set_title(f"Layer {i+1} Distribution")
        axes[1, i].axvline(x=0, color="r", linestyle="--", alpha=0.5)

    plt.suptitle("Activations Through Wave Layers")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

    return activations


def visualize_class_patterns(model, data, labels, save_path=None):
    """Visualize how different classes produce different representations."""
    model.eval()

    # Get pooled representations before classifier
    pooled_reps = []

    def hook_fn(module, input, output):
        pooled_reps.append(input[0].detach())  # input to classifier

    hook = model.classifier.register_forward_hook(hook_fn)

    with torch.no_grad():
        output = model(data)
        predictions = output.argmax(dim=1)

    hook.remove()

    pooled = pooled_reps[0].cpu().numpy()  # (batch, dim)

    # t-SNE or PCA visualization
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(pooled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter by true label
    axes[0].scatter(
        reduced[:, 0], reduced[:, 1], c=labels.numpy(), cmap="tab10", s=100, edgecolors="black"
    )
    for i, (x, y) in enumerate(reduced):
        axes[0].annotate(
            SPEECH_COMMANDS_LABELS[labels[i]][:4], (x, y), fontsize=8, ha="center", va="bottom"
        )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("True Labels")

    # Scatter by prediction
    axes[1].scatter(
        reduced[:, 0], reduced[:, 1], c=predictions.numpy(), cmap="tab10", s=100, edgecolors="black"
    )
    for i, (x, y) in enumerate(reduced):
        pred_label = SPEECH_COMMANDS_LABELS[predictions[i]][:4]
        axes[1].annotate(pred_label, (x, y), fontsize=8, ha="center", va="bottom")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Predictions")

    plt.suptitle(f"Class Representations (PCA: {pca.explained_variance_ratio_.sum():.1%} var)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

    return pooled, predictions


def visualize_complex_layer_internals(model, x, save_path=None):
    """Visualize what happens inside WaveLayerComplex."""
    model.eval()

    # Hook into the first wave layer to see mag/phase projections
    layer = model.layers[0]

    with torch.no_grad():
        # Get embeddings first
        mag_input = x[:, 0:1, :, :]
        phase_input = x[:, 1:2, :, :]

        mag_emb = model.mag_embed(mag_input)
        phase_emb = model.phase_embed(phase_input)
        combined = torch.cat([mag_emb, phase_emb], dim=1)
        tokens = combined.flatten(2).transpose(1, 2)
        tokens = tokens + model.pos_embed

        # Now trace through first wave layer
        normed = layer.norm(tokens)

        # Project to mag and phase
        mag_proj = torch.abs(layer.mag_proj(normed)) + layer.eps
        phase_proj = layer.phase_proj(normed)

        # Create complex and modulate
        c = mag_proj * torch.exp(1j * phase_proj)
        c_global = c.mean(dim=1, keepdim=True)
        c_modulated = c * c_global.conj()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # First row: magnitude pathway
        axes[0, 0].imshow(normed[0, :, :64].cpu().numpy().T, aspect="auto", cmap="RdBu")
        axes[0, 0].set_title("Normalized Input")
        axes[0, 0].set_xlabel("Patch")

        axes[0, 1].imshow(mag_proj[0, :, :64].cpu().numpy().T, aspect="auto", cmap="viridis")
        axes[0, 1].set_title("Magnitude Projection")
        axes[0, 1].set_xlabel("Patch")

        axes[0, 2].imshow(torch.abs(c)[0, :, :64].cpu().numpy().T, aspect="auto", cmap="viridis")
        axes[0, 2].set_title("Complex Magnitude")
        axes[0, 2].set_xlabel("Patch")

        axes[0, 3].imshow(
            torch.abs(c_modulated)[0, :, :64].cpu().numpy().T, aspect="auto", cmap="viridis"
        )
        axes[0, 3].set_title("After Global Modulation")
        axes[0, 3].set_xlabel("Patch")

        # Second row: phase pathway
        axes[1, 0].hist(normed[0].cpu().numpy().flatten(), bins=50, density=True)
        axes[1, 0].set_title("Input Distribution")

        axes[1, 1].imshow(
            phase_proj[0, :, :64].cpu().numpy().T,
            aspect="auto",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[1, 1].set_title("Phase Projection")
        axes[1, 1].set_xlabel("Patch")

        axes[1, 2].imshow(
            torch.angle(c)[0, :, :64].cpu().numpy().T,
            aspect="auto",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[1, 2].set_title("Complex Phase")
        axes[1, 2].set_xlabel("Patch")

        axes[1, 3].imshow(
            torch.angle(c_modulated)[0, :, :64].cpu().numpy().T,
            aspect="auto",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[1, 3].set_title("Phase After Modulation")
        axes[1, 3].set_xlabel("Patch")

        plt.suptitle("Inside WaveLayerComplex")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.close()

        # Print statistics
        print("\n=== WaveLayerComplex Statistics ===")
        print(f"Magnitude projection: mean={mag_proj.mean():.4f}, std={mag_proj.std():.4f}")
        print(f"Phase projection: mean={phase_proj.mean():.4f}, std={phase_proj.std():.4f}")
        print(f"Global complex mean magnitude: {torch.abs(c_global).mean():.4f}")
        print(f"After modulation magnitude: mean={torch.abs(c_modulated).mean():.4f}")


def main():
    """Run visualization on trained audio model."""
    save_dir = "data/results/wave_viz/audio"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Visualizing Trained Wave Audio STFT Model")
    print("=" * 60)

    # Load sample data
    print("\n--- Loading Sample Data ---")
    data, labels = load_sample_data(n_samples=16)
    print(f"Loaded {len(data)} samples")

    # Show one STFT input
    visualize_stft_input(data[0], labels[0].item(), f"{save_dir}/stft_input_example.png")

    # Create model (untrained for now - just to show architecture behavior)
    print("\n--- Creating Model ---")
    model = WaveAudioSTFT(
        num_classes=35, freq_bins=201, time_frames=101, embedding_dim=256, num_layers=4
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Visualize embeddings
    print("\n--- Visualizing Embeddings ---")
    visualize_embeddings(model, data, f"{save_dir}/embeddings.png")

    # Visualize layer activations
    print("\n--- Visualizing Layer Activations ---")
    visualize_layer_activations(model, data, f"{save_dir}/layer_activations.png")

    # Visualize complex layer internals
    print("\n--- Visualizing WaveLayerComplex Internals ---")
    visualize_complex_layer_internals(model, data[:1], f"{save_dir}/complex_layer_internals.png")

    # Visualize class patterns
    print("\n--- Visualizing Class Patterns ---")
    visualize_class_patterns(model, data, labels, f"{save_dir}/class_patterns.png")

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
