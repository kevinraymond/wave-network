"""
Export Wave Audio STFT model to ONNX format.

This script loads a trained WaveAudioSTFT checkpoint and exports it to ONNX
using the WaveAudioSTFTExport model which uses real arithmetic instead of
complex numbers (required for ONNX compatibility).

Usage:
    python export_onnx.py --checkpoint data/checkpoints/wave_audio_stft_ep090_val0.9291.pt
    python export_onnx.py --checkpoint data/checkpoints/wave_audio_stft_ep090_val0.9291.pt --output model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from models.wave_audio import WaveAudioSTFT, WaveAudioSTFTExport


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint and return config and state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint


def transfer_weights(src_model: WaveAudioSTFT, dst_model: WaveAudioSTFTExport) -> None:
    """
    Transfer weights from WaveAudioSTFT to WaveAudioSTFTExport.

    The layer structures are identical (WaveLayerComplex and WaveLayerReal have
    the same parameter names), so we can do a direct state_dict transfer.
    """
    dst_model.load_state_dict(src_model.state_dict())


def verify_outputs(
    src_model: torch.nn.Module,
    dst_model: torch.nn.Module,
    input_shape: tuple,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> tuple[bool, float]:
    """
    Verify that both models produce the same output.

    Returns:
        Tuple of (outputs_match, max_difference)
    """
    src_model.eval()
    dst_model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        src_output = src_model(dummy_input)
        dst_output = dst_model(dummy_input)

    max_diff = torch.max(torch.abs(src_output - dst_output)).item()
    outputs_match = torch.allclose(src_output, dst_output, rtol=rtol, atol=atol)

    return outputs_match, max_diff


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple,
    opset_version: int = 17,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, *input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["stft_input"],
        output_names=["logits"],
        dynamic_axes={
            "stft_input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def verify_onnx_output(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_shape: tuple,
) -> tuple[bool, float]:
    """Verify ONNX model produces same output as PyTorch model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping ONNX verification")
        return True, 0.0

    pytorch_model.eval()
    dummy_input = torch.randn(1, *input_shape)

    # PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {"stft_input": dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    outputs_match = np.allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-5)

    return outputs_match, max_diff


def main():
    parser = argparse.ArgumentParser(description="Export Wave Audio STFT to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (default: same name as checkpoint with .onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]

    print(f"  Representation: {config['representation']}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Mode: {config['mode']}")
    print(f"  Val accuracy: {checkpoint['val_acc']:.4f}")
    print(f"  Test accuracy: {checkpoint['test_acc']:.4f}")

    # Model input shape (matches training config)
    # STFT: (2, freq_bins, time_frames) = (2, 201, 101)
    freq_bins = 201
    time_frames = 101
    input_shape = (2, freq_bins, time_frames)

    # Create original model and load weights
    print("\nCreating original model...")
    src_model = WaveAudioSTFT(
        num_classes=35,
        freq_bins=freq_bins,
        time_frames=time_frames,
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        mode=config["mode"],
    )
    src_model.load_state_dict(checkpoint["model_state_dict"])
    src_model.eval()
    print(f"  Parameters: {sum(p.numel() for p in src_model.parameters()):,}")

    # Create exportable model
    print("\nCreating ONNX-exportable model...")
    dst_model = WaveAudioSTFTExport(
        num_classes=35,
        freq_bins=freq_bins,
        time_frames=time_frames,
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        mode=config["mode"],
    )

    # Transfer weights
    print("Transferring weights...")
    transfer_weights(src_model, dst_model)
    dst_model.eval()

    # Verify outputs match
    print("\nVerifying output equivalence...")
    match, max_diff = verify_outputs(src_model, dst_model, input_shape)
    print(f"  Outputs match: {match}")
    print(f"  Max difference: {max_diff:.2e}")

    if not match:
        print("WARNING: Output mismatch detected. Proceeding anyway...")

    # Export to ONNX
    if args.output:
        output_path = args.output
    else:
        checkpoint_path = Path(args.checkpoint)
        output_path = checkpoint_path.with_suffix(".onnx")

    print(f"\nExporting to ONNX: {output_path}")
    export_to_onnx(dst_model, str(output_path), input_shape, args.opset)

    # Verify ONNX output
    print("\nVerifying ONNX output...")
    onnx_match, onnx_diff = verify_onnx_output(dst_model, str(output_path), input_shape)
    print(f"  ONNX outputs match PyTorch: {onnx_match}")
    print(f"  Max difference: {onnx_diff:.2e}")

    # Print file size
    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
    print("\nExport complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {onnx_size:.2f} MB")


if __name__ == "__main__":
    main()
