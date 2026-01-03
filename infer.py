#!/usr/bin/env python3
"""
Speech Commands Inference CLI

Classify spoken words from audio files using the Wave Network model.

Usage:
    python infer.py audio.wav
    python infer.py audio1.wav audio2.wav audio3.wav
    python infer.py --top 5 audio.wav
    python infer.py --pytorch audio.wav  # Use PyTorch instead of ONNX
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Speech Commands labels
LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]

# Audio config
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160

# Default model paths
DEFAULT_ONNX = "data/checkpoints/wave_audio_stft_ep090_val0.9291.onnx"
DEFAULT_CHECKPOINT = "data/checkpoints/wave_audio_stft_ep090_val0.9291.pt"


def preprocess_audio(audio_path: str) -> np.ndarray:
    """Load and preprocess audio file to STFT representation."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Pad or trim to 1 second
    target_length = SAMPLE_RATE
    if waveform.shape[1] < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_length]

    # Compute STFT
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        return_complex=True,
    )

    # Stack magnitude and phase
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    x = torch.stack([magnitude, phase], dim=0)

    return x.unsqueeze(0).numpy()  # (1, 2, freq, time)


def load_onnx_model(model_path: str):
    """Load ONNX model."""
    import onnxruntime as ort

    return ort.InferenceSession(model_path)


def load_pytorch_model(checkpoint_path: str):
    """Load PyTorch model."""
    from models.wave_audio import WaveAudioSTFTExport

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = WaveAudioSTFTExport(
        num_classes=35,
        freq_bins=201,
        time_frames=101,
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        mode=config["mode"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def infer_onnx(session, x: np.ndarray) -> np.ndarray:
    """Run ONNX inference."""
    logits = session.run(None, {"stft_input": x.astype(np.float32)})[0]
    return logits


def infer_pytorch(model, x: np.ndarray) -> np.ndarray:
    """Run PyTorch inference."""
    with torch.no_grad():
        logits = model(torch.from_numpy(x.astype(np.float32)))
    return logits.numpy()


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def print_prediction(audio_path: str, logits: np.ndarray, top_k: int = 3):
    """Print prediction results."""
    probs = softmax(logits[0])
    top_indices = np.argsort(probs)[::-1][:top_k]

    pred_label = LABELS[top_indices[0]]
    pred_conf = probs[top_indices[0]]

    # Print filename and top prediction
    print(f"\n{Path(audio_path).name}")
    print(f"  Prediction: {pred_label} ({pred_conf:.1%})")

    # Print top-k if requested
    if top_k > 1:
        print(f"  Top {top_k}:")
        for i, idx in enumerate(top_indices):
            bar_len = int(probs[idx] * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            print(f"    {i+1}. {LABELS[idx]:10s} [{bar}] {probs[idx]:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify spoken words from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py recording.wav
  python infer.py --top 5 recording.wav
  python infer.py *.wav
  python infer.py --pytorch recording.wav

Supported words (35 classes):
  backward, bed, bird, cat, dog, down, eight, five, follow, forward,
  four, go, happy, house, learn, left, marvin, nine, no, off, on, one,
  right, seven, sheila, six, stop, three, tree, two, up, visual, wow,
  yes, zero
        """,
    )
    parser.add_argument("audio_files", nargs="+", help="Audio file(s) to classify")
    parser.add_argument("--top", type=int, default=1, help="Show top N predictions (default: 1)")
    parser.add_argument("--pytorch", action="store_true", help="Use PyTorch instead of ONNX")
    parser.add_argument("--model", type=str, default=None, help="Path to model file")
    parser.add_argument("--quiet", action="store_true", help="Only print predictions")

    args = parser.parse_args()

    # Determine model path
    if args.model:
        model_path = args.model
    elif args.pytorch:
        model_path = DEFAULT_CHECKPOINT
    else:
        model_path = DEFAULT_ONNX

    # Check model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        print("Run 'python export_onnx.py --checkpoint <checkpoint>' first.", file=sys.stderr)
        sys.exit(1)

    # Load model
    if not args.quiet:
        backend = "PyTorch" if args.pytorch else "ONNX"
        print(f"Loading {backend} model...")

    if args.pytorch:
        model = load_pytorch_model(model_path)
    else:
        session = load_onnx_model(model_path)

    # Process each audio file
    for audio_path in args.audio_files:
        if not Path(audio_path).exists():
            print(f"Error: File not found: {audio_path}", file=sys.stderr)
            continue

        try:
            x = preprocess_audio(audio_path)
            if args.pytorch:
                logits = infer_pytorch(model, x)
            else:
                logits = infer_onnx(session, x)
            print_prediction(audio_path, logits, args.top)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
