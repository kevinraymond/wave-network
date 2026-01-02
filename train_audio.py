"""
Audio Benchmark Training Script

Train and evaluate Wave Audio models on speech classification tasks.

Usage:
    # Raw waveform
    python train_audio.py --representation waveform

    # Mel spectrogram
    python train_audio.py --representation melspec

    # Complex STFT
    python train_audio.py --representation stft

    # Custom configuration
    python train_audio.py --representation waveform --epochs 50 --lr 1e-3
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from benchmarks.audio import (
    SPEECH_COMMANDS_LABELS,
    AudioConfig,
    get_audio_dataloaders,
    get_input_shape,
)
from models.wave_audio import create_wave_audio_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if scheduler is not None and hasattr(scheduler, "step"):
            pass  # Step at end of epoch for CosineAnnealingLR

    if scheduler is not None:
        scheduler.step()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return total_loss / total, correct / total


def save_results(results: dict, output_path: str):
    """Save results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Wave Audio models")
    parser.add_argument(
        "--representation",
        type=str,
        default="waveform",
        choices=["waveform", "melspec", "stft"],
        help="Audio representation",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--mlflow", action="store_true", default=True, help="Log to MLflow")
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Audio configuration
    audio_config = AudioConfig(representation=args.representation)
    input_shape = get_input_shape(audio_config)
    logger.info(f"Representation: {args.representation}")
    logger.info(f"Input shape: {input_shape}")

    # Load data
    logger.info("Loading Speech Commands dataset...")
    train_loader, val_loader, test_loader = get_audio_dataloaders(
        config=audio_config,
        batch_size=args.batch_size,
        num_workers=4,
    )
    logger.info(
        f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}"
    )

    # Create model
    num_classes = len(SPEECH_COMMANDS_LABELS)
    model_kwargs = {
        "num_classes": num_classes,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
    }

    # Add representation-specific kwargs
    if args.representation == "melspec":
        model_kwargs["freq_bins"] = input_shape[0]
        model_kwargs["time_frames"] = input_shape[1]
    elif args.representation == "stft":
        model_kwargs["freq_bins"] = input_shape[1]
        model_kwargs["time_frames"] = input_shape[2]

    model = create_wave_audio_model(args.representation, **model_kwargs)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # MLflow setup
    if args.mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("wave-audio-experiments")
        mlflow.start_run(run_name=f"wave_audio_{args.representation}")
        mlflow.log_params(
            {
                "representation": args.representation,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "embedding_dim": args.embedding_dim,
                "num_layers": args.num_layers,
                "seed": args.seed,
                "num_params": num_params,
            }
        )

    # Training loop
    best_val_acc = 0
    best_test_metrics = None
    history = []

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Track history
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_metrics)

        # MLflow logging
        if args.mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                },
                step=epoch,
            )

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_metrics = {"loss": test_loss, "accuracy": test_acc}

        # Log progress
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: {train_loss:.4f} / {train_acc:.4f} | "
            f"Val: {val_loss:.4f} / {val_acc:.4f} | "
            f"Test: {test_loss:.4f} / {test_acc:.4f}"
        )

    # Final results
    results = {
        "representation": args.representation,
        "best_val_acc": best_val_acc,
        "best_test_metrics": best_test_metrics,
        "final_test_metrics": {"loss": test_loss, "accuracy": test_acc},
        "history": history,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "seed": args.seed,
        },
        "num_params": num_params,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = f"data/results/wave_audio_{args.representation}_seed{args.seed}.json"
    save_results(results, output_path)

    # MLflow cleanup
    if args.mlflow and MLFLOW_AVAILABLE:
        mlflow.log_metrics(
            {
                "best_val_acc": best_val_acc,
                "best_test_acc": best_test_metrics["accuracy"],
            }
        )
        mlflow.end_run()

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best test accuracy: {best_test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
