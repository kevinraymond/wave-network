"""
Vision Benchmark Training Script

Train and evaluate Wave Vision models on image classification tasks.

Usage:
    # CIFAR-10
    python train_vision.py --task cifar10 --model wave_vision

    # CIFAR-100
    python train_vision.py --task cifar100 --model wave_vision

    # Custom configuration
    python train_vision.py --task cifar10 --model wave_vision --epochs 200 --lr 5e-4
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


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


try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from benchmarks.vision import (  # noqa: E402
    TASK_HYPERPARAMS,
    VISION_TASKS,
    VisionMetrics,
    load_vision_task,
    print_task_summary,
)
from models.wave_vision import WaveVisionNetwork, create_wave_vision  # noqa: E402
from models.wave_vision_2d import WaveVisionNetwork2D  # noqa: E402
from models.wave_vision_hybrid import CNNWaveVision  # noqa: E402


# Mixup helper functions
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to input data.

    Args:
        x: Input images (batch, C, H, W)
        y: Labels (batch,)
        alpha: Mixup interpolation strength

    Returns:
        Mixed images, labels_a, labels_b, lambda
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model registry
MODEL_REGISTRY = {
    "wave_vision": WaveVisionNetwork,
    "wave_vision_tiny": lambda **kwargs: create_wave_vision("wave_vision_tiny", **kwargs),
    "wave_vision_small": lambda **kwargs: create_wave_vision("wave_vision_small", **kwargs),
    "wave_vision_base": lambda **kwargs: create_wave_vision("wave_vision_base", **kwargs),
    "wave_vision_2d": WaveVisionNetwork2D,
    "cnn_wave": CNNWaveVision,
}

# Model-specific default configs
MODEL_CONFIGS = {
    "wave_vision": {
        "embedding_dim": 384,
        "num_layers": 3,
        "mode": "modulation",
        "dropout": 0.1,
    },
    "wave_vision_tiny": {},
    "wave_vision_small": {},
    "wave_vision_base": {},
    "wave_vision_2d": {
        "embedding_dim": 384,
        "num_layers": 3,
        "mode": "modulation",
        "dropout": 0.1,
    },
    "cnn_wave": {
        "base_channels": 64,
        "embedding_dim": 256,
        "num_wave_layers": 2,
        "mode": "modulation",
        "dropout": 0.2,
        "wave_dropout": 0.1,
    },
}


class VisionTrainer:
    """Trainer for vision benchmark tasks."""

    def __init__(
        self,
        model: nn.Module,
        task_name: str,
        device: str = "cuda",
        use_mlflow: bool = False,
        label_smoothing: float = 0.0,
    ):
        self.model = model.to(device)
        self.task_name = task_name
        self.task = VISION_TASKS[task_name]
        self.device = device
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE

        # Metrics
        self.metrics = VisionMetrics(task_name)

        # Loss function with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def train_epoch(
        self,
        train_loader,
        optimizer,
        scheduler=None,
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            # Apply mixup if enabled
            if getattr(self, "use_mixup", False):
                images, labels_a, labels_b, lam = mixup_data(images, labels, self.mixup_alpha)
                logits = self.model(images)
                loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            loss.backward()

            # Check for NaN/Inf gradients
            grad_ok = True
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning(f"Invalid gradient in {name}, skipping update")
                        grad_ok = False
                        break

            if not grad_ok:
                optimizer.zero_grad()
                continue

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        # Step scheduler after epoch (for cosine annealing)
        if scheduler is not None:
            scheduler.step()

        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        task_metrics = self.metrics.compute(all_preds, all_labels)

        return {"loss": avg_loss, **task_metrics}

    def evaluate(self, eval_loader) -> dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in eval_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs)

        avg_loss = total_loss / len(eval_loader)
        task_metrics = self.metrics.compute(all_preds, all_labels, all_probs)

        return {"loss": avg_loss, **task_metrics}

    def train(
        self,
        train_loader,
        test_loader,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.05,
        max_grad_norm: float = 1.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.2,
    ) -> dict[str, Any]:
        """Full training loop."""
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_val_metric = 0.0
        best_metrics = {}
        history = []

        # Determine primary metric
        primary_metric = self.task.metric_names[0]

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler, max_grad_norm)

            # Evaluate
            val_metrics = self.evaluate(test_loader)

            # Get current LR
            current_lr = scheduler.get_last_lr()[0]

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"LR: {current_lr:.2e}, "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Test Loss: {val_metrics['loss']:.4f}, "
                f"Test {primary_metric}: {val_metrics.get(primary_metric, 0):.4f}"
            )

            # MLflow logging
            if self.use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "test_loss": val_metrics["loss"],
                        "learning_rate": current_lr,
                    },
                    step=epoch + 1,
                )
                for k, v in val_metrics.items():
                    if k != "loss":
                        mlflow.log_metric(f"test_{k}", v, step=epoch + 1)

            # Track best
            current_metric = val_metrics.get(primary_metric, 0)
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_metrics = val_metrics.copy()

            history.append(
                {
                    "epoch": epoch + 1,
                    "learning_rate": current_lr,
                    "train": train_metrics,
                    "test": val_metrics,
                }
            )

        return {
            "best_test_metrics": best_metrics,
            "final_test_metrics": val_metrics,
            "history": history,
        }


def create_model(
    model_name: str,
    task_name: str,
) -> nn.Module:
    """Create a model instance."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    task = VISION_TASKS[task_name]
    params = TASK_HYPERPARAMS[task_name]

    model_class = MODEL_REGISTRY[model_name]
    config = MODEL_CONFIGS.get(model_name, {}).copy()

    config["image_size"] = task.image_size
    config["patch_size"] = params["patch_size"]
    config["num_classes"] = task.num_classes

    return model_class(**config)


def run_task(
    task_name: str,
    model_name: str,
    device: str = "cuda",
    use_mlflow: bool = False,
    output_dir: str = "data/results",
    learning_rate: Optional[float] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    use_randaugment: bool = False,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    label_smoothing: float = 0.0,
    experiment_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Run training and evaluation on a single vision task."""
    # Set seed for reproducibility
    if seed is not None:
        set_seed(seed)
        logger.info(f"Using seed: {seed}")

    logger.info(f"Running {task_name} with {model_name}")

    params = TASK_HYPERPARAMS[task_name].copy()

    # Override hyperparameters if specified
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
        logger.info(f"Using custom learning rate: {learning_rate}")
    if num_epochs is not None:
        params["num_epochs"] = num_epochs
        logger.info(f"Using custom epochs: {num_epochs}")
    if batch_size is not None:
        params["batch_size"] = batch_size
        logger.info(f"Using custom batch size: {batch_size}")
    if patch_size is not None:
        params["patch_size"] = patch_size
        logger.info(f"Using custom patch size: {patch_size}")

    # Log augmentation settings
    if use_randaugment:
        logger.info("Using RandAugment")
    if use_mixup:
        logger.info(f"Using Mixup with alpha={mixup_alpha}")
    if label_smoothing > 0:
        logger.info(f"Using label smoothing={label_smoothing}")

    # Load data with augmentation options
    data = load_vision_task(
        task_name=task_name,
        batch_size=params["batch_size"],
        use_randaugment=use_randaugment,
    )

    # Create model
    model = create_model(
        model_name=model_name,
        task_name=task_name,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Initialize MLflow
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name or "wave-vision")
        mlflow.start_run(run_name=f"{model_name}_{task_name}")
        mlflow.log_params(
            {
                "model": model_name,
                "task": task_name,
                "num_params": num_params,
                "seed": seed,
                "use_randaugment": use_randaugment,
                "use_mixup": use_mixup,
                "mixup_alpha": mixup_alpha,
                "label_smoothing": label_smoothing,
                **params,
            }
        )

    # Train
    trainer = VisionTrainer(
        model=model,
        task_name=task_name,
        device=device,
        use_mlflow=use_mlflow,
        label_smoothing=label_smoothing,
    )

    results = trainer.train(
        train_loader=data["train_loader"],
        test_loader=data["test_loader"],
        num_epochs=params["num_epochs"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
    )

    # Add metadata
    results["task"] = task_name
    results["model"] = model_name
    results["num_params"] = num_params
    results["seed"] = seed
    results["timestamp"] = datetime.now().isoformat()
    results["hyperparams"] = params
    results["augmentation"] = {
        "use_randaugment": use_randaugment,
        "use_mixup": use_mixup,
        "mixup_alpha": mixup_alpha,
        "label_smoothing": label_smoothing,
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        result_file = output_path / f"{model_name}_{task_name}_seed{seed}.json"
    else:
        result_file = output_path / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {result_file}")

    # Print summary
    print_result_summary(results)

    # Close MLflow
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_metric("best_accuracy", results["best_test_metrics"].get("accuracy", 0))
        mlflow.end_run()

    return results


def print_result_summary(results: dict[str, Any]):
    """Print a summary of training results."""
    print("\n" + "=" * 60)
    print(f"RESULTS - {results['model']} on {results['task']}")
    print("=" * 60)
    print(f"Parameters: {results['num_params']:,}")
    print("Best Test Metrics:")
    for k, v in results["best_test_metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train models on vision benchmarks")

    parser.add_argument(
        "--task",
        type=str,
        default="cifar10",
        help="Vision task name (cifar10, cifar100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wave_vision",
        help="Model to train",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Use MLflow logging",
    )
    parser.add_argument(
        "--randaugment",
        action="store_true",
        help="Use RandAugment data augmentation",
    )
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="Use Mixup data augmentation",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Mixup alpha parameter (default: 0.2)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (default: 0.0)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Override patch size (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.list_tasks:
        print_task_summary()
        return

    run_task(
        task_name=args.task,
        model_name=args.model,
        device=args.device,
        use_mlflow=args.mlflow,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        use_randaugment=args.randaugment,
        use_mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
