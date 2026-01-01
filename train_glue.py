"""
GLUE Benchmark Training Script

Train and evaluate models on GLUE tasks.

Usage:
    # Single task
    python train_glue.py --task sst2 --model wave_network

    # All tasks
    python train_glue.py --task all --model wave_network

    # Compare models
    python train_glue.py --task sst2 --model wave_network fnet
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from benchmarks.glue import (
    GLUE_TASKS,
    TASK_HYPERPARAMS,
    GLUEMetrics,
    TaskType,
    list_tasks,
    load_glue_task,
    print_task_summary,
)
from models.fnet import FNet, FNetLite
from wave_network import WaveNetwork
from wave_network_deep import DeepWaveNetwork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Model registry
MODEL_REGISTRY = {
    "wave_network": WaveNetwork,
    "deep_wave_network": DeepWaveNetwork,
    "fnet": FNet,
    "fnet_lite": FNetLite,
}

# Model-specific default configs
MODEL_CONFIGS = {
    "wave_network": {
        "embedding_dim": 768,
        "mode": "modulation",
    },
    "deep_wave_network": {
        "embedding_dim": 768,
        "num_layers": 3,
        "mode": "modulation",
    },
    "fnet": {
        "embedding_dim": 768,
        "num_layers": 6,
    },
    "fnet_lite": {
        "embedding_dim": 768,
        "num_layers": 3,
    },
}


class GLUETrainer:
    """Trainer for GLUE benchmark tasks."""

    def __init__(
        self,
        model: nn.Module,
        task_name: str,
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.task_name = task_name
        self.task = GLUE_TASKS[task_name]
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Metrics
        self.metrics = GLUEMetrics(task_name)

        # Loss function
        if self.task.task_type == TaskType.REGRESSION:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

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
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask)

            # Handle regression
            if self.task.task_type == TaskType.REGRESSION:
                logits = logits.squeeze(-1)
                loss = self.criterion(logits, labels.float())
                preds = logits.detach().cpu().tolist()
            else:
                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=-1).cpu().tolist()

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
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        # Compute metrics
        avg_loss = total_loss / len(train_loader)

        # For regression, we need to round for some metrics
        if self.task.task_type == TaskType.REGRESSION:
            # Keep as-is for correlation metrics
            metrics = {"loss": avg_loss}
        else:
            task_metrics = self.metrics.compute(all_preds, all_labels)
            metrics = {"loss": avg_loss, **task_metrics}

        return metrics

    def evaluate(self, eval_loader) -> dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)

                if self.task.task_type == TaskType.REGRESSION:
                    logits = logits.squeeze(-1)
                    loss = self.criterion(logits, labels.float())
                    preds = logits.cpu().tolist()
                else:
                    loss = self.criterion(logits, labels)
                    preds = logits.argmax(dim=-1).cpu().tolist()

                total_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(eval_loader)
        task_metrics = self.metrics.compute(all_preds, all_labels)

        return {"loss": avg_loss, **task_metrics}

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
    ) -> dict[str, Any]:
        """Full training loop."""
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_metric = 0.0
        best_metrics = {}
        history = []

        # Determine primary metric
        primary_metric = self.task.metric_names[0]

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler, max_grad_norm)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val {primary_metric}: {val_metrics.get(primary_metric, 0):.4f}"
            )

            # W&B logging
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                }
                for k, v in val_metrics.items():
                    if k != "loss":
                        log_dict[f"val_{k}"] = v
                wandb.log(log_dict)

            # Track best
            current_metric = val_metrics.get(primary_metric, 0)
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_metrics = val_metrics.copy()

            history.append(
                {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

        return {
            "best_val_metrics": best_metrics,
            "final_val_metrics": val_metrics,
            "history": history,
        }


def create_model(
    model_name: str,
    num_labels: int,
    vocab_size: int = 30522,
) -> nn.Module:
    """Create a model instance."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    config = MODEL_CONFIGS[model_name].copy()
    config["vocab_size"] = vocab_size
    config["num_classes"] = num_labels

    return model_class(**config)


def run_task(
    task_name: str,
    model_name: str,
    device: str = "cuda",
    use_wandb: bool = False,
    output_dir: str = "data/results",
    learning_rate: Optional[float] = None,
) -> dict[str, Any]:
    """Run training and evaluation on a single GLUE task."""
    logger.info(f"Running {task_name} with {model_name}")

    task = GLUE_TASKS[task_name]
    params = TASK_HYPERPARAMS[task_name].copy()

    # Override learning rate if specified
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
        logger.info(f"Using custom learning rate: {learning_rate}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    data = load_glue_task(
        task_name=task_name,
        tokenizer=tokenizer,
        batch_size=params["batch_size"],
        max_length=params["max_length"],
    )

    # Create model
    model = create_model(
        model_name=model_name,
        num_labels=task.num_labels,
        vocab_size=tokenizer.vocab_size,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="wave-network-glue",
            name=f"{model_name}_{task_name}",
            config={
                "model": model_name,
                "task": task_name,
                "num_params": num_params,
                **params,
            },
            reinit=True,
        )

    # Train
    trainer = GLUETrainer(
        model=model,
        task_name=task_name,
        device=device,
        use_wandb=use_wandb,
    )

    results = trainer.train(
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        num_epochs=params["num_epochs"],
        learning_rate=params["learning_rate"],
    )

    # Add metadata
    results["task"] = task_name
    results["model"] = model_name
    results["num_params"] = num_params
    results["timestamp"] = datetime.now().isoformat()

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {result_file}")

    # Close W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return results


def run_all_tasks(
    model_name: str,
    tasks: Optional[list[str]] = None,
    device: str = "cuda",
    use_wandb: bool = False,
    output_dir: str = "data/results",
    learning_rate: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """Run on multiple GLUE tasks."""
    if tasks is None:
        tasks = list_tasks()

    all_results = {}

    for task_name in tasks:
        try:
            results = run_task(
                task_name=task_name,
                model_name=model_name,
                device=device,
                use_wandb=use_wandb,
                output_dir=output_dir,
                learning_rate=learning_rate,
            )
            all_results[task_name] = results
        except Exception as e:
            logger.error(f"Failed on {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}

    # Print summary
    print_results_summary(all_results, model_name)

    return all_results


def print_results_summary(results: dict[str, dict], model_name: str):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print(f"GLUE RESULTS - {model_name}")
    print("=" * 70)
    print(f"{'Task':<10} {'Primary Metric':<20} {'Value':<10} {'Params'}")
    print("-" * 70)

    for task_name, result in results.items():
        if "error" in result:
            print(f"{task_name:<10} {'ERROR':<20} {result['error'][:30]}")
            continue

        best = result.get("best_val_metrics", {})
        task = GLUE_TASKS[task_name]
        metric_name = task.metric_names[0]
        metric_value = best.get(metric_name, 0)
        num_params = result.get("num_params", 0)

        print(f"{task_name:<10} {metric_name:<20} {metric_value:.4f}     {num_params:,}")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train models on GLUE benchmark")

    parser.add_argument(
        "--task",
        type=str,
        default="sst2",
        help="GLUE task name or 'all' for all tasks",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["wave_network"],
        help="Model(s) to train",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging",
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
        help="Override learning rate (default: use task-specific)",
    )

    args = parser.parse_args()

    if args.list_tasks:
        print_task_summary()
        return

    # Determine tasks
    if args.task.lower() == "all":
        tasks = list_tasks()
    else:
        tasks = [args.task]

    # Run for each model
    for model_name in args.model:
        logger.info(f"Training {model_name}")

        if len(tasks) == 1:
            run_task(
                task_name=tasks[0],
                model_name=model_name,
                device=args.device,
                use_wandb=args.wandb,
                output_dir=args.output_dir,
                learning_rate=args.lr,
            )
        else:
            run_all_tasks(
                model_name=model_name,
                tasks=tasks,
                device=args.device,
                use_wandb=args.wandb,
                output_dir=args.output_dir,
                learning_rate=args.lr,
            )


if __name__ == "__main__":
    main()
