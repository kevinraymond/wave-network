"""
Ablation study runner.

Orchestrates running multiple experiments for ablation studies,
handling seed management, W&B grouping, and results aggregation.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from experiments.config_manager import (
    AblationConfig,
    ExperimentConfig,
    create_model_from_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AblationRunner:
    """
    Runs ablation studies across multiple configurations.

    Features:
    - Automatic seed management
    - W&B integration with experiment grouping
    - Results saving to JSON
    - Summary statistics generation
    """

    def __init__(
        self,
        ablation_config: AblationConfig,
        device: str | None = None,
    ):
        """
        Args:
            ablation_config: Configuration for the ablation study
            device: Device to run on (default: auto-detect)
        """
        self.config = ablation_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Setup output directory
        self.output_dir = Path(ablation_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: list[dict[str, Any]] = []

    def run_single_experiment(
        self,
        exp_config: ExperimentConfig,
        data_loaders: dict,
    ) -> dict[str, Any]:
        """
        Run a single experiment.

        Args:
            exp_config: Experiment configuration
            data_loaders: Dict with train_loader, val_loader, test_loader

        Returns:
            Dictionary with experiment results
        """
        # Set seed
        set_seed(exp_config.training.seed)

        logger.info(f"Running experiment: {exp_config.name}")
        logger.info(f"  Model: {exp_config.model.model_type}")
        logger.info(f"  Seed: {exp_config.training.seed}")

        # Initialize W&B if enabled
        if exp_config.logging.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=exp_config.logging.wandb_project,
                entity=exp_config.logging.wandb_entity,
                name=exp_config.name,
                config=exp_config.to_dict(),
                group=self.config.group_name,
                tags=exp_config.tags,
                reinit=True,
            )

        # Create model
        model = create_model_from_config(exp_config.model)
        model = model.to(self.device)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {num_params:,}")

        # Training
        metrics = self._train_and_evaluate(
            model=model,
            config=exp_config,
            data_loaders=data_loaders,
        )

        # Close W&B
        if exp_config.logging.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Build result
        result = {
            "name": exp_config.name,
            "config": exp_config.to_dict(),
            "metrics": metrics,
            "num_params": num_params,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def _train_and_evaluate(
        self,
        model: torch.nn.Module,
        config: ExperimentConfig,
        data_loaders: dict,
    ) -> dict[str, float]:
        """
        Train model and return metrics.

        This is a simplified training loop. For production, use the full
        Trainer class from train.py.
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        metrics = {}

        for epoch in range(config.training.num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for batch in data_loaders["train_loader"]:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.max_grad_norm,
                )
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(logits.argmax(dim=1).cpu().tolist())
                train_labels.extend(labels.cpu().tolist())

            train_acc = accuracy_score(train_labels, train_preds)

            # Validation
            model.eval()
            val_preds, val_labels = [], []

            with torch.no_grad():
                for batch in data_loaders["val_loader"]:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = model(input_ids, attention_mask)
                    val_preds.extend(logits.argmax(dim=1).cpu().tolist())
                    val_labels.extend(labels.cpu().tolist())

            val_acc = accuracy_score(val_labels, val_preds)
            val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average="weighted", zero_division=0
            )

            logger.info(
                f"  Epoch {epoch + 1}/{config.training.num_epochs}: "
                f"train_loss={train_loss / len(data_loaders['train_loader']):.4f}, "
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
            )

            # Log to W&B
            if config.logging.use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss / len(data_loaders["train_loader"]),
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                    }
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                metrics = {
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "val_f1": val_f1,
                }

        # Test evaluation
        if "test_loader" in data_loaders:
            model.eval()
            test_preds, test_labels = [], []

            with torch.no_grad():
                for batch in data_loaders["test_loader"]:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["label"].to(self.device)

                    logits = model(input_ids, attention_mask)
                    test_preds.extend(logits.argmax(dim=1).cpu().tolist())
                    test_labels.extend(labels.cpu().tolist())

            test_acc = accuracy_score(test_labels, test_preds)
            test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
                test_labels, test_preds, average="weighted", zero_division=0
            )

            metrics.update(
                {
                    "test_acc": test_acc,
                    "test_precision": test_prec,
                    "test_recall": test_rec,
                    "test_f1": test_f1,
                }
            )

            logger.info(f"  Test: acc={test_acc:.4f}, f1={test_f1:.4f}")

        return metrics

    def run_all(self, data_loaders: dict) -> list[dict[str, Any]]:
        """
        Run all experiments in the ablation study.

        Args:
            data_loaders: Dict with train_loader, val_loader, test_loader

        Returns:
            List of all experiment results
        """
        experiments = self.config.generate_experiments()
        logger.info(f"Running {len(experiments)} experiments")

        for exp_config in experiments:
            result = self.run_single_experiment(exp_config, data_loaders)
            self.results.append(result)
            self._save_result(result)

        # Generate summary
        self._generate_summary()

        return self.results

    def _save_result(self, result: dict[str, Any]) -> None:
        """Save individual result to JSON file."""
        filename = f"{result['name']}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"  Saved: {filepath}")

    def _generate_summary(self) -> None:
        """Generate summary statistics across all runs."""
        from collections import defaultdict

        # Group by base name (without seed)
        grouped = defaultdict(list)
        for result in self.results:
            # Extract base name (remove _seedXXX suffix)
            name = result["name"]
            if "_seed" in name:
                base_name = name.rsplit("_seed", 1)[0]
            else:
                base_name = name
            grouped[base_name].append(result["metrics"])

        # Compute statistics
        summary = {}
        for name, metrics_list in grouped.items():
            if not metrics_list:
                continue

            # Get all metric keys
            metric_keys = metrics_list[0].keys()

            summary[name] = {
                "n_runs": len(metrics_list),
            }

            for key in metric_keys:
                values = [m.get(key, 0) for m in metrics_list if key in m]
                if values:
                    summary[name][f"{key}_mean"] = np.mean(values)
                    summary[name][f"{key}_std"] = np.std(values)

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {summary_path}")

        # Print summary table
        self._print_summary_table(summary)

    def _print_summary_table(self, summary: dict) -> None:
        """Print summary as formatted table."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print("=" * 80)

        # Header
        print(f"{'Configuration':<30} {'Val Acc':<15} {'Test Acc':<15} {'N':<5}")
        print("-" * 80)

        # Rows
        for name, stats in sorted(summary.items()):
            val_acc = stats.get("val_acc_mean", 0)
            val_std = stats.get("val_acc_std", 0)
            test_acc = stats.get("test_acc_mean", 0)
            test_std = stats.get("test_acc_std", 0)
            n = stats.get("n_runs", 0)

            print(
                f"{name:<30} "
                f"{val_acc:.4f} +/- {val_std:.4f}  "
                f"{test_acc:.4f} +/- {test_std:.4f}  "
                f"{n:<5}"
            )

        print("=" * 80 + "\n")


def run_ablation_from_yaml(
    config_path: str,
    data_loaders: dict,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to run ablation study from YAML config.

    Args:
        config_path: Path to ablation YAML config
        data_loaders: Dict with train_loader, val_loader, test_loader
        device: Device to run on

    Returns:
        List of experiment results
    """
    config = AblationConfig.from_yaml(config_path)
    runner = AblationRunner(config, device=device)
    return runner.run_all(data_loaders)
