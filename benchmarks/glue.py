"""
GLUE Benchmark Integration

Provides task definitions, data loading, and evaluation for the
General Language Understanding Evaluation (GLUE) benchmark.

GLUE Tasks:
- SST-2: Sentiment analysis (single sentence)
- CoLA: Linguistic acceptability (single sentence)
- MRPC: Paraphrase detection (sentence pair)
- QQP: Question pair similarity (sentence pair)
- QNLI: Question-sentence entailment (sentence pair)
- RTE: Textual entailment (sentence pair)
- MNLI: Multi-genre NLI (sentence pair, 3 classes)
- STS-B: Semantic similarity regression (sentence pair)

References:
- GLUE Paper: https://arxiv.org/abs/1804.07461
- HuggingFace: https://huggingface.co/datasets/nyu-mll/glue
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import evaluate

    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False


class TaskType(Enum):
    """Types of GLUE tasks."""

    SINGLE_SENTENCE = "single"
    SENTENCE_PAIR = "pair"
    REGRESSION = "regression"


@dataclass
class GLUETask:
    """Definition of a GLUE task."""

    name: str
    task_type: TaskType
    num_labels: int
    metric_names: list[str]
    text_columns: tuple[str, ...]
    label_column: str = "label"

    # For sentence pair tasks
    sentence1_key: str | None = None
    sentence2_key: str | None = None

    # Dataset split names
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"


# GLUE Task Definitions
GLUE_TASKS: dict[str, GLUETask] = {
    "sst2": GLUETask(
        name="sst2",
        task_type=TaskType.SINGLE_SENTENCE,
        num_labels=2,
        metric_names=["accuracy"],
        text_columns=("sentence",),
    ),
    "cola": GLUETask(
        name="cola",
        task_type=TaskType.SINGLE_SENTENCE,
        num_labels=2,
        metric_names=["matthews_correlation"],
        text_columns=("sentence",),
    ),
    "mrpc": GLUETask(
        name="mrpc",
        task_type=TaskType.SENTENCE_PAIR,
        num_labels=2,
        metric_names=["accuracy", "f1"],
        text_columns=("sentence1", "sentence2"),
        sentence1_key="sentence1",
        sentence2_key="sentence2",
    ),
    "qqp": GLUETask(
        name="qqp",
        task_type=TaskType.SENTENCE_PAIR,
        num_labels=2,
        metric_names=["accuracy", "f1"],
        text_columns=("question1", "question2"),
        sentence1_key="question1",
        sentence2_key="question2",
    ),
    "qnli": GLUETask(
        name="qnli",
        task_type=TaskType.SENTENCE_PAIR,
        num_labels=2,
        metric_names=["accuracy"],
        text_columns=("question", "sentence"),
        sentence1_key="question",
        sentence2_key="sentence",
    ),
    "rte": GLUETask(
        name="rte",
        task_type=TaskType.SENTENCE_PAIR,
        num_labels=2,
        metric_names=["accuracy"],
        text_columns=("sentence1", "sentence2"),
        sentence1_key="sentence1",
        sentence2_key="sentence2",
    ),
    "mnli": GLUETask(
        name="mnli",
        task_type=TaskType.SENTENCE_PAIR,
        num_labels=3,
        metric_names=["accuracy"],
        text_columns=("premise", "hypothesis"),
        sentence1_key="premise",
        sentence2_key="hypothesis",
        validation_split="validation_matched",  # MNLI has matched/mismatched
    ),
    "stsb": GLUETask(
        name="stsb",
        task_type=TaskType.REGRESSION,
        num_labels=1,
        metric_names=["pearson", "spearmanr"],
        text_columns=("sentence1", "sentence2"),
        sentence1_key="sentence1",
        sentence2_key="sentence2",
    ),
}


# Task-specific hyperparameters (based on BERT fine-tuning recommendations)
TASK_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "sst2": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "max_length": 128,
        "num_epochs": 3,
    },
    "cola": {
        "learning_rate": 5e-4,
        "batch_size": 32,
        "max_length": 128,
        "num_epochs": 5,
    },
    "mrpc": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "max_length": 128,
        "num_epochs": 5,
    },
    "qqp": {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "max_length": 128,
        "num_epochs": 3,
    },
    "qnli": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "max_length": 256,
        "num_epochs": 3,
    },
    "rte": {
        "learning_rate": 5e-4,
        "batch_size": 16,
        "max_length": 256,
        "num_epochs": 10,  # Small dataset, needs more epochs
    },
    "mnli": {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "max_length": 256,
        "num_epochs": 3,
    },
    "stsb": {
        "learning_rate": 5e-4,
        "batch_size": 32,
        "max_length": 128,
        "num_epochs": 5,
    },
}


class GLUEDataset(Dataset):
    """PyTorch Dataset for GLUE tasks."""

    def __init__(
        self,
        task_name: str,
        split: str,
        tokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            task_name: Name of the GLUE task (e.g., 'sst2', 'mrpc')
            split: Dataset split ('train', 'validation', 'test')
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Please install datasets: pip install datasets")

        self.task = GLUE_TASKS[task_name]
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Handle MNLI special split names
        if task_name == "mnli" and split == "validation":
            split = "validation_matched"

        # Load dataset
        self.dataset = load_dataset("glue", task_name, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Get text(s)
        if self.task.task_type == TaskType.SINGLE_SENTENCE:
            text = item[self.task.text_columns[0]]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            # Sentence pair
            text1 = item[self.task.sentence1_key]
            text2 = item[self.task.sentence2_key]
            encoding = self.tokenizer(
                text1,
                text2,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Get label
        label = item[self.task.label_column]

        # Handle regression (STS-B)
        if self.task.task_type == TaskType.REGRESSION:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
        }


def load_glue_task(
    task_name: str,
    tokenizer,
    batch_size: int | None = None,
    max_length: int | None = None,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """
    Load a GLUE task with train/validation/test dataloaders.

    Args:
        task_name: Name of the GLUE task
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size (uses task default if None)
        max_length: Max sequence length (uses task default if None)
        num_workers: Number of data loading workers

    Returns:
        Dictionary with 'train_loader', 'val_loader', 'test_loader'
    """
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(GLUE_TASKS.keys())}")

    task = GLUE_TASKS[task_name]
    params = TASK_HYPERPARAMS[task_name]

    if batch_size is None:
        batch_size = params["batch_size"]
    if max_length is None:
        max_length = params["max_length"]

    # Create datasets
    train_dataset = GLUEDataset(task_name, "train", tokenizer, max_length)
    val_dataset = GLUEDataset(task_name, "validation", tokenizer, max_length)

    # Test set may not have labels (for submission)
    try:
        test_dataset = GLUEDataset(task_name, "test", tokenizer, max_length)
    except Exception:
        test_dataset = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "task": task,
    }

    if test_dataset is not None:
        result["test_loader"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return result


class GLUEMetrics:
    """Compute GLUE task metrics."""

    def __init__(self, task_name: str):
        """
        Args:
            task_name: Name of the GLUE task
        """
        self.task = GLUE_TASKS[task_name]
        self.task_name = task_name

        # Load metrics
        if EVALUATE_AVAILABLE:
            self.metric = evaluate.load("glue", task_name)
        else:
            self.metric = None

    def compute(
        self,
        predictions: list[int],
        references: list[int],
    ) -> dict[str, float]:
        """
        Compute metrics for predictions.

        Args:
            predictions: Model predictions
            references: Ground truth labels

        Returns:
            Dictionary of metric names to values
        """
        if self.metric is not None:
            return self.metric.compute(predictions=predictions, references=references)

        # Fallback implementation
        return self._compute_fallback(predictions, references)

    def _compute_fallback(
        self,
        predictions: list[int],
        references: list[int],
    ) -> dict[str, float]:
        """Fallback metric computation without evaluate library."""
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            matthews_corrcoef,
        )

        results = {}

        for metric_name in self.task.metric_names:
            if metric_name == "accuracy":
                results["accuracy"] = accuracy_score(references, predictions)
            elif metric_name == "f1":
                results["f1"] = f1_score(references, predictions, average="binary")
            elif metric_name == "matthews_correlation":
                results["matthews_correlation"] = matthews_corrcoef(references, predictions)
            elif metric_name == "pearson":
                results["pearson"] = pearsonr(references, predictions)[0]
            elif metric_name == "spearmanr":
                results["spearmanr"] = spearmanr(references, predictions)[0]

        return results


def get_task_info(task_name: str) -> dict[str, Any]:
    """
    Get information about a GLUE task.

    Args:
        task_name: Name of the GLUE task

    Returns:
        Dictionary with task information
    """
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(GLUE_TASKS.keys())}")

    task = GLUE_TASKS[task_name]
    params = TASK_HYPERPARAMS[task_name]

    return {
        "name": task.name,
        "type": task.task_type.value,
        "num_labels": task.num_labels,
        "metrics": task.metric_names,
        "text_columns": task.text_columns,
        "recommended_lr": params["learning_rate"],
        "recommended_batch_size": params["batch_size"],
        "recommended_max_length": params["max_length"],
        "recommended_epochs": params["num_epochs"],
    }


def list_tasks() -> list[str]:
    """List all available GLUE tasks."""
    return list(GLUE_TASKS.keys())


def print_task_summary():
    """Print a summary of all GLUE tasks."""
    print("\n" + "=" * 80)
    print("GLUE BENCHMARK TASKS")
    print("=" * 80)
    print(f"{'Task':<8} {'Type':<12} {'Labels':<8} {'Metrics':<25} {'Columns'}")
    print("-" * 80)

    for name, task in GLUE_TASKS.items():
        metrics_str = ", ".join(task.metric_names)
        cols_str = ", ".join(task.text_columns)
        print(
            f"{name:<8} {task.task_type.value:<12} {task.num_labels:<8} {metrics_str:<25} {cols_str}"
        )

    print("=" * 80 + "\n")
