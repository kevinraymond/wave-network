"""
Vision Benchmark Integration

Provides task definitions, data loading, and evaluation for
image classification benchmarks.

Supported Datasets:
- CIFAR-10: 10-class natural images (32x32)
- CIFAR-100: 100-class natural images (32x32)
- TinyImageNet: 200-class subset of ImageNet (64x64)

References:
- CIFAR: https://www.cs.toronto.edu/~kriz/cifar.html
- TinyImageNet: https://tiny-imagenet.herokuapp.com/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision
    import torchvision.transforms as transforms

    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, top_k_accuracy_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class VisionTask:
    """Definition of a vision task."""

    name: str
    num_classes: int
    image_size: int
    metric_names: list[str]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    train_split: str = "train"
    test_split: str = "test"


# Vision Task Definitions
VISION_TASKS: dict[str, VisionTask] = {
    "cifar10": VisionTask(
        name="cifar10",
        num_classes=10,
        image_size=32,
        metric_names=["accuracy"],
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "cifar100": VisionTask(
        name="cifar100",
        num_classes=100,
        image_size=32,
        metric_names=["accuracy", "top5_accuracy"],
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
}


# Task-specific hyperparameters
TASK_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "num_epochs": 100,
        "patch_size": 4,
        "weight_decay": 0.05,
    },
    "cifar100": {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "num_epochs": 100,
        "patch_size": 4,
        "weight_decay": 0.05,
    },
}


def get_transforms(
    task: VisionTask,
    train: bool = True,
    augment: bool = True,
    use_randaugment: bool = False,
) -> transforms.Compose:
    """
    Get transforms for a vision task.

    Args:
        task: Vision task definition
        train: Whether this is for training
        augment: Whether to apply data augmentation
        use_randaugment: Whether to use RandAugment

    Returns:
        Composed transforms
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("Please install torchvision: pip install torchvision")

    transform_list = []

    if train and augment:
        # Training augmentations
        transform_list.extend(
            [
                transforms.RandomCrop(task.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

        # Add RandAugment if requested
        if use_randaugment:
            transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(task.mean, task.std),
        ]
    )

    return transforms.Compose(transform_list)


class VisionDataset(Dataset):
    """PyTorch Dataset wrapper for vision tasks."""

    def __init__(
        self,
        task_name: str,
        split: str = "train",
        augment: bool = True,
        use_randaugment: bool = False,
        root: str = "./data",
    ):
        """
        Args:
            task_name: Name of the vision task (e.g., 'cifar10', 'cifar100')
            split: Dataset split ('train' or 'test')
            augment: Whether to apply data augmentation (train only)
            use_randaugment: Whether to use RandAugment
            root: Root directory for dataset storage
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Please install torchvision: pip install torchvision")

        if task_name not in VISION_TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(VISION_TASKS.keys())}")

        self.task = VISION_TASKS[task_name]
        self.task_name = task_name
        is_train = split == "train"

        # Get transforms
        transform = get_transforms(
            self.task,
            train=is_train,
            augment=augment,
            use_randaugment=use_randaugment if is_train else False,
        )

        # Load dataset
        if task_name == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=root,
                train=is_train,
                download=True,
                transform=transform,
            )
        elif task_name == "cifar100":
            self.dataset = torchvision.datasets.CIFAR100(
                root=root,
                train=is_train,
                download=True,
                transform=transform,
            )
        else:
            raise ValueError(f"Dataset {task_name} not yet implemented")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[idx]
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_vision_task(
    task_name: str,
    batch_size: int | None = None,
    num_workers: int = 4,
    root: str = "./data",
    augment: bool = True,
    use_randaugment: bool = False,
) -> dict[str, Any]:
    """
    Load a vision task with train/test dataloaders.

    Args:
        task_name: Name of the vision task
        batch_size: Batch size (uses task default if None)
        num_workers: Number of data loading workers
        root: Root directory for dataset storage
        augment: Whether to apply data augmentation
        use_randaugment: Whether to use RandAugment

    Returns:
        Dictionary with 'train_loader', 'test_loader', 'task'
    """
    if task_name not in VISION_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(VISION_TASKS.keys())}")

    task = VISION_TASKS[task_name]
    params = TASK_HYPERPARAMS[task_name]

    if batch_size is None:
        batch_size = params["batch_size"]

    # Create datasets
    train_dataset = VisionDataset(
        task_name, "train", augment=augment, use_randaugment=use_randaugment, root=root
    )
    test_dataset = VisionDataset(task_name, "test", augment=False, root=root)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "task": task,
        "params": params,
    }


class VisionMetrics:
    """Compute vision task metrics."""

    def __init__(self, task_name: str):
        """
        Args:
            task_name: Name of the vision task
        """
        self.task = VISION_TASKS[task_name]
        self.task_name = task_name

    def compute(
        self,
        predictions: list[int],
        references: list[int],
        probabilities: list[list[float]] | None = None,
    ) -> dict[str, float]:
        """
        Compute metrics for predictions.

        Args:
            predictions: Model predictions (class indices)
            references: Ground truth labels
            probabilities: Optional prediction probabilities for top-k accuracy

        Returns:
            Dictionary of metric names to values
        """
        if not SKLEARN_AVAILABLE:
            return self._compute_fallback(predictions, references, probabilities)

        results = {}

        for metric_name in self.task.metric_names:
            if metric_name == "accuracy":
                results["accuracy"] = accuracy_score(references, predictions)
            elif metric_name == "top5_accuracy":
                if probabilities is not None:
                    results["top5_accuracy"] = top_k_accuracy_score(
                        references, probabilities, k=5, labels=range(self.task.num_classes)
                    )
                else:
                    results["top5_accuracy"] = 0.0  # Can't compute without probs

        return results

    def _compute_fallback(
        self,
        predictions: list[int],
        references: list[int],
        probabilities: list[list[float]] | None = None,
    ) -> dict[str, float]:
        """Fallback metric computation without sklearn."""
        results = {}

        if "accuracy" in self.task.metric_names:
            correct = sum(p == r for p, r in zip(predictions, references, strict=True))
            results["accuracy"] = correct / len(references) if references else 0.0

        if "top5_accuracy" in self.task.metric_names:
            if probabilities is not None:
                # Manual top-5 accuracy
                correct = 0
                for probs, ref in zip(probabilities, references, strict=True):
                    top5 = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
                    if ref in top5:
                        correct += 1
                results["top5_accuracy"] = correct / len(references) if references else 0.0
            else:
                results["top5_accuracy"] = 0.0

        return results


def get_task_info(task_name: str) -> dict[str, Any]:
    """
    Get information about a vision task.

    Args:
        task_name: Name of the vision task

    Returns:
        Dictionary with task information
    """
    if task_name not in VISION_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(VISION_TASKS.keys())}")

    task = VISION_TASKS[task_name]
    params = TASK_HYPERPARAMS[task_name]

    return {
        "name": task.name,
        "num_classes": task.num_classes,
        "image_size": task.image_size,
        "metrics": task.metric_names,
        "recommended_lr": params["learning_rate"],
        "recommended_batch_size": params["batch_size"],
        "recommended_epochs": params["num_epochs"],
        "recommended_patch_size": params["patch_size"],
    }


def list_tasks() -> list[str]:
    """List all available vision tasks."""
    return list(VISION_TASKS.keys())


def print_task_summary():
    """Print a summary of all vision tasks."""
    print("\n" + "=" * 70)
    print("VISION BENCHMARK TASKS")
    print("=" * 70)
    print(f"{'Task':<12} {'Classes':<10} {'Size':<8} {'Metrics'}")
    print("-" * 70)

    for name, task in VISION_TASKS.items():
        metrics_str = ", ".join(task.metric_names)
        print(
            f"{name:<12} {task.num_classes:<10} {task.image_size}x{task.image_size:<4} {metrics_str}"
        )

    print("=" * 70 + "\n")
