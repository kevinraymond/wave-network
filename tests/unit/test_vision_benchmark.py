"""Unit tests for Vision benchmark integration."""

import pytest


class TestVisionTasks:
    """Test Vision task definitions."""

    def test_all_tasks_defined(self):
        """Test that vision tasks are defined."""
        from benchmarks.vision import VISION_TASKS

        assert "cifar10" in VISION_TASKS
        assert "cifar100" in VISION_TASKS

    def test_task_properties(self):
        """Test task properties are correct."""
        from benchmarks.vision import VISION_TASKS

        cifar10 = VISION_TASKS["cifar10"]
        assert cifar10.num_classes == 10
        assert cifar10.image_size == 32
        assert "accuracy" in cifar10.metric_names

        cifar100 = VISION_TASKS["cifar100"]
        assert cifar100.num_classes == 100
        assert cifar100.image_size == 32

    def test_task_normalization_values(self):
        """Test that normalization values are defined."""
        from benchmarks.vision import VISION_TASKS

        for task in VISION_TASKS.values():
            assert len(task.mean) == 3
            assert len(task.std) == 3
            # Values should be in valid range
            for m in task.mean:
                assert 0 <= m <= 1
            for s in task.std:
                assert 0 < s <= 1


class TestTaskHyperparams:
    """Test task hyperparameters."""

    def test_all_tasks_have_params(self):
        """Test that all tasks have hyperparameters defined."""
        from benchmarks.vision import TASK_HYPERPARAMS, VISION_TASKS

        for task_name in VISION_TASKS:
            assert task_name in TASK_HYPERPARAMS

    def test_required_params_present(self):
        """Test that required hyperparameters are present."""
        from benchmarks.vision import TASK_HYPERPARAMS

        required_params = ["learning_rate", "batch_size", "num_epochs", "weight_decay"]

        for task_name, params in TASK_HYPERPARAMS.items():
            for param in required_params:
                assert param in params, f"Missing {param} for {task_name}"

    def test_reasonable_values(self):
        """Test that hyperparameter values are reasonable."""
        from benchmarks.vision import TASK_HYPERPARAMS

        for params in TASK_HYPERPARAMS.values():
            assert 0 < params["learning_rate"] < 1
            assert params["batch_size"] > 0
            assert params["num_epochs"] > 0


class TestVisionTask:
    """Test VisionTask dataclass."""

    def test_task_creation(self):
        """Test creating a VisionTask."""
        from benchmarks.vision import VisionTask

        task = VisionTask(
            name="test_task",
            num_classes=10,
            image_size=32,
            metric_names=["accuracy"],
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

        assert task.name == "test_task"
        assert task.num_classes == 10
        assert task.image_size == 32

    def test_task_defaults(self):
        """Test default values for VisionTask."""
        from benchmarks.vision import VisionTask

        task = VisionTask(
            name="test",
            num_classes=10,
            image_size=32,
            metric_names=["accuracy"],
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

        assert task.train_split == "train"
        assert task.test_split == "test"


class TestVisionMetrics:
    """Test VisionMetrics class."""

    def test_metrics_creation(self):
        """Test creating VisionMetrics."""
        from benchmarks.vision import VisionMetrics

        metrics = VisionMetrics("cifar10")
        assert metrics.task_name == "cifar10"

    def test_compute_accuracy(self):
        """Test computing accuracy metric."""
        from benchmarks.vision import VisionMetrics

        metrics = VisionMetrics("cifar10")

        # Perfect predictions
        predictions = [0, 1, 2, 3, 4]
        references = [0, 1, 2, 3, 4]

        result = metrics.compute(predictions, references)

        assert "accuracy" in result
        assert result["accuracy"] == 1.0

    def test_compute_partial_accuracy(self):
        """Test computing partial accuracy."""
        from benchmarks.vision import VisionMetrics

        metrics = VisionMetrics("cifar10")

        # 50% accuracy
        predictions = [0, 1, 2, 3, 4]
        references = [0, 1, 0, 0, 0]

        result = metrics.compute(predictions, references)

        assert "accuracy" in result
        assert 0 < result["accuracy"] < 1


class TestUtilityFunctions:
    """Test utility functions."""

    def test_list_tasks(self):
        """Test list_tasks function."""
        from benchmarks.vision import list_tasks

        tasks = list_tasks()
        assert "cifar10" in tasks
        assert "cifar100" in tasks

    def test_get_task_info(self):
        """Test get_task_info function."""
        from benchmarks.vision import get_task_info

        info = get_task_info("cifar10")
        assert "name" in info
        assert "num_classes" in info
        assert info["num_classes"] == 10

    def test_get_task_info_invalid(self):
        """Test get_task_info with invalid task."""
        from benchmarks.vision import get_task_info

        with pytest.raises(ValueError, match="Unknown task"):
            get_task_info("invalid_task")
