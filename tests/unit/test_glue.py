"""Unit tests for GLUE benchmark integration."""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.glue import (
    GLUE_TASKS,
    TASK_HYPERPARAMS,
    GLUEMetrics,
    GLUETask,
    TaskType,
    get_task_info,
    list_tasks,
)


class TestGLUETasks:
    """Test GLUE task definitions."""

    def test_all_tasks_defined(self):
        """Test that all 8 GLUE tasks are defined."""
        expected_tasks = ["sst2", "cola", "mrpc", "qqp", "qnli", "rte", "mnli", "stsb"]
        for task in expected_tasks:
            assert task in GLUE_TASKS, f"Missing task: {task}"

    def test_task_types(self):
        """Test that task types are correctly assigned."""
        # Single sentence tasks
        assert GLUE_TASKS["sst2"].task_type == TaskType.SINGLE_SENTENCE
        assert GLUE_TASKS["cola"].task_type == TaskType.SINGLE_SENTENCE

        # Sentence pair tasks
        assert GLUE_TASKS["mrpc"].task_type == TaskType.SENTENCE_PAIR
        assert GLUE_TASKS["qqp"].task_type == TaskType.SENTENCE_PAIR
        assert GLUE_TASKS["qnli"].task_type == TaskType.SENTENCE_PAIR
        assert GLUE_TASKS["rte"].task_type == TaskType.SENTENCE_PAIR
        assert GLUE_TASKS["mnli"].task_type == TaskType.SENTENCE_PAIR

        # Regression
        assert GLUE_TASKS["stsb"].task_type == TaskType.REGRESSION

    def test_num_labels(self):
        """Test that num_labels is correct for each task."""
        assert GLUE_TASKS["sst2"].num_labels == 2
        assert GLUE_TASKS["cola"].num_labels == 2
        assert GLUE_TASKS["mrpc"].num_labels == 2
        assert GLUE_TASKS["qqp"].num_labels == 2
        assert GLUE_TASKS["qnli"].num_labels == 2
        assert GLUE_TASKS["rte"].num_labels == 2
        assert GLUE_TASKS["mnli"].num_labels == 3  # 3-way classification
        assert GLUE_TASKS["stsb"].num_labels == 1  # Regression

    def test_metrics_defined(self):
        """Test that metrics are defined for each task."""
        for task_name, task in GLUE_TASKS.items():
            assert len(task.metric_names) > 0, f"No metrics for {task_name}"

    def test_sentence_pair_keys(self):
        """Test that sentence pair tasks have proper keys."""
        for task_name, task in GLUE_TASKS.items():
            if task.task_type == TaskType.SENTENCE_PAIR:
                assert task.sentence1_key is not None, f"Missing sentence1_key for {task_name}"
                assert task.sentence2_key is not None, f"Missing sentence2_key for {task_name}"


class TestTaskHyperparams:
    """Test task-specific hyperparameters."""

    def test_all_tasks_have_params(self):
        """Test that all tasks have hyperparameters defined."""
        for task_name in GLUE_TASKS:
            assert task_name in TASK_HYPERPARAMS, f"Missing params for {task_name}"

    def test_required_params_present(self):
        """Test that required parameters are present."""
        required = ["learning_rate", "batch_size", "max_length", "num_epochs"]
        for task_name, params in TASK_HYPERPARAMS.items():
            for param in required:
                assert param in params, f"Missing {param} for {task_name}"

    def test_reasonable_values(self):
        """Test that hyperparameters have reasonable values."""
        for _task_name, params in TASK_HYPERPARAMS.items():
            assert 1e-6 < params["learning_rate"] < 1e-1
            assert 8 <= params["batch_size"] <= 256
            assert 32 <= params["max_length"] <= 512
            assert 1 <= params["num_epochs"] <= 20


class TestGLUEMetrics:
    """Test GLUE metrics computation."""

    def test_accuracy_metric(self):
        """Test accuracy computation."""
        metrics = GLUEMetrics("sst2")
        predictions = [0, 1, 1, 0, 1]
        references = [0, 1, 0, 0, 1]

        result = metrics.compute(predictions, references)
        assert "accuracy" in result
        assert result["accuracy"] == 0.8  # 4/5 correct

    def test_matthews_correlation(self):
        """Test Matthews correlation for CoLA."""
        metrics = GLUEMetrics("cola")
        predictions = [0, 1, 1, 0, 1, 0]
        references = [0, 1, 0, 0, 1, 1]

        result = metrics.compute(predictions, references)
        assert "matthews_correlation" in result

    def test_f1_metric(self):
        """Test F1 computation for MRPC."""
        metrics = GLUEMetrics("mrpc")
        predictions = [1, 1, 0, 1, 0]
        references = [1, 0, 0, 1, 1]

        result = metrics.compute(predictions, references)
        assert "accuracy" in result
        assert "f1" in result


class TestUtilityFunctions:
    """Test utility functions."""

    def test_list_tasks(self):
        """Test listing tasks."""
        tasks = list_tasks()
        assert len(tasks) == 8
        assert "sst2" in tasks
        assert "mnli" in tasks

    def test_get_task_info(self):
        """Test getting task info."""
        info = get_task_info("sst2")

        assert info["name"] == "sst2"
        assert info["type"] == "single"
        assert info["num_labels"] == 2
        assert "accuracy" in info["metrics"]
        assert info["recommended_lr"] > 0
        assert info["recommended_batch_size"] > 0

    def test_get_task_info_invalid(self):
        """Test error handling for invalid task."""
        with pytest.raises(ValueError):
            get_task_info("invalid_task")


class TestGLUETask:
    """Test GLUETask dataclass."""

    def test_task_creation(self):
        """Test creating a custom task."""
        task = GLUETask(
            name="test",
            task_type=TaskType.SINGLE_SENTENCE,
            num_labels=2,
            metric_names=["accuracy"],
            text_columns=("text",),
        )

        assert task.name == "test"
        assert task.task_type == TaskType.SINGLE_SENTENCE
        assert task.num_labels == 2

    def test_task_defaults(self):
        """Test task default values."""
        task = GLUETask(
            name="test",
            task_type=TaskType.SINGLE_SENTENCE,
            num_labels=2,
            metric_names=["accuracy"],
            text_columns=("text",),
        )

        assert task.label_column == "label"
        assert task.train_split == "train"
        assert task.validation_split == "validation"
        assert task.test_split == "test"


class TestTaskType:
    """Test TaskType enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert TaskType.SINGLE_SENTENCE.value == "single"
        assert TaskType.SENTENCE_PAIR.value == "pair"
        assert TaskType.REGRESSION.value == "regression"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert TaskType.SINGLE_SENTENCE == TaskType.SINGLE_SENTENCE
        assert TaskType.SINGLE_SENTENCE != TaskType.SENTENCE_PAIR
