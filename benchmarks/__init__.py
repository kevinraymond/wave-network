"""Benchmarks package for Wave Network evaluation."""

from benchmarks.glue import (
    GLUE_TASKS,
    TASK_HYPERPARAMS as GLUE_HYPERPARAMS,
    GLUEDataset,
    GLUEMetrics,
    GLUETask,
    TaskType,
    get_task_info as get_glue_task_info,
    list_tasks as list_glue_tasks,
    load_glue_task,
    print_task_summary as print_glue_summary,
)

from benchmarks.vision import (
    VISION_TASKS,
    TASK_HYPERPARAMS as VISION_HYPERPARAMS,
    VisionDataset,
    VisionMetrics,
    VisionTask,
    get_task_info as get_vision_task_info,
    list_tasks as list_vision_tasks,
    load_vision_task,
    print_task_summary as print_vision_summary,
)

__all__ = [
    # GLUE
    "GLUE_TASKS",
    "GLUE_HYPERPARAMS",
    "TaskType",
    "GLUETask",
    "GLUEDataset",
    "GLUEMetrics",
    "load_glue_task",
    "get_glue_task_info",
    "list_glue_tasks",
    "print_glue_summary",
    # Vision
    "VISION_TASKS",
    "VISION_HYPERPARAMS",
    "VisionTask",
    "VisionDataset",
    "VisionMetrics",
    "load_vision_task",
    "get_vision_task_info",
    "list_vision_tasks",
    "print_vision_summary",
]
