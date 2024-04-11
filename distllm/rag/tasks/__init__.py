"""Module for evaluation tasks."""

from __future__ import annotations

from pathlib import Path

from distllm.rag.tasks.base import EvaluationTask
from distllm.rag.tasks.litqa_task import LitQATask

TASKS: dict[str, type[EvaluationTask]] = {
    'litqa': LitQATask,
}


def get_task(name: str, download_dir: Path) -> EvaluationTask:
    """Get the evaluation task.

    Parameters
    ----------
    name : str
        The name of the task.
    download_dir : Path
        The directory to download the dataset to.

    Returns
    -------
    EvaluationTask
        The task.
    """
    return TASKS[name](download_dir)
