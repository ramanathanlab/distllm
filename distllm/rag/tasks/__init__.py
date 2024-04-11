"""Module for evaluation tasks."""

from __future__ import annotations

from pathlib import Path

from distllm.rag.tasks.base import EvaluationTask
from distllm.rag.tasks.question_answer_task import LitQATask
from distllm.rag.tasks.question_answer_task import ProteinInteractionQATask

TASKS: dict[str, type[EvaluationTask]] = {
    'litqa': LitQATask,
    'protein_interaction_qa': ProteinInteractionQATask,
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
