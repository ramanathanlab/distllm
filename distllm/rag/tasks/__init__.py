"""Module for evaluation tasks."""

from __future__ import annotations

from pathlib import Path

from distllm.rag.tasks.base import EvaluationTask
from distllm.rag.tasks.litqa import LitQATask
from distllm.rag.tasks.protein_function_qa import ProteinFunctionQATask
from distllm.rag.tasks.protein_interaction_qa import ProteinInteractionQATask
from distllm.rag.tasks.pubmedqa import PubmedQATask
from distllm.rag.tasks.sciq import SciQTask

TASKS: dict[str, type[EvaluationTask]] = {
    'sciq': SciQTask,
    'pubmedqa': PubmedQATask,
    'litqa': LitQATask,
    'protein_interaction_qa': ProteinInteractionQATask,
    'protein_function_qa': ProteinFunctionQATask,
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
