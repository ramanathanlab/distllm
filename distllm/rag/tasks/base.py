"""Interface for an evaluation task."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Protocol

from rag.response_synthesizer import RagGenerator


class EvaluationTask(Protocol):
    """Evaluation task."""

    def __init__(self, download_dir: Path) -> None:
        """Initialize the task.

        Parameters
        ----------
        download_dir : Path
            The directory to download the dataset to.
        """
        ...

    def evaluate(self, generator: RagGenerator) -> dict[str, Any]:
        """Evaluate the generator on the task.

        Parameters
        ----------
        generator : RagGenerator
            The RagGenerator to use for generating responses.

        Returns
        -------
        dict[str, Any]
            The evaluation results.
        """
        ...
