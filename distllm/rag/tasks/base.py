"""Interface for an evaluation task."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Protocol

from distllm.generate import get_prompt_template
from distllm.rag.response_synthesizer import RagGenerator


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


class QuestionAnswerTask(ABC):
    """Question answer evaluation task."""

    # Name of the task (should be overridden in subclasses)
    task_name = ''

    def __init__(self, download_dir: Path) -> None:
        """Initialize the task.

        Parameters
        ----------
        download_dir : Path
            The directory to download the dataset to.
        """
        # Ensure the download URL and task name are set
        if not self.task_name:
            raise NotImplementedError('task_name must be set in the subclass.')

        # Initialize the prompt template
        self.prompt_template = get_prompt_template({'name': 'question_answer'})

        # Set the download directory and data file
        self.download_dir = download_dir / self.task_name
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self) -> None:
        """Download the dataset."""
        ...

    @abstractmethod
    def load_data(self) -> tuple[list[str], list[str]]:
        """Load the data from the dataset.

        Returns
        -------
        tuple[list[str], list[str]]
            The questions and ground truth answers.
        """
        ...

    def compute_accuracy(
        self,
        ground_truths: list[str],
        preds: list[str],
    ) -> float:
        """Compute the accuracy of the model.

        Parameters
        ----------
        ground_truths : list[str]
            The ground truth answers.

        preds : list[str]
            The predicted answers.

        Returns
        -------
        float
            The accuracy of the model.
        """
        correct = sum(g == a for g, a in zip(ground_truths, preds))
        return correct / len(ground_truths)

    def compute_precision(
        self,
        ground_truths: list[str],
        preds: list[str],
    ) -> float:
        """Compute the precision of the model.

        Parameters
        ----------
        ground_truths : list[str]
            The ground truth answers.

        preds : list[str]
            The predicted answers.

        Returns
        -------
        float
            The precision of the model.
        """
        # TODO: write the 'not sure' answers to a file to investigate.
        sure_preds = [a for a in preds if a != 'I cannot answer.']
        precision = self.compute_accuracy(ground_truths, sure_preds)
        return precision

    def evaluate(self, generator: RagGenerator) -> dict[str, float]:
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
        # Download the dataset (skips if already downloaded)
        self.download()

        # Load the data from the dataset
        questions, ground_truths = self.load_data()

        # Generate answer predictions for the questions
        preds = generator.generate(questions, self.prompt_template)

        # Compute the accuracy and precision
        accuracy = self.compute_accuracy(ground_truths, preds)
        precision = self.compute_precision(ground_truths, preds)

        return {'accuracy': accuracy, 'precision': precision}
