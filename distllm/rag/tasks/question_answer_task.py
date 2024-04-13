"""LitQA evaluation task."""

from __future__ import annotations

import json
import random
import subprocess
from abc import ABC
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from distllm.generate import get_prompt_template
from distllm.rag.response_synthesizer import RagGenerator


class QuestionAnswerEntry(BaseModel):
    """Question entry format within a jsonl file.

    Uses the LitQA Benchmark format: https://github.com/Future-House/LitQA
    """

    id: str = Field(..., description='The unique identifier for the question.')
    question: str = Field(..., description='The question to answer.')
    ideal: str = Field(..., description='The ideal answer to the question.')
    distractors: list[str] = Field(
        ...,
        description='The distractor answers to the question.',
    )
    sources: list[str] = Field(
        ...,
        description='The sources for the question.',
    )

    def get_multiple_choice(self) -> str:
        """Build a multiple choice question from the entry."""
        options = [self.ideal, *self.distractors]
        random.shuffle(options)
        return f"{self.question}? Choose one of these options: {','.join(options)}"  # noqa E501


class QuestionAnswerTask(ABC):
    """LitQA evaluation task."""

    # URL to download the dataset (should be overridden in subclasses)
    download_url = ''

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
        if not self.download_url or not self.task_name:
            raise NotImplementedError(
                'download_url and task_name must be set in the subclass.',
            )

        # Initialize the prompt template
        self.prompt_template = get_prompt_template({'name': 'question_answer'})

        # Set the download directory and data file
        download_dir = download_dir / self.task_name
        download_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = download_dir / f'{self.task_name}.jsonl'

    def download(self) -> None:
        """Download the dataset."""
        if not self.data_file.exists():
            command = f'curl -o {self.data_file} {self.download_url}'
            subprocess.run(command.split(), check=False)

    def _compute_accuracy(
        self,
        ground_truths: list[str],
        preds: list[str],
    ) -> float:
        """Compute the accuracy of the model."""
        correct = sum(g == a for g, a in zip(ground_truths, preds))
        return correct / len(ground_truths)

    def _compute_precision(
        self,
        ground_truths: list[str],
        preds: list[str],
    ) -> float:
        """Compute the precision of the model."""
        # TODO: write the 'not sure' answers to a file to investigate.
        sure_preds = [a for a in preds if a != 'I cannot answer.']
        precision = self._compute_accuracy(ground_truths, sure_preds)
        return precision

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
        # Download the dataset (skips if already downloaded)
        self.download()

        # Read in the jsonl file containing the questions
        lines = self.data_file.read_text().strip().split('\n')

        # Parse the entries from the json lines
        entries = [QuestionAnswerEntry(**json.loads(line)) for line in lines]

        # Generate multiple choice questions
        questions = [entry.get_multiple_choice() for entry in entries]

        # Get the ground truth answers
        ground_truths = [entry.ideal for entry in entries]

        # Generate answer predictions for the questions
        preds = generator.generate(questions, self.prompt_template)

        # Compute the accuracy and precision
        accuracy = self._compute_accuracy(ground_truths, preds)
        precision = self._compute_precision(ground_truths, preds)

        return {'accuracy': accuracy, 'precision': precision}


class LitQATask(QuestionAnswerTask):
    """LitQA evaluation task."""

    # URL to download
    download_url = 'https://raw.githubusercontent.com/Future-House/LitQA/main/litqa-v0.jsonl'
    # Name of the task
    task_name = 'litqa'


class ProteinInteractionQATask(QuestionAnswerTask):
    """Protein interaction QA evaluation task."""

    # URL to download
    download_url = 'https://raw.githubusercontent.com/ramanathanlab/AmpQA/main/interactionQA.json'
    # Name of the task
    task_name = 'protein_interaction_qa'
