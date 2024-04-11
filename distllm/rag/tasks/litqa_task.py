"""LitQA evaluation task."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from distllm.generate import get_prompt_template
from distllm.rag.response_synthesizer import RagGenerator


class LitQAEntry(BaseModel):
    """Encapsulation for a LitQA benchmark entry."""

    id: str = Field(..., description='The unique identifier for the question.')
    question: str = Field(..., description='The question to answer.')
    ideal: str = Field(..., description='The ideal answer to the question.')
    distractors: list[str] = Field(
        ...,
        description='The distractor answers to the question.',
    )
    sources: str = Field(..., description='The sources for the question.')

    def get_multiple_choice(self) -> str:
        """Build a multiple choice question from the LitQA entry."""
        options = [self.ideal, *self.distractors]
        random.shuffle(options)
        return f"{self.question}? Choose one of these options: {','.join(options)}"  # noqa E501


class LitQATask:
    """LitQA evaluation task."""

    # URL to download the LitQA dataset
    download_url = 'https://raw.githubusercontent.com/Future-House/LitQA/main/litqa-v0.jsonl'

    def __init__(self, download_dir: Path) -> None:
        """Initialize the LitQATask.

        Parameters
        ----------
        download_dir : Path
            The directory to download the dataset to.
        """
        # Initialize the prompt template
        self.prompt_template = get_prompt_template({'name': 'question_answer'})

        # Set the download directory and data file
        download_dir = download_dir / 'litqa'
        download_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = download_dir / 'litqa-v0.jsonl'

    def download(self) -> None:
        """Download the LitQA dataset."""
        if not self.data_file.exists():
            command = f'curl -o {self.data_file} {self.download_url}'
            subprocess.run(command.split(), check=False)

    def _compute_accuracy(
        self,
        ground_truths: list[str],
        preds: list[str],
    ) -> float:
        """Compute the accuracy of the model on the LitQA task."""
        correct = sum(g == a for g, a in zip(ground_truths, preds))
        return correct / len(ground_truths)

    def _compute_precision(
        self,
        ground_truths: list[str],
        preds: list[str],
    ) -> float:
        """Compute the precision of the model on the LitQA task."""
        sure_preds = [a for a in preds if a != 'I cannot answer.']
        precision = self._compute_accuracy(ground_truths, sure_preds)
        # TODO: write the 'not sure' answers to a file to investigate.
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
        entries = [LitQAEntry(**json.loads(line)) for line in lines]

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
