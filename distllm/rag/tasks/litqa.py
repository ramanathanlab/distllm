"""LitQA evaluation task."""

from __future__ import annotations

import json
import random

from pydantic import BaseModel
from pydantic import Field

from distllm.rag.tasks.base import QuestionAnswerTask
from distllm.utils import curl_download


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
        """Build a multiple choice question from the entry.

        Returns
        -------
        str
            The multiple choice question.
        """
        # Pick 3 distractors at random
        distractors = random.sample(self.distractors, 3)

        # Collect the answer options
        options = [self.ideal, *distractors]

        # Shuffle the options
        random.shuffle(options)

        # Check if the question ends in a question mark
        mark = '' if self.question.endswith('?') else '?'

        # Format the multiple choice question
        # mc_question = (
        #     f'{self.question}{mark} Choose one of these '
        #     f'options: {",".join(options)}'
        # )

        mc_question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n'.format(
            f'{self.question}{mark}',
            *options,
        )

        return mc_question


class LitQATask(QuestionAnswerTask):
    """LitQA evaluation task."""

    # Name of the task
    task_name = 'litqa'

    def download(self) -> None:
        """Download the dataset."""
        # The URL to download the dataset
        download_url = 'https://raw.githubusercontent.com/Future-House/LitQA/main/litqa-v0.jsonl'

        # Set the path to the data file
        self.data_file = self.download_dir / 'litqa.jsonl'

        # Download the dataset
        curl_download(self.data_file, download_url)

    def load_data(self) -> tuple[list[str], list[str]]:
        """Load the data from the dataset."""
        # Read in the jsonl file containing the questions
        lines = self.data_file.read_text().strip().split('\n')

        # Parse the entries from the json lines
        entries = [QuestionAnswerEntry(**json.loads(line)) for line in lines]

        # Generate multiple choice questions
        questions = [entry.get_multiple_choice() for entry in entries]

        # Get the ground truth answers
        ground_truths = [entry.ideal for entry in entries]

        return questions, ground_truths
