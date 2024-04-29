"""Protein function QA evaluation task."""

from __future__ import annotations

import json
import random

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

from distllm.rag.tasks.base import QuestionAnswerTask
from distllm.utils import BaseConfig
from distllm.utils import curl_download


class ProteinFunctionQAEntry(BaseModel):
    """Question entry format within a jsonl file.

    Uses the FunctionQA Benchmark at:
        https://github.com/ramanathanlab/AmpQA/blob/main/FunctionQA.jsonl
    """

    id: str = Field(..., description='The unique identifier for the question.')
    question: str = Field(..., description='The question to answer.')
    ideal: str = Field(..., description='The ideal answer to the question.')
    distractors: list[str] = Field(
        ...,
        description='The distractor answers to the question.',
    )
    source: str = Field(
        ...,
        description='The source for the question.',
    )

    @validator('ideal', pre=True, always=True)
    def lowercase_ideal(cls, value: str) -> str:  # noqa N805
        """Convert ideal answer to lowercase."""
        return value.lower()

    @validator('distractors', pre=True, each_item=True, always=True)
    def lowercase_distractors(cls, value: str) -> str:  # noqa N805
        """Convert each distractor to lowercase."""
        return value.lower()

    def get_multiple_choice(self) -> str:
        """Build a multiple choice question from the entry.

        Returns
        -------
        str
            The multiple choice question.
        """
        # Pick 3 distractors at random
        k = 3
        distractors = random.sample(
            self.distractors,
            min(k, len(self.distractors)),
        )
        if len(distractors) < k:
            # TODO: Think about this.
            distractors.extend([''] * (k - len(distractors)))

        # Collect the answer options
        options = [self.ideal, *distractors]

        # Shuffle the options
        random.shuffle(options)

        # Check if the question ends in a question mark
        mark = '' if self.question.endswith('?') else '?'

        mc_question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n'.format(
            f'{self.question}{mark}',
            *options,
        )

        return mc_question


class ProteinFunctionQAData(BaseConfig):
    """Format for the protein interaction QA task."""

    entries: list[ProteinFunctionQAEntry]


class ProteinFunctionQATask(QuestionAnswerTask):
    """Protein function QA evaluation task."""

    # Name of the task
    task_name = 'protein_function_qa'

    def download(self) -> None:
        """Download the dataset."""
        # URL to download
        download_url = 'https://raw.githubusercontent.com/ramanathanlab/AmpQA/main/FunctionQA.jsonl'

        # Set the path to the data file
        self.data_file = self.download_dir / 'functionQA.json'

        # Download the dataset
        curl_download(self.data_file, download_url)

    def load_data(self) -> tuple[list[str], list[str]]:
        """Load the data from the dataset."""
        # Read in the jsonl file containing the questions
        lines = self.data_file.read_text().strip().split('\n')

        # Parse the entries from the json lines
        entries = [
            ProteinFunctionQAEntry(**json.loads(line)) for line in lines
        ]

        # Remove all data where ideal is longer than 200 words
        max_len = 200
        entries = [
            entry for entry in entries if len(entry.ideal.split()) <= max_len
        ]

        # Generate multiple choice questions
        questions = [entry.get_multiple_choice() for entry in entries]

        # Get the ground truth answers
        ground_truths = [entry.ideal for entry in entries]

        return questions, ground_truths
