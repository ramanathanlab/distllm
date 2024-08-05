"""SciQ evaluation task."""

from __future__ import annotations

import json

from pydantic import BaseModel
from pydantic import Field

from distllm.rag.tasks.base import QuestionAnswerTask
from distllm.utils import BaseConfig
from distllm.utils import curl_download


class SciQEntry(BaseModel):
    """Question entry format within a jsonl file.

    Uses the Allen AI SciQ benchmark at:
        https://allenai.org/data/sciq
    """

    question: str = Field(..., description='The text of the question.')
    distractor3: str = Field(..., description='The third distractor answer.')
    distractor2: str = Field(..., description='The second distractor answer.')
    distractor1: str = Field(..., description='The first distractor answer.')
    correct_answer: str = Field(
        ...,
        description='The correct answer to the question.',
    )
    support: str = Field(
        ...,
        description=' A sentence that supports the correct answer option..',
    )

    def get_multiple_choice(self) -> str:
        """Build a multiple choice question from the entry.

        Returns
        -------
        str
            The multiple choice question.
        """
        # Check if the question ends in a question mark
        mark = '' if self.question.endswith('?') else '?'

        # Collect the answer options
        options = [
            self.correct_answer,
            self.distractor1,
            self.distractor2,
            self.distractor3,
        ]

        # SciQ provides correct answer explanation
        # Commenting this out because not sure if we can include it in prompt.
        # additional_context = self.support

        # Build the multiple choice question
        mc_question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n'.format(
            # Commenting this out because not sure if we can include it.
            # f"{additional_context}",
            f'{self.question}{mark}',
            *options,
        )

        return mc_question


class SciQData(BaseConfig):
    """Format for the protein interaction QA task."""

    entries: list[SciQEntry]


class SciQTask(QuestionAnswerTask):
    """SciQ evaluation task."""

    # Name of the task
    task_name = 'sciq'

    def download(self) -> None:
        """Download the dataset."""
        # URL to download
        download_url = 'https://raw.githubusercontent.com/ogkdmr/sciqa_questions/main/test.json'

        # Set the path to the data file
        self.data_file = self.download_dir / 'sciq.json'

        # Download the dataset
        curl_download(self.data_file, download_url)

    def load_data(self) -> tuple[list[str], list[str]]:
        """Load the data from the dataset."""
        # Read in the json file containing the questions
        with open(self.data_file) as fp:
            data = json.load(fp)

        # Parse the entries from the json
        data = SciQData(entries=data)

        # Get the data entries
        entries = data.entries

        # Generate multiple choice questions
        questions = [entry.get_multiple_choice() for entry in entries]

        # Get the ground truth answers
        ground_truths = [entry.correct_answer for entry in entries]

        return questions, ground_truths
