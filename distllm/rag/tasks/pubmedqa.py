"""PubmedQA evaluation task."""

from __future__ import annotations

import json

from pydantic import BaseModel
from pydantic import Field

from distllm.rag.tasks.base import QuestionAnswerTask
from distllm.utils import BaseConfig
from distllm.utils import curl_download


class PubmedQAEntry(BaseModel):
    """Question entry format within a jsonl file.

    Uses the PubmedQA Benchmark at:
        https://https://github.com/pubmedqa/pubmedqa
    """

    QUESTION: str = Field(..., description='The text of the question.')

    CONTEXTS: list[str] = Field(
        ...,
        description='Context paragraphs related to the question.',
    )

    final_decision: str = Field(
        ...,
        description='Correct answer for the question.',
    )

    def get_multiple_choice(self) -> str:
        """Build a multiple choice question from the entry.

        Returns
        -------
        str
            The multiple choice question.
        """
        # Check if the question ends in a question mark
        mark = '' if self.QUESTION.endswith('?') else '?'

        # All PubmedQA answers are "yes", "no" or "maybe"
        options = ['yes', 'no', 'maybe']

        # Get the PubmedQA-provided contexts and join them.
        joined_contexts = '\n'.join(self.CONTEXTS)

        # Indicator of PubmedQA provided ground truth context
        gt_context_indicator = 'Most relevant context:'

        # Add the PubmedQA provided context to the question
        mc_question = '{}\n{}\n{}\nOptions:\n1. {}\n2. {}\n3. {}\n'.format(
            f'{gt_context_indicator}',
            f'{joined_contexts}',
            f'{self.QUESTION}{mark}',
            *options,
        )
        return mc_question


class PubmedQAData(BaseConfig):
    """Format for the PubmedQA task."""

    entries: list[PubmedQAEntry]


class PubmedQATask(QuestionAnswerTask):
    """PubmedQA evaluation task."""

    # Name of the task
    task_name = 'pubmedqa'

    def download(self) -> None:
        """Download the dataset."""
        # URL to download
        download_url = 'https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json'

        # Set the path to the data file
        self.data_file = self.download_dir / 'pubmedQA.json'

        # Download the dataset
        curl_download(self.data_file, download_url)

    def load_data(self) -> tuple[list[str], list[str]]:
        """Load the data from the dataset."""
        # Read in the json file containing the questions
        with open(self.data_file) as fp:
            data = json.load(fp)

            # Pack the entries into a list that PubmedQAData expects.
            data = list(data.values())

        # Parse the entries from the json
        data = PubmedQAData(entries=data)

        # Get the data entries
        entries = list(data.entries)

        # Generate multiple choice questions
        questions = [entry.get_multiple_choice() for entry in entries]

        # Get the ground truth answers
        ground_truths = [entry.final_decision for entry in entries]

        return questions, ground_truths
