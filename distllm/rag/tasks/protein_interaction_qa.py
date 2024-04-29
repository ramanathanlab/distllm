"""Protein interaction QA evaluation task."""

from __future__ import annotations

import json

from distllm.rag.tasks.base import QuestionAnswerTask
from distllm.rag.tasks.litqa import QuestionAnswerEntry
from distllm.utils import BaseConfig
from distllm.utils import curl_download


class ProteinInteractionQAData(BaseConfig):
    """Format for the protein interaction QA task."""

    entries: list[QuestionAnswerEntry]


class ProteinInteractionQATask(QuestionAnswerTask):
    """Protein interaction QA evaluation task."""

    # Name of the task
    task_name = 'protein_interaction_qa'

    def download(self) -> None:
        """Download the dataset."""
        # URL to download
        download_url = 'https://raw.githubusercontent.com/ramanathanlab/AmpQA/main/interactionQA.json'

        # Set the path to the data file
        self.data_file = self.download_dir / 'interactionQA.json'

        # Download the dataset
        curl_download(self.data_file, download_url)

    def load_data(self) -> tuple[list[str], list[str]]:
        """Load the data from the dataset."""
        # Read in the json file containing the questions
        with open(self.data_file) as fp:
            data = json.load(fp)

        # Parse the entries from the json
        data = ProteinInteractionQAData(entries=data)

        # Remove all data where 'ideal' is longer than 200 words.
        max_len = 200
        entries = [
            entry
            for entry in data.entries
            if len(entry.ideal.split()) <= max_len
        ]

        # Generate multiple choice questions
        questions = [entry.get_multiple_choice() for entry in entries]

        # Get the ground truth answers
        ground_truths = [entry.ideal for entry in entries]

        return questions, ground_truths
