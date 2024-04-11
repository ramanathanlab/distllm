"""Script for running the LitQA evaluation task."""

from __future__ import annotations

import json
import random
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from distllm.utils import BaseConfig


class LitQAPromptTemplateConfig(BaseConfig):
    """Configuration for the LitQAPromptTemplate."""

    name: Literal['litqa'] = 'litqa'  # type: ignore[assignment]


class LitQAPromptTemplate:
    """LitQA prompt template."""

    template: str = (
        'Answer the question below with the context.\n\n'
        'Context (with relevance scores):\n\n{context}\n\n----\n\n'
        'Question: {question}\n\n'
        'Write an answer based on the context. '
        'If the context provides insufficient information and '
        'the question cannot be directly answered, reply '
        '"I cannot answer."'
        ' Write in the style of a Wikipedia article,'
        ' with concise sentences and coherent paragraphs. '
        'The context comes from a variety of sources and is only a summary, '
        'so there may inaccuracies or ambiguities. If quotes are present and '
        'relevant, use them in the answer. This answer will go directly onto '
        'Wikipedia, so do not add any extraneous information.\n\n'
        'Answer: '
    )

    def __init__(self, config: LitQAPromptTemplateConfig) -> None:
        """Initialize the IdentityPromptTemplate."""
        self.config = config

    def _format_prompt(
        self,
        question: str,
        context: list[str],
        score: list[float],
    ) -> str:
        """Format the prompt with the question and context."""
        context_concat = '\n'.join(
            [f'Context: {c}, score: {s}' for c, s in zip(context, score)],
        )
        return self.template.format(context=context_concat, question=question)

    def preprocess(
        self,
        text: str | list[str],  # question
        contexts: list[list[str]] | None = None,
        scores: list[list[float]] | None = None,
    ) -> list[str]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to format.
        contexts : list[list[str]], optional
            The contexts to include for each text, by default None.
        scores : list[list[float]], optional
            The scores for each context, by default None.


        Returns
        -------
        list[str]
            The formatted prompts.
        """
        # early exit if contexts and scores could not be retrieved.
        if contexts is None or scores is None:
            raise ValueError('Contexts and scores were not provided.')

        # batchify single question
        if isinstance(text, str):
            text = [text]

        # build the prompts using the template.
        prompts = [
            self._format_prompt(q, c, s)
            for q, c, s in zip(text, contexts, scores)
        ]

        return prompts

    def postprocess(self, responses: list[str]) -> list[str]:
        """Postprocess the responses.

        Parameters
        ----------
        responses : list[str]
            The responses to postprocess.

        Returns
        -------
        list[str]
            The postprocessed responses.
        """
        return responses


class LitQAEntry(BaseModel):
    """Encapsulation for a LitQA benchmark entry."""

    id: str = Field(description='The unique identifier for the question.')
    question: str = Field(description='The question to answer.')
    ideal: str = Field(description='The ideal answer to the question.')
    distractors: list[str] = Field(
        description='The distractor answers to the question.',
    )
    sources: str = Field(description='The sources for the question.')

    def get_multiple_choice(self) -> str:
        """Build a multiple choice question from the LitQA entry."""
        options = [self.ideal, *self.distractors]
        random.shuffle(options)
        return f"{self.question}? Choose one of these options: {','.join(options)}"  # noqa E501


class LitQATaskConfig(BaseConfig):
    """Configuration for the LitQA task."""

    name: Literal['litqa'] = 'litqa'  # type: ignore[assignment]
    download_url: str = Field(
        default='https://raw.githubusercontent.com/Future-House/LitQA/main/litqa-v0.jsonl',
        description='The URL to download the data from.',
    )
    download_dir: Path = Field(
        description='The directory to download the data to.',
    )


class LitQATask:
    """LitQA evaluation task."""

    def __init__(self, config: LitQATaskConfig) -> None:
        """Initialize the LitQATask."""
        self.config = config
        self.data_file = self.config.download_dir / 'litqa-v0.jsonl'

    def download(self) -> None:
        """Download the LitQA dataset."""
        # read in the jsonl, get the questions.
        if not self.data_file.exists():
            command = f'curl -o {self.data_file} {self.config.download_url}'
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
        """Evaluate the model on the LitQA task.

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
        preds = generator.generate(questions)

        # Compute the accuracy and precision
        accuracy = self._compute_accuracy(ground_truths, preds)
        precision = self._compute_precision(ground_truths, preds)

        return {'accuracy': accuracy, 'precision': precision}


if __name__ == '__main__':
    import json
    import subprocess
    from argparse import ArgumentParser
    from pathlib import Path

    from distllm.embed import get_encoder
    from distllm.embed import get_pooler
    from distllm.generate import get_generator
    from distllm.rag.response_synthesizer import RagGenerator
    from distllm.rag.search import FaissIndex
    from distllm.rag.search import Retriever

    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', '-ds', type=Path, required=True)
    parser.add_argument('--download_dir', '-dw', type=Path, required=True)
    parser.add_argument('--encoder_path', '-ep', type=Path, required=True)

    args = parser.parse_args()

    # Initialize the modules
    encoder = get_encoder(
        {'name': 'auto', 'pretrained_model_name_or_path': args.encoder_path},
    )

    pooler = get_pooler({})
    generator = get_generator({})
    faiss_index = FaissIndex(args.dataset_dir)

    retriever = Retriever(
        encoder=encoder,
        pooler=pooler,
        faiss_index=faiss_index,
    )
    generator = RagGenerator(retriever=retriever, generator=generator)

    # Initialize the LitQA task
    config = LitQATaskConfig(download_dir=args.download_dir)
    task = LitQATask(config)

    # Evaluate the model on the LitQA task
    results = task.evaluate(generator)
