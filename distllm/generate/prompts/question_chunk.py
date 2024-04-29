"""Question chunk prompt for generating questions from a chunk of text."""

from __future__ import annotations

from typing import Literal

from nltk.tokenize import sent_tokenize

from distllm.utils import BaseConfig


class QuestionChunkPromptTemplateConfig(BaseConfig):
    """Configuration for the QuestionChunkPromptTemplate."""

    name: Literal['question_chunk'] = 'question_chunk'  # type: ignore[assignment]


class QuestionChunkPromptTemplate:
    """Question chunk prompt."""

    template = """
    You are a scientific researcher. Given the following chunk of text,
     generate a high-quality question that requires deep understanding of
     the concepts presented in the text. Do not include questions that refer
     to specific aspects of the paper, like results, findings, or references.
    \n\n

    Text: {chunk}\nQuestion:
    """

    def __init__(self, config: QuestionChunkPromptTemplateConfig) -> None:
        """Initialize the QuestionChunkPromptTemplate."""
        self.config = config

    def preprocess(
        self,
        text: str | list[str],
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
        if isinstance(text, str):
            text = [text]

        prompts = [self.template.format(chunk=chunk) for chunk in text]
        return prompts

    def _parse_response(self, response: str) -> str:
        """Parse the response to extract a question."""
        # Tokenize the response into sentences
        sentences = sent_tokenize(response)

        # Get sentences ending with '?'
        questions = [
            sentence
            for sentence in sentences
            if sentence.strip().endswith('?')
        ]

        # Only keep the first question
        return '' if not questions else questions[0]

    def postprocess(self, responses: list[str]) -> list[str]:
        """Extract the questions from the results.

        Parameters
        ----------
        responses : list[str]
            The responses to extract questions from.

        Returns
        -------
        list[str]
            The questions extracted from the results.
        """
        questions = [self._parse_response(response) for response in responses]
        return questions
