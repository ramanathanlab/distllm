"""Module for handling prompts for a language model."""

from __future__ import annotations

from typing import Literal

from nltk.tokenize import sent_tokenize

from distllm.utils import BaseConfig


class QuestionChunkPromptConfig(BaseConfig):
    """Configuration for the QuestionChunkPrompt."""

    name: Literal['question_chunk'] = 'question_chunk'  # type: ignore[assignment]


class QuestionChunkPrompt:
    """Question generator using a language model."""

    prompt = """
    You are a scientific researcher. Given the following chunk of text,
     generate a high-quality question that requires deep understanding of
     the concepts presented in the text. Do not include questions that refer
     to specific aspects of the paper, like results, findings, or references.
    \n\n

    Text: {chunk}\nQuestion: '
    """

    def __init__(self, config: QuestionChunkPromptConfig) -> None:
        """Initialize the LLMGenerator."""
        self.config = config

    def preprocess(self, text: str | list[str]) -> list[str]:
        """Preprocess the text documents into prompts.

        Parameters
        ----------
        text : str
            The text to format.

        Returns
        -------
        list[str]
        """
        if isinstance(text, str):
            text = [text]

        prompts = [self.prompt.format(chunk=chunk) for chunk in text]
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
        # TODO: We could use a process pool here.
        questions = [self._parse_response(response) for response in responses]
        return questions
