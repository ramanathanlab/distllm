"""Module for handling prompts for a language model."""

from __future__ import annotations

from typing import Literal

from distllm.generate.generators import LLMResult
from distllm.utils import BaseConfig


class QuestionPromptConfig(BaseConfig):
    """Configuration for the QuestionPrompt."""

    name: Literal['question_prompt'] = 'question_prompt'  # type: ignore[assignment]


class QuestionPrompt:
    """Question generator using a language model."""

    prompt = """
    You are an advanced AGI for science.
    Here is a paper: {document}.
    Please provide several expert-level questions about this paper.
    """

    def __init__(self, config: QuestionPromptConfig) -> None:
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

        prompts = [self.prompt.format(document=document) for document in text]
        return prompts

    def postprocess(self, responses: list[LLMResult]) -> list[str]:
        """Extract the questions from the responses.

        Parameters
        ----------
        responses : list[LLMResult]
            The responses to extract questions from.

        Returns
        -------
        list[str]
            The questions extracted from the responses.
        """
        questions = [response.response for response in responses]
        return questions
