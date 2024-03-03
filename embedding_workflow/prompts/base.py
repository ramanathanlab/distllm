"""Interface for all prompts to follow."""

from __future__ import annotations

from typing import Protocol

from embedding_workflow.generators import LLMResponse
from embedding_workflow.utils import BaseConfig


class Prompt(Protocol):
    """Prompt protocol for all prompts to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the prompt with the configuration."""
        ...

    def preprocess(self, text: str | list[str]) -> list[str]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to preprocess.

        Returns
        -------
        list[str]
            The preprocessed prompts.
        """
        ...

    def postprocess(self, responses: list[LLMResponse]) -> list[str]:
        """Postprocess the responses.

        Parameters
        ----------
        responses : list[LLMResponse]
            The responses to postprocess.

        Returns
        -------
        list[str]
            The postprocessed responses.
        """
        ...
