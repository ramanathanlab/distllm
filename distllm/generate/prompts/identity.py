"""Identity prompt to skip any processing."""

from __future__ import annotations

from typing import Literal

from distllm.utils import BaseConfig


class IdentityPromptTemplateConfig(BaseConfig):
    """Configuration for the IdentityPromptTemplate."""

    name: Literal['identity'] = 'identity'  # type: ignore[assignment]


class IdentityPromptTemplate:
    """Identity prompt."""

    def __init__(self, config: IdentityPromptTemplateConfig) -> None:
        """Initialize the IdentityPromptTemplate."""
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

        return text

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
