"""Interface for all prompts to follow."""

from __future__ import annotations

from typing import Protocol

from distllm.utils import BaseConfig

# TODO: Refactor strategy protocols so that the configuration is not required
# for the __init__ method. To do this, add a from_config method to the config
# objects that returns the instance of the class. Call the arbitrary init
# method with the parameters of the dataclass config. This removes the need
# to have the factory functions since pydantic will already handle the specific
# type lookups.


class PromptTemplate(Protocol):
    """PromptTemplate protocol for all prompts to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the prompt with the configuration."""
        ...

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
            The text to preprocess.
        contexts : list[list[str]], optional
            The contexts to include for each text, by default None.
        scores : list[list[float]], optional
            The scores for each context, by default None.

        Returns
        -------
        list[str]
            The preprocessed prompts.
        """
        ...

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
        ...
