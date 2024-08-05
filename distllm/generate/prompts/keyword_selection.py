"""Keyword selection prompt template for selecting relevant keywords."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Union

from distllm.utils import BaseConfig


class KeywordSelectionPromptTemplateConfig(BaseConfig):
    """Configuration for the KeywordSelectionPromptTemplate."""

    name: Literal['keyword_selection'] = 'keyword_selection'  # type: ignore[assignment]

    keywords: Union[Path, list[str]]  # noqa: UP007
    """List of keywords to select from.
      If path, requires newline separated keywords."""


class KeywordSelectionPromptTemplate:
    """Keyword selection prompt template for selecting relevant keywords."""

    template: str = (
        'You are a perfect scientist in the domain of radiation-based medicine'
        ' and biology.\n'
        'You are also highly capable and well-read in all adjacent scientific '
        'domains.\n'
        'Given a list of keywords of the domain and a paragraph,\n'
        'You are tasked with selecting the 3 keywords that are most relevant '
        'for the given paragraph.\n'
        'Order the 3 keywords by relevance in ascending order.\n'
        'The document:\n\n{document}\n\n----\n\n'
        'List of keywords: {keywords_list}\n\n'
        'Write an answer based on the context.\n'
        'If all keywords in the list are equally irrelevant, return the str '
        '`None of the above` 3 times.\n'
        'Answer: '
    )

    def __init__(self, config: KeywordSelectionPromptTemplateConfig) -> None:
        """Initialize the KeywordSelectionPromptTemplate."""
        self.config = config
        if isinstance(self.config.keywords, Path):
            self.keywords_list = self.config.keywords.read_text().splitlines()
        else:
            self.keywords_list = self.config.keywords

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
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]

        return [
            self.template.format(
                keywords_list=self.keywords_list,
                document=paper,
            )
            for paper in text
        ]

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
