"""Keyword selection prompt template for selecting relevant keywords."""

from __future__ import annotations

from typing import Literal

from distllm.utils import BaseConfig


class KeywordSelectionPromptTemplateConfig(BaseConfig):
    """Configuration for the KeywordSelectionPromptTemplate."""

    name: Literal['keyword_selection'] = 'keyword_selection'  # type: ignore[assignment]


class KeywordSelectionPromptTemplate:
    """Keyword selection prompt template for selecting relevant keywords."""

    # TODO: Read the keywords from a file that is specified in the config
    keywords_list: list[str] = [  # noqa: RUF012
        'Radiosensitivity',
        'Radioprotection',
        'Radiobiology',
        'DNA damage',
        'Low-dose radiation',
        'Radiation therapy',
        'Ionizing radiation',
        'Non-ionizing radiation',
        'Radiation exposure',
        'Radiation effects',
        'Cellular response',
        'Oncology',
        'Carcinogenesis',
        'Mutation',
        'Cancer therapy',
        'Radiation oncology',
        'Dosimetry',
        'Health physics',
        'Radiation protection',
        'Radiation poisoning',
        'Radiation sickness',
        'Chronic exposure',
        'Acute exposure',
        'Stochastic effects',
        'Deterministic effects',
        'Linear no-threshold model',
        'Threshold dose',
        'Epidemiology',
        'Risk assessment',
        'Bioinformatics',
        'Genomics',
        'Proteomics',
        'Cell cycle',
        'Apoptosis',
        'Repair mechanisms',
        'Oxidative stress',
        'Free radicals',
        'Immune response',
        'Inflammation',
        'Radiation dermatitis',
    ]

    # TODO: Update prompt to sample keywords for full documents
    template: str = (
        'You are a perfect scientist in the domain of radiation-based medicine and biology.\n'  # noqa: E501
        'You are also highly capable and well-read in all adjacent scientific domains.\n'  # noqa: E501
        'Given a list of keywords of the domain and a paragraph,\n'
        'You are tasked with selecting the 3 keywords that are most relevant for the given paragraph.\n'  # noqa: E501
        'Order the 3 keywords by relevance in ascending order.\n'
        'The paragraph:\n\n{paragraph}\n\n----\n\n'
        'List of keywords: {keywords_list}\n\n'
        'Write an answer based on the context.\n'
        'If all keywords in the list are equally irrelevant, return the str `None of the above` 3 times.\n'  # noqa: E501
        'Answer: '
    )

    def __init__(self, config: KeywordSelectionPromptTemplateConfig) -> None:
        """Initialize the KeywordSelectionPromptTemplate."""
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
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]

        return [
            self.template.format(
                keywords_list=self.keywords_list,
                paragraph=paper,
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
