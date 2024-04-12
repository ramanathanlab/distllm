"""Question answer prompt template for generating answers to questions."""

from __future__ import annotations

from typing import Literal

from distllm.utils import BaseConfig


class QuestionAnswerPromptTemplateConfig(BaseConfig):
    """Configuration for the QuestionAnswerPromptTemplate."""

    name: Literal['question_answer'] = 'question_answer'  # type: ignore[assignment]


class QuestionAnswerPromptTemplate:
    """Question answer prompt template."""

    keywords_list : list[str] = [
    "Radiosensitivity",
    "Radioprotection",
    "Radiobiology",
    "DNA damage",
    "Low-dose radiation",
    "Radiation therapy",
    "Ionizing radiation",
    "Non-ionizing radiation",
    "Radiation exposure",
    "Radiation effects",
    "Cellular response",
    "Oncology",
    "Carcinogenesis",
    "Mutation",
    "Cancer therapy",
    "Radiation oncology",
    "Dosimetry",
    "Health physics",
    "Radiation protection",
    "Radiation poisoning",
    "Radiation sickness",
    "Chronic exposure",
    "Acute exposure",
    "Stochastic effects",
    "Deterministic effects",
    "Linear no-threshold model",
    "Threshold dose",
    "Epidemiology",
    "Risk assessment",
    "Bioinformatics",
    "Genomics",
    "Proteomics",
    "Cell cycle",
    "Apoptosis",
    "Repair mechanisms",
    "Oxidative stress",
    "Free radicals",
    "Immune response",
    "Inflammation",
    "Radiation dermatitis",]

    template_with_context: str = (
        'You are a perfect scientist in the domain of radiation-based medicine and biology.\n'
        'You are also highly capable and well-read in all adjacent scientific domains.\n'
        'Given a list of keywords of the domain and a paragraph,\n'
        'You are tasked with selecting the 3 keywords that are most relevant for the given paragraph.\n'
        'Order the 3 keywords by relevance in ascending order.\n'
        'The paragraph:\n\n{paragraph}\n\n----\n\n'
        'List of keywords: {keywords_list}\n\n'
        'Write an answer based on the context.\n'
        'If all keywords in the list are equally irrelevant, return the str `None of the above` 3 times.\n'
        'Answer: '
    )

    def __init__(self, config: QuestionAnswerPromptTemplateConfig) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def _format_prompt(
        self,
        keywords_list: list[str],
        paragraph: str,
    ) -> str:
        """Format the prompt with the question and context."""
        return self.template_with_context.format(
            keywords_list=keywords_list,
            paragraph=paragraph,
        )

    def preprocess(
        self,
        text: str | list[str],  # question, we don;t have 
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

        # If no contexts are provided, use the no-context template
        if contexts is None or scores is None:
            return [self.template_no_context.format(question=q) for q in text]

        # Build the prompts using the template
        return list(map(self._format_prompt, text, contexts, scores))

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
