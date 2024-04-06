"""Question answer prompt template for generating answers to questions."""

from __future__ import annotations

from typing import Literal

from distllm.utils import BaseConfig


class QuestionAnswerPromptTemplateConfig(BaseConfig):
    """Configuration for the QuestionAnswerPromptTemplate."""

    name: Literal['question_answer'] = 'question_answer'  # type: ignore[assignment]


class QuestionAnswerPromptTemplate:
    """Question answer prompt."""

    template = """
    You are a scientific researcher. Given the question and the relevant
    context, generate a high-quality answer that requires deep understanding
    of the concepts presented in the text. If a context isn't provided, rely
    on your own knowledge. Be concise and truthful in your
    response.

    Context:\n{context}\n\nQuestion: {question}\nAnswer:
    """

    def __init__(self, config: QuestionAnswerPromptTemplateConfig) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def preprocess(
        self,
        text: str | list[str],
        contexts: list[list[str]] | None = None,
    ) -> list[str]:
        """Preprocess the text into prompts.

        Parameters
        ----------
        text : str
            The text to format.
        contexts : list[list[str]], optional
            The contexts to include for each text, by default None.

        Returns
        -------
        list[str]
            The formatted prompts.
        """
        if isinstance(text, str):
            text = [text]

        # If contexts are not provided, use an empty context
        if contexts is None:
            contexts = [['']] * len(text)

        # Format the prompts
        prompts = [
            self.template.format(context='\n'.join(context), question=question)
            for question, context in zip(text, contexts)
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
            The answers to the questions.
        """
        return responses
