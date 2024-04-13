"""Question answer prompt template for generating answers to questions."""

from __future__ import annotations

from typing import Literal

from distllm.utils import BaseConfig


class QuestionAnswerPromptTemplateConfig(BaseConfig):
    """Configuration for the QuestionAnswerPromptTemplate."""

    name: Literal['question_answer'] = 'question_answer'  # type: ignore[assignment]


# class QuestionAnswerPromptTemplate:
#     """Question answer prompt template."""

#     template_with_context: str = (
#         'Answer the question below with the context.\n\n'
#         'Context (with relevance scores):\n\n{context}\n\n----\n\n'
#         'Question: {question}\n\n'
#         'Choose one of the given options and only output that. '
#         'If the context provides insufficient information and '
#         'the question cannot be directly answered, reply '
#         '"I cannot answer. \n\n Answer: " '
#     )

#     template_no_context: str = (
#         'Answer the question below.\n\n'
#         'Question: {question}\n\n'
#         'Choose one of the given options and only output that. If the question' # noqa
#         ' cannot be directly answered, reply "I cannot answer. \n\n Answer" '
#     )

#     def __init__(self, config: QuestionAnswerPromptTemplateConfig) -> None:
#         """Initialize the QuestionAnswerPromptTemplate."""
#         self.config = config

#     def _format_prompt(
#         self,
#         question: str,
#         context: list[str],
#         score: list[float],
#     ) -> str:
#         """Format the prompt with the question and context."""
#         context_concat = '\n'.join(
#             f'Context: {c}, score: {s}' for c, s in zip(context, score)
#         )
#         return self.template_with_context.format(
#             context=context_concat,
#             question=question,
#         )

#     def preprocess(
#         self,
#         text: str | list[str],  # question
#         contexts: list[list[str]] | None = None,
#         scores: list[list[float]] | None = None,
#     ) -> list[str]:
#         """Preprocess the text into prompts.

#         Parameters
#         ----------
#         text : str
#             The text to format.
#         contexts : list[list[str]], optional
#             The contexts to include for each text, by default None.
#         scores : list[list[float]], optional
#             The scores for each context, by default None.


#         Returns
#         -------
#         list[str]
#             The formatted prompts.
#         """
#         # Ensure text is a list
#         if isinstance(text, str):
#             text = [text]

#         # If no contexts are provided, use the no-context template
#         if contexts is None or scores is None:
#             return [self.template_no_context.format(question=q) for q in text] # noqa

#         # Build the prompts using the template
#         return list(map(self._format_prompt, text, contexts, scores))

#     def postprocess(self, responses: list[str]) -> list[str]:
#         """Postprocess the responses.

#         Parameters
#         ----------
#         responses : list[str]
#             The responses to postprocess.

#         Returns
#         -------
#         list[str]
#             The postprocessed responses.
#         """
#         return responses


class QuestionAnswerPromptTemplate:
    """Question answer prompt template."""

    template_with_context: str = (
        'Answer the question below with the context.\n\n'
        'Context (with relevance scores):\n\n{context}\n\n----\n\n'
        'Question: {question}\n\n'
        'Choose one of the given options and only output that. '
        'If the context provides insufficient information and '
        'the question cannot be directly answered, reply '
        '"I cannot answer. \n\n Answer: " '
    )

    template_no_context: str = (
        'Question:'
        '{question}'
        '[INST] Answer this question by choosing one of the options. [/INST]'
        'Answer: '
    )

    def __init__(self, config: QuestionAnswerPromptTemplateConfig) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def _format_prompt(
        self,
        question: str,
        context: list[str],
        score: list[float],
    ) -> str:
        """Format the prompt with the question and context."""
        context_concat = '\n'.join(
            f'Context: {c}, score: {s}' for c, s in zip(context, score)
        )
        return self.template_with_context.format(
            context=context_concat,
            question=question,
        )

    def preprocess(
        self,
        text: str | list[str],  # question
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
