"""Question answer prompt template for generating answers to questions."""

from __future__ import annotations

import json
import re
from typing import Any
from typing import Literal

from distllm.utils import BaseConfig


class AMPQuestionPromptConfig(BaseConfig):
    """Configuration for the QuestionAnswerPromptTemplate."""

    name: Literal['amp_question'] = 'amp_question'  # type: ignore[assignment]


class AMPQuestionPromptTemplate:
    """Question answer prompt template for generating answers to questions."""

    template = (
        'Generate a biologically accurate multiple-choice question '
        'to which there is only one answer by explicitly using the '
        "protein name '{protein_name}' based on its function as "
        "described here: '{function_description}'. Format the output "
        "with the question followed by 'Question:', four short answer "
        'options labeled (A, B, C, D), and finally specify the correct '
        "answer following 'Answer:'. Ensure the answers are concise "
        'and correct.'
    )

    def __init__(self, config: AMPQuestionPromptConfig) -> None:
        """Initialize the QuestionAnswerPromptTemplate."""
        self.config = config

    def _format_input(self, text: str) -> str:
        """Format the input text into the template appropriately."""
        data = json.loads(text)
        protein_name = data['Protein_Name']
        function_description = data['Function']
        return self.template.format(
            protein_name=protein_name,
            function_description=function_description,
        )

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
        return list(map(self._format_input, text))

    def _postprocess_response(
        self,
        response: str,
    ) -> str:
        """Postprocess the response.

        Parameters
        ----------
        response : str
            The model output to postprocess.

        Returns
        -------
        str
            The string representation of a json object of the response.
        """
        # TODO: More robust regex to split the text for format issues.
        parts = re.split(r'\n\s*Question:', response, flags=re.IGNORECASE)
        output: dict[str, Any] = {
            'full_question_text': None,
            'correct_answer': None,
            'distractors': [],
        }
        if len(parts) > 1:
            # Extract everything after "Question:"
            question_and_options = parts[1].strip()

            # Detect the correct answer label from the text
            correct_answer_match = re.search(
                r'Answer:\s*([A-D])\)',
                question_and_options,
            )
            correct_answer_label = (
                correct_answer_match.group(1) if correct_answer_match else None
            )

            # More robust detection for where options start.
            options_start = re.search(r'\s*\bA\)', question_and_options)
            if options_start:
                question_text = question_and_options[
                    : options_start.start()
                ].strip()
                options_text = question_and_options[
                    options_start.start() :
                ].strip()

                # Remove the explicit answer part from the options text
                options_text_clean = re.sub(
                    r'\s*Answer:\s*[A-D]\).*',
                    '',
                    options_text,
                    flags=re.IGNORECASE,
                ).strip()

                # Construct the full question including the options
                full_question_text = f'{question_text} {options_text_clean}'

                # Extract the correct answer and distractors from the options
                correct_answer = None
                distractors = []
                options_list = re.split(r'\s+(?=[A-D]\))', options_text_clean)
                for option in options_list:
                    option_label = option[
                        :2
                    ]  # This is "A)", "B)", "C)", or "D)"
                    option_text = option[3:].strip()

                    if option_label == f'{correct_answer_label})':
                        correct_answer = option_text
                    else:
                        distractors.append(option_text)

                output['full_question_text'] = full_question_text
                output['correct_answer'] = correct_answer
                output['distractors'] = distractors
                return json.dumps(output)

        # Could not split the response correctly
        return json.dumps(output)

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
        return list(map(self._postprocess_response, responses))
