"""PromptTemplate module."""

from __future__ import annotations

from typing import Any
from typing import Union

from distllm.generate.prompts.amp_question import AMPQuestionPromptConfig
from distllm.generate.prompts.amp_question import AMPQuestionPromptTemplate
from distllm.generate.prompts.base import PromptTemplate
from distllm.generate.prompts.identity import IdentityPromptTemplate
from distllm.generate.prompts.identity import IdentityPromptTemplateConfig
from distllm.generate.prompts.keyword_selection import (
    KeywordSelectionPromptTemplate,
)
from distllm.generate.prompts.keyword_selection import (
    KeywordSelectionPromptTemplateConfig,
)
from distllm.generate.prompts.question_answer import (
    QuestionAnswerPromptTemplate,
)
from distllm.generate.prompts.question_answer import (
    QuestionAnswerPromptTemplateConfig,
)
from distllm.generate.prompts.question_chunk import QuestionChunkPromptTemplate
from distllm.generate.prompts.question_chunk import (
    QuestionChunkPromptTemplateConfig,
)
from distllm.utils import BaseConfig

PromptTemplateConfigs = Union[
    IdentityPromptTemplateConfig,
    QuestionChunkPromptTemplateConfig,
    QuestionAnswerPromptTemplateConfig,
    KeywordSelectionPromptTemplateConfig,
    AMPQuestionPromptConfig,
]

STRATEGIES: dict[str, tuple[type[BaseConfig], type[PromptTemplate]]] = {
    'identity': (IdentityPromptTemplateConfig, IdentityPromptTemplate),
    'question_chunk': (
        QuestionChunkPromptTemplateConfig,
        QuestionChunkPromptTemplate,
    ),
    'question_answer': (
        QuestionAnswerPromptTemplateConfig,
        QuestionAnswerPromptTemplate,
    ),
    'keyword_selection': (
        KeywordSelectionPromptTemplateConfig,
        KeywordSelectionPromptTemplate,
    ),
    'amp_question': (AMPQuestionPromptConfig, AMPQuestionPromptTemplate),
}


def get_prompt_template(kwargs: dict[str, Any]) -> PromptTemplate:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - identity
    - question_chunk
    - question_answer
    - keyword_selection

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    PromptTemplate
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)
    if not strategy:
        raise ValueError(
            f'Unknown prompt name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))
