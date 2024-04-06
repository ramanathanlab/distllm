"""PromptTemplate module."""

from __future__ import annotations

from typing import Any

from distllm.generate.prompts.base import PromptTemplate
from distllm.generate.prompts.identity import IdentityPromptTemplate
from distllm.generate.prompts.identity import IdentityPromptTemplateConfig
from distllm.generate.prompts.question_chunk import QuestionChunkPromptTemplate
from distllm.generate.prompts.question_chunk import (
    QuestionChunkPromptTemplateConfig,
)
from distllm.utils import BaseConfig

PromptTemplateConfigs = QuestionChunkPromptTemplateConfig

STRATEGIES: dict[str, tuple[type[BaseConfig], type[PromptTemplate]]] = {
    'question_chunk': (
        QuestionChunkPromptTemplateConfig,
        QuestionChunkPromptTemplate,
    ),
    'identity': (IdentityPromptTemplateConfig, IdentityPromptTemplate),
}


def get_prompt_template(kwargs: dict[str, Any]) -> PromptTemplate:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - question_chunk
    - identity

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
