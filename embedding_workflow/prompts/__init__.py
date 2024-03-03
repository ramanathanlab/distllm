"""Prompt module."""

from __future__ import annotations

from typing import Any

from embedding_workflow.prompts.base import Prompt
from embedding_workflow.prompts.question import QuestionPrompt
from embedding_workflow.prompts.question import QuestionPromptConfig
from embedding_workflow.utils import BaseConfig

PromptConfigs = QuestionPromptConfig

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Prompt]]] = {
    'question': (QuestionPromptConfig, QuestionPrompt),
}


def get_prompt(kwargs: dict[str, Any]) -> Prompt:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - question

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    Prompt
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
