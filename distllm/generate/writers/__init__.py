"""Module for writer."""

from __future__ import annotations

from typing import Any
from typing import Union

from distllm.generate.writers.amp_json import AMPJSONLWriter
from distllm.generate.writers.amp_json import AMPJSONLWriterConfig
from distllm.generate.writers.base import Writer
from distllm.generate.writers.huggingface import HuggingFaceWriter
from distllm.generate.writers.huggingface import HuggingFaceWriterConfig
from distllm.utils import BaseConfig

WriterConfigs = Union[HuggingFaceWriterConfig, AMPJSONLWriterConfig]

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Writer]]] = {
    'huggingface': (HuggingFaceWriterConfig, HuggingFaceWriter),
    'amp_jsonl': (AMPJSONLWriterConfig, AMPJSONLWriter),
}


def get_writer(kwargs: dict[str, Any]) -> Writer:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - huggingface

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    Writer
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
            f'Unknown writer name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))
