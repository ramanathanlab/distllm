"""Module for readers."""

from __future__ import annotations

from typing import Any

from distllm.generate.readers.base import Reader
from distllm.generate.readers.huggingface import HuggingFaceReader
from distllm.generate.readers.huggingface import HuggingFaceReaderConfig
from distllm.utils import BaseConfig

ReaderConfigs = HuggingFaceReaderConfig

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Reader]]] = {
    'huggingface': (HuggingFaceReaderConfig, HuggingFaceReader),
}


def get_reader(kwargs: dict[str, Any]) -> Reader:
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
    Reader
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
            f'Unknown reader name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))
