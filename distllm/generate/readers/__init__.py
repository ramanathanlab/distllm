"""Module for readers."""

from __future__ import annotations

from typing import Any
from typing import Union

from distllm.generate.readers.amp_json import AMPJsonReader
from distllm.generate.readers.amp_json import AMPJsonReaderConfig
from distllm.generate.readers.base import Reader
from distllm.generate.readers.huggingface import HuggingFaceReader
from distllm.generate.readers.huggingface import HuggingFaceReaderConfig
from distllm.generate.readers.jsonl import JsonlReader
from distllm.generate.readers.jsonl import JsonlReaderConfig
from distllm.utils import BaseConfig

ReaderConfigs = Union[
    HuggingFaceReaderConfig,
    JsonlReaderConfig,
    AMPJsonReaderConfig,
]

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Reader]]] = {
    'huggingface': (HuggingFaceReaderConfig, HuggingFaceReader),
    'jsonl': (JsonlReaderConfig, JsonlReader),
    'amp_json': (AMPJsonReaderConfig, AMPJsonReader),
}


def get_reader(kwargs: dict[str, Any]) -> Reader:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - huggingface
    - jsonl

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
