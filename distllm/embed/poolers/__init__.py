"""Module for poolers."""

from __future__ import annotations

from typing import Any
from typing import Union

from distllm.embed.poolers.base import Pooler
from distllm.embed.poolers.last_token import LastTokenPooler
from distllm.embed.poolers.last_token import LastTokenPoolerConfig
from distllm.embed.poolers.mean import MeanPooler
from distllm.embed.poolers.mean import MeanPoolerConfig
from distllm.utils import BaseConfig

PoolerConfigs = Union[MeanPoolerConfig, LastTokenPoolerConfig]

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Pooler]]] = {
    'mean': (MeanPoolerConfig, MeanPooler),
    'last_token': (LastTokenPoolerConfig, LastTokenPooler),
}


def get_pooler(kwargs: dict[str, Any]) -> Pooler:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - mean
    - last_token

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    Pooler
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
            f'Unknown pooler name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))
