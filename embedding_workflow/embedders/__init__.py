"""Embedder module for embedding sequences."""

from __future__ import annotations

from typing import Any

from embedding_workflow.embedders.auto import AutoEmbedder
from embedding_workflow.embedders.auto import AutoEmbedderConfig
from embedding_workflow.embedders.base import BaseEmbedder
from embedding_workflow.embedders.base import BaseEmbedderConfig
from embedding_workflow.embedders.esm2 import Esm2Embedder
from embedding_workflow.embedders.esm2 import Esm2EmbedderConfig
from embedding_workflow.registry import registry

EmbedderConfigTypes = Esm2EmbedderConfig | AutoEmbedderConfig
EmbedderTypes = Esm2Embedder | AutoEmbedder

_EmbedderTypes = tuple[type[EmbedderConfigTypes], type[EmbedderTypes]]

STRATEGIES: dict[str, _EmbedderTypes] = {
    'esm2': (Esm2EmbedderConfig, Esm2Embedder),
    'auto': (AutoEmbedderConfig, AutoEmbedder),
}


def get_embedder(
    kwargs: dict[str, Any],
    register: bool = False,
) -> EmbedderTypes:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - esm2
    - auto

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.
    register : bool, optional
        Register the embedder instance for warmstart. Caches the
        embedder instance based on the kwargs, by default False.

    Returns
    -------
    EmbedderTypes
        The embedder instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)
    if not strategy:
        raise ValueError(f'Unknown embedder name: {name}')

    # Get the config and classes
    config_cls, cls = strategy

    # Make a function to combine the config and instance initialization
    # since the registry only accepts functions with hashable arguments.
    def factory_fn(**kwargs: dict[str, Any]) -> EmbedderTypes:
        # Create the config
        config = config_cls(**kwargs)
        # Create the instance
        return cls(config)

    # Register and create the embedder instance
    if register:
        registry.register(factory_fn)
        embedder = registry.get(factory_fn, **kwargs)
    else:
        embedder = factory_fn(**kwargs)

    return embedder
