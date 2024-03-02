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

EMBEDDER_STRATEGIES: dict[str, _EmbedderTypes] = {
    'esm2': (Esm2EmbedderConfig, Esm2Embedder),
    'auto': (AutoEmbedderConfig, AutoEmbedder),
}


def get_embedder(
    embedder_kwargs: dict[str, Any],
    register: bool = False,
) -> EmbedderTypes:
    """Get the embedder instance based on the embedder name and kwargs.

    Caches the embedder instance based on the embedder name and kwargs.
    Currently supports the following embedders: esm2, auto.

    Parameters
    ----------
    embedder_kwargs : dict[str, Any]
        The embedder configuration. Contains an extra `name` argument
        to specify the embedder to use.
    register : bool, optional
        Register the embedder instance for warmstart, by default False.

    Returns
    -------
    EmbedderTypes
        The embedder instance.

    Raises
    ------
    ValueError
        If the embedder name is unknown.
    """
    name = embedder_kwargs.get('name', '')
    embedder_strategy = EMBEDDER_STRATEGIES.get(name)
    if not embedder_strategy:
        raise ValueError(f'Unknown embedder name: {name}')

    # Unpack the embedder strategy
    config_cls, embedder_cls = embedder_strategy

    # Make a function to combine the config and embedder initialization
    # since the registry only accepts functions with hashable arguments.
    def embedder_factory(**embedder_kwargs: dict[str, Any]) -> EmbedderTypes:
        # Create the embedder config
        config = config_cls(**embedder_kwargs)
        # Create the embedder instance
        return embedder_cls(config)

    # Register and create the embedder instance
    if register:
        registry.register(embedder_factory)
        embedder = registry.get(embedder_factory, **embedder_kwargs)
    else:
        embedder = embedder_factory(**embedder_kwargs)

    return embedder
