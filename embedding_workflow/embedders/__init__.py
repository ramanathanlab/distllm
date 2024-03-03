"""Embedder module."""

from __future__ import annotations

from typing import Any

from embedding_workflow.embedders.auto import AutoEmbedder
from embedding_workflow.embedders.auto import AutoEmbedderConfig
from embedding_workflow.embedders.base import Embedder
from embedding_workflow.embedders.esm2 import Esm2Embedder
from embedding_workflow.embedders.esm2 import Esm2EmbedderConfig
from embedding_workflow.registry import registry
from embedding_workflow.utils import BaseConfig

EmbedderConfigs = Esm2EmbedderConfig | AutoEmbedderConfig

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Embedder]]] = {
    'esm2': (Esm2EmbedderConfig, Esm2Embedder),
    'auto': (AutoEmbedderConfig, AutoEmbedder),
}


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> Embedder:
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(
            f'Unknown embedder name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))


def get_embedder(
    kwargs: dict[str, Any],
    register: bool = False,
) -> Embedder:
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
        Register the instance for warmstart. Caches the
        instance based on the kwargs, by default False.

    Returns
    -------
    Embedder
        The instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    # Create and register the instance
    if register:
        registry.register(_factory_fn)
        return registry.get(_factory_fn, **kwargs)

    return _factory_fn(**kwargs)
