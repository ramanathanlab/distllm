"""Generator module."""

from __future__ import annotations

from typing import Any

from embedding_workflow.generators.base import LLMGenerator
from embedding_workflow.generators.base import LLMResponse
from embedding_workflow.generators.vllm_backend import VLLMGenerator
from embedding_workflow.generators.vllm_backend import VLLMGeneratorConfig
from embedding_workflow.registry import registry
from embedding_workflow.utils import BaseConfig

LLMGeneratorConfigs = VLLMGeneratorConfig

STRATEGIES: dict[str, tuple[type[BaseConfig], type[LLMGenerator]]] = {
    'vllm': (VLLMGeneratorConfig, VLLMGenerator),
}


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> LLMGenerator:
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(
            f'Unknown generator name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))


def get_generator(
    kwargs: dict[str, Any],
    register: bool = False,
) -> LLMGenerator:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - vllm

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
    LLMGenerator
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
