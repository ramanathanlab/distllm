"""Encoder module."""

from __future__ import annotations

from typing import Any
from typing import Union

from distllm.embed.encoders.auto import AutoEncoder
from distllm.embed.encoders.auto import AutoEncoderConfig
from distllm.embed.encoders.base import Encoder
from distllm.embed.encoders.esm2 import Esm2Encoder
from distllm.embed.encoders.esm2 import Esm2EncoderConfig
from distllm.embed.encoders.esmc import EsmCambrianEncoder
from distllm.embed.encoders.esmc import EsmCambrianEncoderConfig
from distllm.registry import registry
from distllm.utils import BaseConfig

EncoderConfigs = Union[
    Esm2EncoderConfig,
    EsmCambrianEncoderConfig,
    AutoEncoderConfig,
]

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Encoder]]] = {
    'esm2': (Esm2EncoderConfig, Esm2Encoder),
    'esmc': (EsmCambrianEncoderConfig, EsmCambrianEncoder),
    'auto': (AutoEncoderConfig, AutoEncoder),
}


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> Encoder:
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)  # type: ignore[arg-type]
    if not strategy:
        raise ValueError(
            f'Unknown encoder name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))


def get_encoder(
    kwargs: dict[str, Any],
    register: bool = False,
) -> Encoder:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - esm2
    - esmc
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
    Encoder
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
