"""Module for data datasets."""

from __future__ import annotations

from typing import Any

from distllm.datasets.base import Dataset
from distllm.datasets.fasta import FastaDataset
from distllm.datasets.fasta import FastaDatasetConfig
from distllm.datasets.jsonl import JsonlDataset
from distllm.datasets.jsonl import JsonlDatasetConfig
from distllm.datasets.single_line import SequencePerLineDataset
from distllm.datasets.single_line import SequencePerLineDatasetConfig
from distllm.utils import BaseConfig

DatasetConfigs = (
    FastaDatasetConfig | SequencePerLineDatasetConfig | JsonlDatasetConfig
)

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Dataset]]] = {
    'fasta': (FastaDatasetConfig, FastaDataset),
    'sequence_per_line': (
        SequencePerLineDatasetConfig,
        SequencePerLineDataset,
    ),
    'jsonl': (JsonlDatasetConfig, JsonlDataset),
}


def get_dataset(kwargs: dict[str, Any]) -> Dataset:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - fasta
    - sequence_per_line
    - jsonl

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    Dataset
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
            f'Unknown dataset name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))