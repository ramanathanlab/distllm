"""Module for data datasets."""

from __future__ import annotations

from typing import Any

from embedding_workflow.datasets.base import Dataset
from embedding_workflow.datasets.fasta import FastaDataset
from embedding_workflow.datasets.fasta import FastaDatasetConfig
from embedding_workflow.datasets.single_line import SequencePerLineDataset
from embedding_workflow.datasets.single_line import (
    SequencePerLineDatasetConfig,
)
from embedding_workflow.utils import BaseConfig

DatasetConfigs = FastaDatasetConfig | SequencePerLineDatasetConfig

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Dataset]]] = {
    'fasta': (FastaDatasetConfig, FastaDataset),
    'sequence_per_line': (
        SequencePerLineDatasetConfig,
        SequencePerLineDataset,
    ),
}


def get_dataset(kwargs: dict[str, Any]) -> Dataset:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - fasta
    - sequence_per_line

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
            f'Unknown embedder name: {name}.'
            f' Available: {set(STRATEGIES.keys())}',
        )

    # Get the config and classes
    config_cls, cls = strategy

    return cls(config_cls(**kwargs))
