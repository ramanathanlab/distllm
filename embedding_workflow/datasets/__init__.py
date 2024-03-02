"""Module for data datasets."""

from __future__ import annotations

from typing import Any

from embedding_workflow.datasets.base import BaseDataset
from embedding_workflow.datasets.base import BaseDatasetConfig
from embedding_workflow.datasets.fasta import FastaDataset
from embedding_workflow.datasets.fasta import FastaDatasetConfig
from embedding_workflow.datasets.single_line import (
    SingleSequencePerLineDataset,
)
from embedding_workflow.datasets.single_line import (
    SingleSequencePerLineDatasetConfig,
)

DatasetConfigTypes = FastaDatasetConfig | SingleSequencePerLineDatasetConfig
DatasetTypes = FastaDataset | SingleSequencePerLineDataset

STRATEGIES: dict[
    str,
    tuple[type[DatasetConfigTypes], type[DatasetTypes]],
] = {
    'fasta': (FastaDatasetConfig, FastaDataset),
    'single_sequence_per_line': (
        SingleSequencePerLineDatasetConfig,
        SingleSequencePerLineDataset,
    ),
}


def get_dataset(kwargs: dict[str, Any]) -> DatasetTypes:
    """Get the instance based on the name and kwargs.

    Currently supports the following strategies:
    - fasta
    - single_sequence_per_line

    Parameters
    ----------
    kwargs : dict[str, Any]
        The configuration. Contains a `name` argument
        to specify the strategy to use.

    Returns
    -------
    DatasetTypes
        The dataset instance.

    Raises
    ------
    ValueError
        If the `name` is unknown.
    """
    name = kwargs.get('name', '')
    strategy = STRATEGIES.get(name)
    if not strategy:
        raise ValueError(f'Unknown dataset name: {name}')

    # Get the config and classes
    config_cls, cls = strategy

    # Initialize the config
    config = config_cls(**kwargs)

    # Return the instance
    return cls(config)
