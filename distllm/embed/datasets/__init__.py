"""Module for data datasets."""

from __future__ import annotations

from typing import Any
from typing import Union

from distllm.embed.datasets.base import Dataset
from distllm.embed.datasets.fasta import FastaDataset
from distllm.embed.datasets.fasta import FastaDatasetConfig
from distllm.embed.datasets.huggingface import HuggingFaceDataset
from distllm.embed.datasets.huggingface import HuggingFaceDatasetConfig
from distllm.embed.datasets.jsonl import JsonlDataset
from distllm.embed.datasets.jsonl import JsonlDatasetConfig
from distllm.embed.datasets.jsonl_chunk import JsonlChunkDataset
from distllm.embed.datasets.jsonl_chunk import JsonlChunkDatasetConfig
from distllm.embed.datasets.single_line import SequencePerLineDataset
from distllm.embed.datasets.single_line import SequencePerLineDatasetConfig
from distllm.utils import BaseConfig

DatasetConfigs = Union[
    FastaDatasetConfig,
    SequencePerLineDatasetConfig,
    JsonlDatasetConfig,
    JsonlChunkDatasetConfig,
    HuggingFaceDatasetConfig,
]

STRATEGIES: dict[str, tuple[type[BaseConfig], type[Dataset]]] = {
    'fasta': (FastaDatasetConfig, FastaDataset),
    'sequence_per_line': (
        SequencePerLineDatasetConfig,
        SequencePerLineDataset,
    ),
    'jsonl': (JsonlDatasetConfig, JsonlDataset),
    'jsonl_chunk': (JsonlChunkDatasetConfig, JsonlChunkDataset),
    'huggingface': (HuggingFaceDatasetConfig, HuggingFaceDataset),
}


def get_dataset(kwargs: dict[str, Any]) -> Dataset:
    """Get the instance based on the kwargs.

    Currently supports the following strategies:
    - fasta
    - sequence_per_line
    - jsonl
    - jsonl_chunk
    - huggingface

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
