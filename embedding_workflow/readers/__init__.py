"""Module for data readers."""

from __future__ import annotations

from typing import Any

from embedding_workflow.readers.base import BaseReader
from embedding_workflow.readers.fasta import FastaReader
from embedding_workflow.readers.fasta import FastaReaderConfig
from embedding_workflow.readers.single_line import SingleSequencePerLineReader
from embedding_workflow.readers.single_line import (
    SingleSequencePerLineReaderConfig,
)

ReaderConfigTypes = FastaReaderConfig | SingleSequencePerLineReaderConfig
ReaderTypes = FastaReader | SingleSequencePerLineReader

READER_STRATEGIES: dict[
    str,
    tuple[type[ReaderConfigTypes], type[ReaderTypes]],
] = {
    'fasta': (FastaReaderConfig, FastaReader),
    'single_sequence_per_line': (
        SingleSequencePerLineReaderConfig,
        SingleSequencePerLineReader,
    ),
}


def get_reader(kwargs: dict[str, Any]) -> ReaderTypes:
    """Get the instance based on the name and kwargs.

    Caches the reader instance based on the reader name and kwargs.
    Currently supports the following readers: .

    Parameters
    ----------
    kwargs : dict[str, Any]
        The reader configuration. Contains an extra `name` argument
        to specify the embedder to use.

    Returns
    -------
    ReaderTypes
        The reader instance.

    Raises
    ------
    ValueError
        If the embedder name is unknown.
    """
    name = kwargs.get('name', '')
    strategy = READER_STRATEGIES.get(name)
    if not strategy:
        raise ValueError(f'Unknown embedder name: {name}')

    # Get the config and reader classes
    config_cls, reader_cls = strategy

    # Initialize the config
    config = config_cls(**kwargs)

    # Return the reader instance
    return reader_cls(config)
