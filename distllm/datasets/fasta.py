"""Fasta file dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader

from distllm.datasets.utils import DataCollator
from distllm.datasets.utils import InMemoryDataset
from distllm.embedders import Embedder
from distllm.utils import BaseConfig
from distllm.utils import PathLike


@dataclass
class Sequence:
    """Biological sequence dataclass."""

    sequence: str
    """Biological sequence (Nucleotide/Amino acid sequence)."""
    tag: str
    """Sequence description tag."""


def read_fasta(fasta_file: PathLike) -> list[Sequence]:
    """Read fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile('^>', re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace('\n', '')
        for seq in non_parsed_seqs
        for line in seq.split('\n', 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag)
        for seq, tag in zip(lines[1::2], lines[::2])
    ]


def write_fasta(
    sequences: Sequence | list[Sequence],
    fasta_file: PathLike,
    mode: str = 'w',
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f'>{seq.tag}\n{seq.sequence}\n')


class FastaDatasetConfig(BaseConfig):
    """Configuration for the FastaDataset."""

    name: Literal['fasta'] = 'fasta'  # type: ignore[assignment]

    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Whether to pin memory for the dataloader.
    pin_memory: bool = True


class FastaDataset:
    """Fasta file dataset."""

    def __init__(self, config: FastaDatasetConfig) -> None:
        """Initialize the dataset."""
        self.config = config

    def get_dataloader(
        self,
        data_file: Path,
        embedder: Embedder,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset.

        Parameters
        ----------
        data_file : Path
            The file to read.
        embedder : Embedder
            The embedder instance.

        Returns
        -------
        DataLoader
            The dataloader instance.
        """
        # Read the sequences from the fasta file
        data = [
            ' '.join(seq.sequence.upper()) for seq in read_fasta(data_file)
        ]

        # Instantiate the dataloader
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(data),
            collate_fn=DataCollator(embedder.tokenizer),
        )
