"""Fasta file reader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from embedding_workflow.readers.base import BaseReader
from embedding_workflow.readers.base import BaseReaderConfig
from embedding_workflow.utils import PathLike


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


class FastaReaderConfig(BaseReaderConfig):
    """Configuration for the FastaReader."""

    name: Literal['fasta'] = 'fasta'  # type: ignore[assignment]


class FastaReader(BaseReader):
    """Fasta file reader."""

    def read(self, data_file: PathLike) -> list[str]:
        """Read the data file.

        Parameters
        ----------
        data_file : Path
            The file to read.
        """
        return [
            ' '.join(seq.sequence.upper()) for seq in read_fasta(data_file)
        ]
