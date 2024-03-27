"""Jsonl file dataset with sentence chunking.

Note: Our semantic chunking implementation is adapted from:
https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/semantic_splitter.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Literal

from pydantic import Field
from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import DataCollator
from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.encoders.base import Encoder
from distllm.utils import BaseConfig


def split_by_sentence_tokenizer() -> Callable[[str], list[str]]:
    """Split the text into sentences using nltk."""
    import nltk

    tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    # get the spans and then return the sentences
    # using the start index of each span
    # instead of using end, use the start of the next span if available
    def split(text: str) -> list[str]:
        spans = list(tokenizer.span_tokenize(text))
        sentences = []
        for i, span in enumerate(spans):
            start = span[0]
            end = spans[i + 1][0] if i < len(spans) - 1 else len(text)
            sentences.append(text[start:end])

        return sentences

    return split


def sentences_to_buffers(split: list[str], buffer_size: int) -> list[str]:
    """Group split into buffers."""
    buffers = []
    for i in range(len(split)):
        combined = ''.join(
            split[j]
            for j in range(
                max(0, i - buffer_size),
                min(i + 1 + buffer_size, len(split)),
            )
        )
        buffers.append(combined)
    return buffers


class JsonlChunkDatasetConfig(BaseConfig):
    """Configuration for the JsonlChunkDatasetConfig."""

    # The name of the dataset
    name: Literal['jsonl_chunk'] = 'jsonl_chunk'  # type: ignore[assignment]

    # The name of the text field in the jsonl file
    text_field: str = 'text'
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Whether to pin memory for the dataloader.
    pin_memory: bool = True

    # The length threshold to filter out small buffers
    # (number of characters in the buffer)
    min_buffer_length: int = Field(
        default=750,
        description=(
            'The minimum number of characters to consider a buffer. '
            'Buffers with fewer characters will be filtered out.'
            'This helps remove citations, etc from the text.'
        ),
    )

    buffer_size: int = Field(
        default=1,
        description=(
            'The number of sentences to group together when evaluating '
            'semantic similarity. Set to 1 to consider each sentence '
            'individually. Set to >1 to group sentences together.'
        ),
    )


class JsonlChunkDataset:
    """Sequence per line file dataset with sentence chunking."""

    def __init__(self, config: JsonlChunkDatasetConfig):
        """Initialize the dataset."""
        self.config = config

        # TODO: In the future we may want a splitter abstraction.
        self.splitter = split_by_sentence_tokenizer()

    def get_dataloader(
        self,
        data_file: Path,
        encoder: Encoder,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset.

        Parameters
        ----------
        data_file : Path
            The file to read.
        encoder : Encoder
            The encoder instance.

        Returns
        -------
        DataLoader
            The dataloader instance.

        Raises
        ------
        ValueError
            If the metadata is empty.
        """
        # Read the jsonl file
        lines = data_file.read_text().strip().split('\n')
        content: list[dict[str, Any]] = [json.loads(line) for line in lines]

        # Extract the text data
        data = [item.pop(self.config.text_field) for item in content]

        # Extract the metadata if needed, note that the metadata is
        # is a dictionary of all the other fields in the jsonl file
        # except for the text field since that is already extracted.
        metadata = content

        # Check if metadata is empty
        if not metadata or any(not item for item in metadata):
            raise ValueError('Metadata is empty. Please check the jsonl file.')

        # Split the data based on the split criteria
        # Each input text is split into a list of str.
        splits = [self.splitter(text) for text in data]

        # Group each text split into windowed buffers
        buffers, metadatas = [], []
        for idx, split in enumerate(splits):
            bufs = sentences_to_buffers(split, self.config.buffer_size)
            buffers.extend(bufs)
            # Add the split to the metadata to be able to unpack the
            # semantic chunks properly
            for sentence in split:
                mdata = metadata[idx].copy()
                mdata['sentence'] = sentence
                metadatas.append(mdata)

        # Apply a length filter to remove any small buffers
        filter_indices = [
            i
            for i, buf in enumerate(buffers)
            if len(buf) > self.config.min_buffer_length
        ]
        buffers = [buffers[i] for i in filter_indices]
        metadatas = [metadatas[i] for i in filter_indices]

        # Instantiate the dataloader
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(buffers, metadatas),
            collate_fn=DataCollator(encoder.tokenizer),
        )
