"""Semantic Chunk Embedder.

Note: Our semantic chunking implementation is adapted from:
https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/semantic_splitter.py
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import DataCollator
from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.embedders.base import EmbedderResult
from distllm.embed.embedders.full_sequence import compute_embeddings
from distllm.embed.encoders.base import Encoder
from distllm.embed.poolers.base import Pooler
from distllm.utils import BaseConfig


def calculate_distances_between_buffer(
    buffer_embeds: np.ndarray,
) -> np.ndarray:
    """Calculate embedding distances between chunks.

    Parameters
    ----------
    buffer_embeds : np.ndarray
        The buffer embeddings.

    Returns
    -------
    np.ndarray
        The distances between the embeddings.
    """
    # We need to convert the buffer embeddings to float32
    # to avoid overflow errors when computing the dot product and norm
    embeddings = buffer_embeds.astype(np.float32)

    # Compute the embedding distances using a sliding window
    distances = np.zeros(len(embeddings) - 1)
    for i in range(len(embeddings) - 1):
        embedding_current = embeddings[i]
        embedding_next = embeddings[i + 1]

        similarity = np.dot(embedding_current, embedding_next) / (
            np.linalg.norm(embedding_current) * np.linalg.norm(embedding_next)
        )

        distances[i] = 1 - similarity

    return distances


def build_chunks(
    distances: np.ndarray,
    breakpoint_percentile_threshold: int,
) -> list[tuple[int, int]]:
    """Build node chunks based on distances.

    Parameters
    ----------
    distances : np.ndarray
        The distances between the embeddings.
    breakpoint_percentile_threshold : int
        The percentile of cosine dissimilarity that must be exceeded
        between a group of sentences and the next to form a chunk. The
        smaller this number is, the more chunks will be generated.

    Returns
    -------
    list[tuple[int, int]]
        The list of tuples with the start and end indices.
    """
    # If, for some reason we didn't get any distances (i.e. very,
    # very small documents) just treat the whole document as a single node
    if len(distances) == 0:
        return [(0, 0)]

    breakpoint_distance_threshold = np.percentile(
        distances,
        breakpoint_percentile_threshold,
    )

    indices_above_threshold = [
        i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
    ]

    # Chunk sentences into semantic groups based on percentile breakpoints
    start_index = 0
    index_groups = []
    for index in indices_above_threshold:
        index_groups.append((start_index, index + 1))
        start_index = index + 1

    # Add the last group
    index_groups.append((start_index, len(distances) + 1))

    return index_groups


def compute_semantic_chunks(
    dataloader: DataLoader,
    encoder: Encoder,
    pooler: Pooler,
    breakpoint_percentile_threshold: int,
    min_chunk_length: int,
) -> InMemoryDataset:
    """Compute semantic chunked embeddings.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to use for batching the data.
    encoder : Encoder
        The encoder to use for inference.
    pooler : Pooler
        The pooler to use for pooling the embeddings.
    breakpoint_percentile_threshold : int
        The percentile of cosine dissimilarity that must be exceeded
        between a group of sentences and the next to form a chunk. The
        smaller this number is, the more chunks will be generated.
    min_chunk_length : int
        The minimum length of a chunk (number of characters) to
        filter out any small chunks.

    Returns
    -------
    InMemoryDataset
        The dataset with the semantically-chunked text and metadata.

    Raises
    ------
    ValueError
        If the dataloader dataset does not have metadata.
    ValueError
        If the metadata does not have a path.
    """
    if dataloader.dataset.metadata is None:
        raise ValueError('Metadata is required for semantic chunking.')

    if dataloader.dataset.metadata[0].get('path') is None:
        raise ValueError('Metadata path is required for semantic chunking.')

    # Group the data such that we only compute distances between
    # buffers within the same document.
    document_indices = []
    current_idx = 0
    current_doc = dataloader.dataset.metadata[0]['path']
    for i, metadata in enumerate(dataloader.dataset.metadata):
        if metadata['path'] != current_doc:
            document_indices.append((current_idx, i))
            current_idx = i
            current_doc = metadata['path']
    document_indices.append((current_idx, len(dataloader.dataset)))

    # Compute embeddings for each buffer (num_examples, embedding_size)
    buffer_embeds = compute_embeddings(dataloader, encoder, pooler)

    dataset_indices = []
    for doc_start, doc_end in document_indices:
        # Calculate distances between sentence groups
        distances = calculate_distances_between_buffer(
            buffer_embeds[doc_start:doc_end],
        )

        # Chunk the sentences into semantic groups
        # Stores indices into buffer_embeds [(0, 5), (6, 8), ...]
        index_groups = build_chunks(distances, breakpoint_percentile_threshold)
        # The start and end indices refer to the indices in the original
        # dataset where documents are split.
        # The start_idx and end_idx refer to the buffer embeddings boundaries
        # within the document.
        dataset_indices.extend(
            [
                (doc_start + start_idx, doc_start + end_idx)
                for start_idx, end_idx in index_groups
            ],
        )

    # Group the data by the index groups
    data = []
    for start, end in dataset_indices:
        group = dataloader.dataset.metadata[start:end]
        chunk = ''.join(g['sentence'] for g in group)
        data.append(chunk)

    # Get the metadata for the chunks
    metadata = [
        dataloader.dataset.metadata[start] for start, _ in dataset_indices
    ]

    # Apply a length filter to remove small chunks
    filter_indices = [
        i for i, x in enumerate(data) if len(x) > min_chunk_length
    ]
    data = [data[i] for i in filter_indices]
    metadata = [metadata[i] for i in filter_indices]

    # Drop the splits from the metadata
    for meta in metadata:
        meta.pop('sentence')

    return InMemoryDataset(data, metadata)


class SemanticChunkEmbedderConfig(BaseConfig):
    """Configuration for the full sequence embedder."""

    name: Literal['semantic_chunk'] = 'semantic_chunk'  # type: ignore[assignment]
    breakpoint_percentile_threshold: int = Field(
        90,
        description='The percentile of cosine dissimilarity that must be '
        'exceeded between a group of sentences and the next to form a chunk. '
        'The smaller this number is, the more chunks will be generated.',
    )
    chunk_batch_size: int = Field(
        8,
        description='The batch size for the chunked text.',
    )
    min_chunk_length: int = Field(
        750,
        description='The minimum length of a chunk (number of characters) to '
        'filter out any small chunks.',
    )
    normalize_embeddings: bool = Field(
        False,
        description='Whether to return normalized the embeddings.',
    )


class SemanticChunkEmbedder:
    """Embedder for semantic chunking."""

    def __init__(self, config: SemanticChunkEmbedderConfig) -> None:
        """Initialize the embedder with the configuration."""
        self.config = config

    def embed(
        self,
        dataloader: DataLoader,
        encoder: Encoder,
        pooler: Pooler,
    ) -> EmbedderResult:
        """Embed the sequences.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader to use for batching the data.
        encoder : Encoder
            The encoder to use for inference.
        pooler : Pooler
            The pooler to use for pooling the embeddings.

        Returns
        -------
        EmbedderResult
            Dataclass with the embeddings, text, and optional metadata.
        """
        dataset = compute_semantic_chunks(
            dataloader=dataloader,
            encoder=encoder,
            pooler=pooler,
            breakpoint_percentile_threshold=self.config.breakpoint_percentile_threshold,
            min_chunk_length=self.config.min_chunk_length,
        )

        # Make a new dataloader with the chunked data
        chunked_dataloader = DataLoader(
            pin_memory=dataloader.pin_memory,
            batch_size=self.config.chunk_batch_size,
            num_workers=dataloader.num_workers,
            dataset=dataset,
            collate_fn=DataCollator(encoder.tokenizer),
        )

        # Compute embeddings for each chunk
        chunked_embeds = compute_embeddings(
            dataloader=chunked_dataloader,
            encoder=encoder,
            pooler=pooler,
            normalize=self.config.normalize_embeddings,
        )

        # Return the result
        return EmbedderResult(
            embeddings=chunked_embeds,
            text=dataset.data,
            metadata=dataset.metadata,
        )
