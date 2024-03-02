"""Distributed inference for generating embeddings."""

from __future__ import annotations

import functools
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import torch
from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import field_validator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from embedding_workflow.embedders import BaseEmbedder
from embedding_workflow.embedders import EmbedderConfigTypes
from embedding_workflow.parsl import ComputeConfigTypes
from embedding_workflow.readers import ReaderConfigTypes
from embedding_workflow.utils import BaseModel

# TODO: For big models, see here: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
# Documentation on using accelerate for inference: https://huggingface.co/docs/accelerate/usage_guides/distributed_inference


class InMemoryDataset(Dataset):
    """Holds the data in memory for efficient batching."""

    def __init__(self, data: list[str]) -> None:
        self.data = data

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        """Get an item from the dataset."""
        return self.data[idx]


class DataCollator:
    """Data collator for batching sequences."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """Initialize the data collator."""
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> BatchEncoding:
        """Collate the batch of sequences."""
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )


def average_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Get the sequence lengths
    seq_lengths = attention_mask.sum(axis=1)

    # Set the attention mask to 0 for start and end tokens
    attention_mask[:, 0] = 0
    attention_mask[:, seq_lengths - 1] = 0

    # Create a mask for the pooling operation (B, SeqLen, HiddenDim)
    pool_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape)

    # Sum the embeddings over the sequence length (use the mask to avoid
    # pad, start, and stop tokens)
    sum_embeds = torch.sum(embeddings * pool_mask, 1)

    # Avoid division by zero for zero length sequences by clamping
    sum_mask = torch.clamp(pool_mask.sum(1), min=1e-9)

    # Compute mean pooled embeddings for each sequence
    return sum_embeds / sum_mask


@torch.no_grad()
def compute_avg_embeddings(
    embedder: BaseEmbedder,
    dataloader: DataLoader,
) -> np.ndarray:
    """Compute averaged hidden embeddings.

    Parameters
    ----------
    embedder : BaseEmbedder
        The embedder to use for inference.
    dataloader : DataLoader
        The dataloader to use for batching the data.

    Returns
    -------
    np.ndarray
        A numpy array of averaged hidden embeddings.
    """
    import torch
    from tqdm import tqdm

    from embedding_workflow.distributed_inference import average_pool

    # Get the number of embeddings and the embedding size
    num_embeddings = len(dataloader.dataset)

    # Initialize a torch tensor for storing embeddings in host memory
    all_embeddings = torch.empty(
        (num_embeddings, embedder.embedding_size),
        dtype=embedder.dtype,
    )

    # Index for storing embeddings
    idx = 0

    for batch in tqdm(dataloader):
        # Move the batch to the model device
        inputs = batch.to(embedder.device)

        # Get the model outputs with a forward pass
        embeddings = embedder.embed(inputs)

        # Compute the average pooled embeddings
        pooled_embeds = average_pool(embeddings, inputs.attention_mask)

        # Get the batch size
        batch_size = inputs.attention_mask.shape[0]

        # Store the pooled embeddings in the output buffer
        all_embeddings[idx : idx + batch_size, :] = pooled_embeds.cpu()

        # Increment the output buffer index by the batch size
        idx += batch_size

    return all_embeddings.numpy()


def embed_file(
    file: Path,
    batch_size: int,
    num_data_workers: int,
    reader_kwargs: dict[str, Any],
    embedder_kwargs: dict[str, Any],
) -> np.ndarray:
    """Embed a single file and return a numpy array with embeddings."""
    # Imports are here since this function is called in a parsl process
    from torch.utils.data import DataLoader

    from embedding_workflow.distributed_inference import compute_avg_embeddings
    from embedding_workflow.distributed_inference import DataCollator
    from embedding_workflow.distributed_inference import InMemoryDataset
    from embedding_workflow.embedders import EmbedderTypes
    from embedding_workflow.embedders import get_embedder
    from embedding_workflow.readers import get_reader

    # Initialize the model and tokenizer
    embedder: EmbedderTypes = get_embedder(embedder_kwargs, register=True)

    # Initialize the data reader
    reader = get_reader(reader_kwargs)

    # Read the data
    data = reader.read(file)

    # Build a torch dataset for efficient batching
    dataloader = DataLoader(
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_data_workers,
        dataset=InMemoryDataset(data),
        collate_fn=DataCollator(embedder.tokenizer),
    )

    # Compute averaged hidden embeddings
    return compute_avg_embeddings(embedder, dataloader)


def embed_and_save_file(  # noqa: PLR0913
    file: Path,
    output_dir: Path,
    batch_size: int,
    num_data_workers: int,
    reader_kwargs: dict[str, Any],
    embedder_kwargs: dict[str, Any],
) -> None:
    """Embed a single file and save a numpy array with embeddings."""
    # Imports are here since this function is called in a parsl process
    import numpy as np

    from embedding_workflow.distributed_inference import embed_file

    # Embed the file
    embeddings = embed_file(
        file=file,
        batch_size=batch_size,
        num_data_workers=num_data_workers,
        reader_kwargs=reader_kwargs,
        embedder_kwargs=embedder_kwargs,
    )

    # Save the embeddings to disk
    np.save(output_dir / f'{file.stem}.npy', embeddings)


class Config(BaseModel):
    """Configuration for distributed inference."""

    # An input directory containing .fasta files.
    input_dir: Path
    # An output directory to save the embeddings.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_patterns: list[str] = Field(default=['*'])
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Strategy for reading the input files.
    reader_config: ReaderConfigTypes
    # Settings for the embedder.
    embedder_config: EmbedderConfigTypes
    # Settings for the parsl compute backend.
    compute_config: ComputeConfigTypes

    @field_validator('input_dir', 'output_dir')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.resolve()


if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Embed text')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the .yaml configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = Config.from_yaml(args.config)

    # Create a directory for the embeddings
    embedding_dir = config.output_dir / 'embeddings'

    # Make the output directory
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Log the configuration
    config.write_yaml(config.output_dir / 'config.yaml')

    # Set the static arguments of the worker function
    worker_fn = functools.partial(
        embed_and_save_file,
        output_dir=embedding_dir,
        batch_size=config.batch_size,
        num_data_workers=config.num_data_workers,
        reader_kwargs=config.reader_config.model_dump(),
        embedder_kwargs=config.embedder_config.model_dump(),
    )

    # Collect all input files
    input_files = []
    for pattern in config.glob_patterns:
        input_files.extend(list(config.input_dir.glob(pattern)))

    # Set the parsl compute settings
    parsl_config = config.compute_config.get_config(
        config.output_dir / 'parsl',
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        pool.map(worker_fn, input_files)
