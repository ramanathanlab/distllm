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

from distllm.datasets import DatasetConfigs
from distllm.embedders import Embedder
from distllm.embedders import EmbedderConfigs
from distllm.parsl import ComputeConfigTypes
from distllm.poolers import Pooler
from distllm.poolers import PoolerConfigs
from distllm.utils import BaseConfig

# TODO: For big models, see here: https://huggingface.co/docs/accelerate/usage_guides/big_modeling


@torch.no_grad()
def compute_embeddings(
    dataloader: DataLoader,
    embedder: Embedder,
    pooler: Pooler,
) -> np.ndarray:
    """Compute pooled hidden embeddings.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to use for batching the data.
    embedder : Embedder
        The embedder to use for inference.
    pooler : Pooler
        The pooler to use for pooling the embeddings.

    Returns
    -------
    np.ndarray
        A numpy array of pooled hidden embeddings.
    """
    import torch
    from tqdm import tqdm

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
        pooled_embeds = pooler.pool(embeddings, inputs.attention_mask)

        # Get the batch size
        batch_size = inputs.attention_mask.shape[0]

        # Store the pooled embeddings in the output buffer
        all_embeddings[idx : idx + batch_size, :] = pooled_embeds.cpu()

        # Increment the output buffer index by the batch size
        idx += batch_size

    return all_embeddings.numpy()


def embed_file(
    file: Path,
    dataset_kwargs: dict[str, Any],
    embedder_kwargs: dict[str, Any],
    pooler_kwargs: dict[str, Any],
) -> np.ndarray:
    """Embed a single file and return a numpy array with embeddings."""
    # Imports are here since this function is called in a parsl process

    from distllm.datasets import get_dataset
    from distllm.distributed_inference import compute_embeddings
    from distllm.embedders import get_embedder
    from distllm.poolers import get_pooler

    # Initialize the model and tokenizer
    embedder = get_embedder(embedder_kwargs, register=True)

    # Initialize the pooler
    pooler = get_pooler(pooler_kwargs)

    # Initialize the dataset
    dataset = get_dataset(dataset_kwargs)

    # Initialize the dataloader
    dataloader = dataset.get_dataloader(file, embedder)

    # Compute averaged hidden embeddings
    return compute_embeddings(dataloader, embedder, pooler)


def embed_and_save_file(
    file: Path,
    output_dir: Path,
    dataset_kwargs: dict[str, Any],
    embedder_kwargs: dict[str, Any],
    pooler_kwargs: dict[str, Any],
) -> None:
    """Embed a single file and save a numpy array with embeddings."""
    # Imports are here since this function is called in a parsl process
    import numpy as np

    from distllm.distributed_inference import embed_file

    # Embed the file
    embeddings = embed_file(
        file=file,
        dataset_kwargs=dataset_kwargs,
        embedder_kwargs=embedder_kwargs,
        pooler_kwargs=pooler_kwargs,
    )

    # Save the embeddings to disk
    np.save(output_dir / f'{file.stem}.npy', embeddings)


class Config(BaseConfig):
    """Configuration for distributed inference."""

    # An input directory containing .fasta files.
    input_dir: Path
    # An output directory to save the embeddings.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_patterns: list[str] = Field(default=['*'])
    # Strategy for reading the input files.
    dataset_config: DatasetConfigs
    # Settings for the embedder.
    embedder_config: EmbedderConfigs
    # Settings for the pooler.
    pooler_config: PoolerConfigs
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
        dataset_kwargs=config.dataset_config.model_dump(),
        embedder_kwargs=config.embedder_config.model_dump(),
        pooler_kwargs=config.pooler_config.model_dump(),
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
