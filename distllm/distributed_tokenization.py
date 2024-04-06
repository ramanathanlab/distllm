"""Distributed tokenization."""

from __future__ import annotations

import functools
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import field_validator

from distllm.parsl import ComputeConfigs
from distllm.utils import BaseConfig


class TokenizerConfig(BaseConfig):
    """Configuration for distributed tokenization."""

    text_field: str = Field(
        default='text',
        description='The name of the text field in the jsonl file',
    )
    tokenizer_name: str = Field(
        default='meta-llama/Llama-2-70b-chat-hf',
        description='The name of the tokenizer to use',
    )
    dotenv_path: Path = Field(
        default=Path('~/.env'),
        description='Path to the .env file',
    )
    save_labels: bool = Field(
        default=False,
        description='Whether to store a separate labels field in the dataset',
    )

    @field_validator('dotenv_path')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.expanduser().resolve()


def tokenizer_worker(
    input_path: Path,
    output_dir: Path,
    tokenizer_kwargs: dict[str, Any],
) -> None:
    """Tokenize a jsonl file and save the dataset to disk."""
    # Imports are here since this function is called in a parsl process

    import json
    import os
    import time
    from uuid import uuid4

    import datasets
    from datasets import Dataset
    from dotenv import load_dotenv
    from huggingface_hub import login
    from transformers import AutoTokenizer

    from distllm.distributed_tokenization import TokenizerConfig

    # Time the worker function
    start = time.time()

    # Initialize the tokenizer configuration
    config = TokenizerConfig(**tokenizer_kwargs)

    # Silence outputs
    datasets.logging.set_verbosity_warning()
    datasets.disable_progress_bars()

    # Load environment variables from .env file
    load_dotenv(config.dotenv_path)

    # Login to the huggingface hub
    login(os.getenv('HF_TOKEN'))

    # Read the jsonl file
    lines = input_path.read_text().strip().split('\n')
    content = [json.loads(line) for line in lines]

    # Extract the text data
    data = [item[config.text_field] for item in content]

    print(
        f'[timer] [Loaded dataset] [{input_path}]'
        f' in [{time.time() - start:.2f}] seconds',
    )
    t_start = time.time()

    # Initialize the tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = '0'
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Tokenize the text data
    result = tokenizer(data, return_tensors='np')

    print(
        f'[timer] [Tokenized text] [{input_path}]'
        f' in [{time.time() - t_start:.2f}] seconds',
    )
    t_start = time.time()

    mapping = {
        'input_ids': result.input_ids,
        'attention_mask': result.attention_mask,
    }

    if config.save_labels:
        mapping['labels'] = result.input_ids

    # Create a dataset
    dataset = Dataset.from_dict(mapping)

    if not len(dataset):
        print(f'Empty dataset for {input_path}')
        return

    # Create the output directory for the dataset
    dataset_dir = output_dir / f'{uuid4()}'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write the dataset to disk
    dataset.save_to_disk(dataset_dir)

    print(
        f'[timer] [Saved dataset] [{input_path}]'
        f' in [{time.time() - t_start:.2f}] seconds',
    )
    print(
        f'[timer] [Finished tokenizing] [{input_path}]'
        f' in [{time.time() - start:.2f}] seconds',
    )


class Config(BaseConfig):
    """Configuration for distributed tokenization."""

    # An input directory containing the files to tokenize.
    input_dir: Path
    # An output directory to save the tokenized text.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_patterns: list[str] = Field(default=['*'])
    # Configuration for the tokenizer
    tokenizer_config: TokenizerConfig
    # Configuration for the compute platform
    compute_config: ComputeConfigs

    @field_validator('input_dir', 'output_dir')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.resolve()


if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Tokenize text')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the .yaml configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = Config.from_yaml(args.config)

    # Create a directory for the outputs
    tokenize_dir = config.output_dir / 'tokenized_text'

    # Make the output directory
    tokenize_dir.mkdir(parents=True, exist_ok=True)

    # Log the configuration
    config.write_yaml(config.output_dir / 'config.yaml')

    # Set the static arguments of the worker function
    worker_fn = functools.partial(
        tokenizer_worker,
        output_dir=tokenize_dir,
        tokenizer_kwargs=config.tokenizer_config.model_dump(),
    )

    # Collect all input files
    input_files = []
    for pattern in config.glob_patterns:
        input_files.extend(list(config.input_dir.glob(pattern)))

    # Log the input files to stdout
    print(f'Found {len(input_files)} input files to tokenize')

    # Set the parsl compute settings
    parsl_config = config.compute_config.get_config(
        config.output_dir / 'parsl',
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        list(pool.map(worker_fn, input_files))
