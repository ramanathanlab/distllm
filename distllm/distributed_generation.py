"""Distributed inference for prompting a generator."""

from __future__ import annotations

import functools
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import field_validator

from distllm.generate import LLMGeneratorConfigs
from distllm.generate import PromptConfigs
from distllm.generate import ReaderConfigs
from distllm.generate import WriterConfigs
from distllm.parsl import ComputeConfigs
from distllm.utils import BaseConfig


def generate_worker(  # noqa: PLR0913
    input_path: Path,
    output_dir: Path,
    prompt_kwargs: dict[str, Any],
    reader_kwargs: dict[str, Any],
    writer_kwargs: dict[str, Any],
    generator_kwargs: dict[str, Any],
) -> None:
    """Generate text for a file and save to the output directory."""
    import time
    from uuid import uuid4

    from distllm.generate import get_generator
    from distllm.generate import get_prompt
    from distllm.generate import get_reader
    from distllm.generate import get_writer

    # Initialize the generator
    generator = get_generator(generator_kwargs, register=True)

    # Initialize the reader
    reader = get_reader(reader_kwargs)

    # Initialize the writer
    writer = get_writer(writer_kwargs)

    # Initialize the prompt
    prompt = get_prompt(prompt_kwargs)

    start = time.time()

    # Read the text from the file
    text, paths = reader.read(input_path)

    # Preprocess the text
    prompts = prompt.preprocess(text)

    # Time the generation
    start = time.time()

    # Generate response for each text
    responses = generator.generate(prompts)

    print(f'Generated responses in {time.time() - start:.2f} seconds')

    # Postprocess the responses
    results = prompt.postprocess(responses)

    # Filter out any empty responses (e.g., empty strings)
    text = [t for t, r in zip(text, results) if r]
    paths = [p for p, r in zip(paths, results) if r]
    results = [r for r in results if r]

    # Create the output directory for the dataset
    dataset_dir = output_dir / f'{uuid4()}'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Write the responses to disk
    writer.write(dataset_dir, paths, text, results)


class Config(BaseConfig):
    """Configuration for distributed inference."""

    # An input directory containing the files/directory to generate text for.
    input_dir: Path
    # An output directory to save the results.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_patterns: list[str] = Field(default=['*'])
    # Settings for the prompt.
    prompt_config: PromptConfigs
    # Settings for the reader.
    reader_config: ReaderConfigs
    # Settings for the writer.
    writer_config: WriterConfigs
    # Settings for the generator.
    generator_config: LLMGeneratorConfigs
    # Settings for the parsl compute backend.
    compute_config: ComputeConfigs

    @field_validator('input_dir', 'output_dir')
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        """Resolve the path to an absolute path."""
        return value.resolve()

    @field_validator('output_dir')
    @classmethod
    def validate_path_not_exists(cls, value: Path) -> Path:
        """Validate that the output directory does not exist."""
        if value.exists():
            raise ValueError(f'Output directory {value} already exists')
        return value


if __name__ == '__main__':
    # Parse arguments from the command line
    parser = ArgumentParser(description='Generate text')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to the .yaml configuration file',
    )
    args = parser.parse_args()

    # Load the configuration
    config = Config.from_yaml(args.config)

    # Create a directory for the generated text
    output_dir = config.output_dir / 'generated_text'

    # Make the output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log the configuration
    config.write_yaml(config.output_dir / 'config.yaml')

    # Set the static arguments of the worker function
    worker_fn = functools.partial(
        generate_worker,
        output_dir=output_dir,
        prompt_kwargs=config.prompt_config.model_dump(),
        reader_kwargs=config.reader_config.model_dump(),
        writer_kwargs=config.writer_config.model_dump(),
        generator_kwargs=config.generator_config.model_dump(),
    )

    # Collect all input files
    input_paths = []
    for pattern in config.glob_patterns:
        input_paths.extend(list(config.input_dir.glob(pattern)))

    # Set the parsl compute settings
    parsl_config = config.compute_config.get_config(
        config.output_dir / 'parsl',
    )

    # Distribute the input files across processes
    with ParslPoolExecutor(parsl_config) as pool:
        pool.map(worker_fn, input_paths)
