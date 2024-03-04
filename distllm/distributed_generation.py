"""Distributed inference for prompting a generator."""

from __future__ import annotations

import functools
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import field_validator

from distllm.generators import LLMGeneratorConfigs
from distllm.parsl import ComputeConfigTypes
from distllm.prompts import PromptConfigs
from distllm.utils import BaseConfig


def generate(
    file: Path,
    output_dir: Path,
    prompt_kwargs: dict[str, Any],
    generator_kwargs: dict[str, Any],
) -> None:
    """Generate text for a file and save to the output directory."""
    import json

    from distllm.generators import get_generator
    from distllm.prompts import get_prompt

    # Initialize the generator
    generator = get_generator(generator_kwargs, register=True)

    # Initialize the prompt
    prompt = get_prompt(prompt_kwargs)

    # Load the jsonl file contents into a list of dictionaries
    # which stores the path and text fields
    with open(file) as f:
        text = [json.loads(line) for line in f]

    # Preprocess the text
    prompts = prompt.preprocess(text)

    # Generate response for each text
    responses = generator.generate(prompts)

    # Postprocess the responses
    results = prompt.postprocess(responses)

    # Format the output dictionary
    outputs = [{'path': str(file), 'result': result} for result in results]

    # Generate an output jsonl string for each item
    # Merge parsed documents into a single string of JSON lines
    lines = ''.join(f'{json.dumps(output)}\n' for output in outputs)

    # Store the JSON lines strings to a disk using a single write operation
    with open(output_dir / f'{generator.unique_id}.jsonl', 'a+') as f:
        f.write(lines)


class Config(BaseConfig):
    """Configuration for distributed inference."""

    # An input directory containing .fasta files.
    input_dir: Path
    # An output directory to save the embeddings.
    output_dir: Path
    # A set of glob patterns to match the input files.
    glob_patterns: list[str] = Field(default=['*'])
    # Settings for the prompt.
    prompt_config: PromptConfigs
    # Settings for the generator.
    generator_config: LLMGeneratorConfigs
    # Settings for the parsl compute backend.
    compute_config: ComputeConfigTypes

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
        generate,
        output_dir=output_dir,
        prompt_kwargs=config.prompt_config.model_dump(),
        generator_kwargs=config.generator_config.model_dump(),
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
