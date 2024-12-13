"""Run evaluation suite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field

from distllm.generate import get_generator
from distllm.generate import LLMGeneratorConfigs
from distllm.rag.response_synthesizer import RagGenerator
from distllm.rag.search import RetrieverConfig
from distllm.rag.tasks import get_task
from distllm.utils import BaseConfig


class RetrievalAugmentedGenerationConfig(BaseConfig):
    """Configuration for the retrieval-augmented generation model."""

    # Settings for the generator
    generator_config: LLMGeneratorConfigs = Field(
        ...,
        description='Settings for the generator',
    )
    # Settings for the retriever
    retriever_config: Optional[RetrieverConfig] = Field(  # noqa: UP007
        None,
        description='Settings for the retriever',
    )

    def get_rag_model(self) -> RagGenerator:
        """Get the retrieval-augmented generation model."""
        # Initialize the generator
        generator = get_generator(self.generator_config.model_dump())

        # Initialize the retriever
        retriever = None
        if self.retriever_config is not None:
            retriever = self.retriever_config.get_retriever()

        # Initialize the RAG model
        rag_model = RagGenerator(generator=generator, retriever=retriever)

        return rag_model


class EvalSuiteConfig(BaseConfig):
    """Configuration for the evaluation suite."""

    # Settings for the retriever
    rag_configs: list[RetrievalAugmentedGenerationConfig] = Field(
        ...,
        description='Settings for the retrieval-augmented generation models',
    )
    # Tasks to evaluate on
    tasks: list[str] = Field(
        ...,
        description='The tasks to evaluate on',
    )
    # Directory to download the datasets to
    download_dir: Path = Field(
        ...,
        description='The directory to download the datasets to',
    )


def run_eval_suite(config: EvalSuiteConfig) -> None:
    """Run the evaluation suite."""
    # Evaluate the models on the tasks
    for rag_config in config.rag_configs:
        # Initialize the RAG model
        rag_model = rag_config.get_rag_model()

        # Evaluate the model on the tasks
        for task_name in config.tasks:
            # Initialize the task
            task = get_task(task_name, config.download_dir)

            # Evaluate the model on the task
            results = task.evaluate(rag_model)

            # TODO: Figure out what to do with the results
            # Print the results
            print(f'{rag_model} {task_name}: {results}')


if __name__ == '__main__':
    from argparse import ArgumentParser

    # Parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    # Load the configuration
    config = EvalSuiteConfig.from_yaml(args.config)

    run_eval_suite(config)
