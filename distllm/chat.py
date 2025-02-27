"""Run evaluation suite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field

from distllm.generate import get_generator
from distllm.generate import LLMGeneratorConfigs
from distllm.rag.response_synthesizer import RagGenerator
from distllm.rag.search import RetrieverConfig
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


class ChatAppConfig(BaseConfig):
    """Configuration for the evaluation suite."""

    # Settings for the retriever
    rag_configs: list[RetrievalAugmentedGenerationConfig] = Field(
        ...,
        description='Settings for the retrieval-augmented generation models',
    )
    # What other configs would a chat application need?


def chat_with_model(config: ChatAppConfig) -> None:
    """Run the evaluation suite."""
    # Evaluate the models on the tasks
    for rag_config in config.rag_configs:
        # Initialize the RAG model
        rag_model = rag_config.get_rag_model()

        # Start an interactive chat session
        print(f'Chatting with model: {rag_model}')
        while True:
            # Get the user input
            user_input = input('You: ')

            # Generate a response
            response = rag_model.generate(
                [user_input],
                prompt_template=None,
                retrieval_top_k=20,
                retrieval_score_threshold=0.5,
            )

            # Print the response
            print(f'Model: {response}')


if __name__ == '__main__':
    from argparse import ArgumentParser

    # Parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    # Load the configuration
    config = ChatAppConfig.from_yaml(args.config)

    chat_with_model(config)
