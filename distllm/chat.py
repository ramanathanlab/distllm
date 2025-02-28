"""Run evaluation suite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
import json

from pydantic import Field

from distllm.rag.search import RetrieverConfig
from distllm.utils import BaseConfig

from distllm.generate import PromptTemplate
from distllm.generate.prompts import IdentityPromptTemplate
from distllm.generate.prompts import IdentityPromptTemplateConfig
from distllm.rag.search import Retriever


class RetrievalAugmentedGenerationConfig(BaseConfig):
    """Configuration for the retrieval-augmented generation model."""

    # Settings for the generator
    generator_config: VLLMGeneratorConfig = Field(
        ...,
        description='Settings for the VLLM generator',
    )

    # Settings for the retriever
    retriever_config: Optional[RetrieverConfig] = Field(  # noqa: UP007
        None,
        description='Settings for the retriever',
    )

    def get_rag_model(self) -> RagGenerator:
        """Get the retrieval-augmented generation model."""
        # Initialize the generator
        generator = VLLMGenerator(self.generator_config)

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
    # rag_configs: list[RetrievalAugmentedGenerationConfig] = Field(
    #     ...,
    #     description='Settings for the retrieval-augmented generation models',
    # )
    rag_configs: RetrievalAugmentedGenerationConfig = Field(
        ...,
        description='Settings for this RAG application.',
    )    


def chat_with_model(config: ChatAppConfig) -> None:
    """Run the evaluation suite."""
    # Initialize the RAG model
    rag_model = config.rag_configs.get_rag_model()

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


class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the vLLM generator."""

    server: str = Field(
        ...,
        description='Cels machine you are running on, e.g, rbdgx1',
    )
    port: int = Field(
        ...,
        description='The port vLLM is listening to.',
    )
    api_key: str = Field(
        ...,
        description='The API key for vLLM server, e.g., CELS',
    )
    model: str = Field(
        ...,
        description='The model that vLLM server is running.',
    )
    temperature: float = Field(
        0.0,
        description='The temperature for sampling from the model.',
    )
    max_tokens: int = Field(
        1024,
        description='The maximum number of tokens to generate.',
    )

    def get_generator(self) -> VLLMGenerator:
        """Get the vLLM generator."""
        generator = VLLMGenerator(
            server=self.server,
            port=self.port,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return generator

class VLLMGenerator:
    def __init__(self, config: VLLMGeneratorConfig) -> None:
        self.server = config.server
        self.model = config.model
        self.port = config.port
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def generate(self,  
                 prompt:str,  
                 temperature: float,
                 max_tokens:int) -> str:
        
        '''
        Queries the local vLLM server with a prompt.
        '''

        # Use provided values or fall back to instance defaults
        temp_to_use = temperature if temperature is not None else self.temperature
        tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
      
        url = f"http://{self.server}.cels.anl.gov:{self.port}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": f"{self.model}",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp_to_use,
            "max_tokens": tokens_to_use
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Error: {response.status_code}")
            result = response.text
        
        return result


class RagGenerator:
    """RAG generator for generating responses to queries."""

    def __init__(
        self,
        generator: VLLMGenerator, # replace with vLLM.
        retriever: Retriever | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    #for now this will just be
    def generate(
        self,
        texts: str | list[str],
        prompt_template: PromptTemplate | None = None,
        retrieval_top_k: int = 5,
        retrieval_score_threshold: float = 0.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate a response to a query given a context.

        Parameters
        ----------
        texts : str | list[str]
            The query or queries to generate a response for.
        prompt_template : PromptTemplate, optional
            The prompt template to use. If None, will default
            to the identity prompt template, by default None.
        retrieval_top_k : int, optional
            The number of retrievals to return, by default 1.
        retrieval_score_threshold : float, optional
            The retrieval score threshold to use. Filters out
            retrievals with scores not satisfying the threshold,
            by default keep all.
        """
        # Use the identity prompt template if none is provided
        if prompt_template is None:
            prompt_template = IdentityPromptTemplate(
                IdentityPromptTemplateConfig(),
            )
        assert prompt_template is not None

        # Contexts are None unless there is a retriever (no-RAG baseline).
        contexts, scores = None, None
        if self.retriever is not None:
            # Retrieve the search results and query embedding
            results, _ = self.retriever.search(
                texts,
                top_k=retrieval_top_k,
                score_threshold=retrieval_score_threshold,
            )

            # Get the text that corresponds to the top indices
            contexts = [
                self.retriever.get_texts(indices)
                for indices in results.total_indices
            ]

            # Get the scores that correspond to the top indices
            scores = results.total_scores

        # Preprocess the text into prompts
        prompts = prompt_template.preprocess(texts, contexts, scores)

        # Generate a response to the query
        responses = self.generator.generate(prompt = prompts[0], temperature=temperature, max_tokens=max_tokens)

        return responses


if __name__ == '__main__':
    from argparse import ArgumentParser

    # Parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    # Load the configuration
    config = ChatAppConfig.from_yaml(args.config)

    chat_with_model(config)