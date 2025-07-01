"""Serves as a chat interface to the RAG datasets built with distllm."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import openai
import requests
from dotenv import load_dotenv
from pydantic import Field

from distllm.generate.prompts import IdentityPromptTemplate
from distllm.generate.prompts import IdentityPromptTemplateConfig
from distllm.rag.search import Retriever
from distllm.rag.search import RetrieverConfig
from distllm.utils import BaseConfig

# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------
class PromptTemplate:
    """Base class for prompt templates."""

    def preprocess(
        self,
        texts: list[str],
        contexts: list[list[str]],
        scores: list[list[float]],
    ) -> list[str]:
        """Preprocess the texts before sending to the model."""
        raise NotImplementedError('Subclasses should implement this method')


class ConversationPromptTemplate(PromptTemplate):
    """Conversation prompt template for RAG.

    Includes the entire conversation history plus the new user question,
    and optionally the retrieved context.
    """

    def __init__(self, conversation_history: list[tuple[str, str]]):
        # conversation_history is a list of (role, text)
        self.conversation_history = conversation_history

    def preprocess(
        self,
        texts: list[str],
        contexts: list[list[str]] | None = None,
        scores: list[list[float]] | None = None,
    ) -> list[str]:
        """
        Preprocess the texts before sending to the model.

        We assume `texts` has exactly one element: the latest user query.
        We build a single string that contains the entire conversation plus
        the new question. If any retrieval contexts are found, we append them.
        """
        if not texts:
            return ['']  # No user input, return empty prompt.

        # The latest user query:
        user_input = texts[0]

        # Build the conversation string
        conversation_str = ''
        for speaker, text in self.conversation_history:
            conversation_str += f'{speaker}: {text}\n'
        # Add the new user question
        conversation_str += f'User: {user_input}\nAssistant:'

        # Optionally, append retrieved context if it exists
        if contexts and len(contexts) > 0 and len(contexts[0]) > 0:
            # contexts[0] is the top-k retrieval results for this query
            conversation_str += '\n\n[Context from retrieval]\n'
            for doc in contexts[0]:
                conversation_str += f'{doc}\n'

        return [conversation_str]


# -----------------------------------------------------------------------------
# RAG Generator
# -----------------------------------------------------------------------------
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
        description='Freeze off the temperature to the keep model grounded.',
    )
    max_tokens: int = Field(
        16384,
        description='The maximum number of tokens to generate.',
    )

    def get_generator(self) -> VLLMGenerator:
        """Get the vLLM generator."""
        generator = VLLMGenerator(
            config=self,
        )
        return generator


class VLLMGenerator:
    """A generator that calls a local or remote vLLM server."""

    def __init__(self, config: VLLMGeneratorConfig) -> None:
        self.server = config.server
        self.port = config.port
        self.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a prompt to the local vLLM server and return the completion."""
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        url = f'http://{self.server}.cels.anl.gov:{self.port}/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': temp_to_use,
            'max_tokens': tokens_to_use,
        }

        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
        )
        if response.status_code == 200:  # noqa: PLR2004
            result = response.json()['choices'][0]['message']['content']
        else:
            print(f'Error: {response.status_code}')
            result = response.text

        return result


class ArgoGeneratorConfig(BaseConfig):
    """Configuration for the Argo generator using OpenAI client."""

    model: str = Field(
        default_factory=lambda: os.getenv('MODEL', 'argo:gpt-4o'),
        description='The model name for Argo proxy.',
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            'BASE_URL',
            'http://localhost:56267',
        ),
        description='The base URL for the Argo proxy server.',
    )
    api_key: str = Field(
        'whatever+random',
        description='The API key for Argo proxy (can be any string).',
    )
    temperature: float = Field(
        0.0,
        description='Freeze off the temperature to keep model grounded.',
    )
    max_tokens: int = Field(
        16384,
        description='The maximum number of tokens to generate.',
    )

    def get_generator(self) -> ArgoGenerator:
        """Get the Argo generator."""
        generator = ArgoGenerator(
            config=self,
        )
        return generator


class ArgoGenerator:
    """A generator that calls the Argo proxy using OpenAI client."""

    def __init__(self, config: ArgoGeneratorConfig) -> None:
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        # Initialize OpenAI client with Argo proxy settings
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=f'{config.base_url}/v1',
        )

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a prompt to the Argo proxy and return the completion."""
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp_to_use,
                max_tokens=tokens_to_use,
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f'Error calling Argo proxy: {e}')
            result = f'Error: {e!s}'

        return result


class RagGenerator:
    """RAG generator for generating responses to queries."""

    def __init__(
        self,
        generator: VLLMGenerator,
        retriever: Retriever | None = None,
        verbose: bool = False,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.verbose = verbose

    def generate(  # noqa: PLR0913
        self,
        texts: str | list[str],
        prompt_template: PromptTemplate = None,
        retrieval_top_k: int = 5,
        retrieval_score_threshold: float = 0.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Generate responses to the given queries.

        If a retriever is present,
        the retrieved context is appended to the prompt.
        """
        if isinstance(texts, str):
            texts = [texts]  # unify type

        # Use the identity prompt template if none is provided
        if prompt_template is None:
            prompt_template = IdentityPromptTemplate(
                IdentityPromptTemplateConfig(),
            )

        # Default: no context
        contexts, scores = None, None

        # Only retrieve using the new user questions
        if self.retriever is not None:
            results, _ = self.retriever.search(
                texts,  # retrieve on just the latest user query
                top_k=retrieval_top_k,
                score_threshold=retrieval_score_threshold,
            )
            contexts = [
                self.retriever.get_texts(indices)  # top docs for each query
                for indices in results.total_indices
            ]

            scores = results.total_scores

        # Build the final prompts
        prompts = prompt_template.preprocess(texts, contexts, scores)

        # If the verbose is true in config, print contexts.
        if self.verbose:
            print(contexts[0] + '\n\n')

        # We only expect one output per query for now
        # (If multiple texts were passed, we would loop.)
        result = self.generator.generate(
            prompt=prompts[0],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Return as list (matching the function signature)
        return [result]


# -----------------------------------------------------------------------------
# Config Classes
# -----------------------------------------------------------------------------
class RetrievalAugmentedGenerationConfig(BaseConfig):
    """Configuration for the retrieval-augmented generation model."""

    generator_config: VLLMGeneratorConfig | ArgoGeneratorConfig = Field(
        ...,
        description='Settings for the generator (VLLM or Argo)',
    )
    retriever_config: RetrieverConfig | None = Field(
        None,
        description='Settings for the retriever',
    )
    verbose: bool = Field(
        default=False,
        description='Whether to print retrieved contexts in chat.',
    )

    def get_rag_model(self) -> RagGenerator:
        """Instantiate the RAG model."""
        # Initialize the generator (either VLLM or Argo)
        if isinstance(self.generator_config, VLLMGeneratorConfig):
            generator = VLLMGenerator(self.generator_config)
        elif isinstance(self.generator_config, ArgoGeneratorConfig):
            generator = ArgoGenerator(self.generator_config)
        else:
            raise ValueError(
                f'Unsupported generator config type: {type(self.generator_config)}',
            )

        # Initialize the retriever
        retriever = None
        if self.retriever_config is not None:
            retriever = self.retriever_config.get_retriever()

        # Initialize the RAG model
        rag_model = RagGenerator(
            generator=generator,
            retriever=retriever,
            verbose=self.verbose,
        )
        return rag_model


class ChatAppConfig(BaseConfig):
    """Configuration for the evaluation suite."""

    rag_configs: RetrievalAugmentedGenerationConfig = Field(
        ...,
        description='Settings for this RAG application.',
    )
    save_conversation_path: Path = Field(
        ...,
        description='Directory to save the output files.',
    )


# -----------------------------------------------------------------------------
# Main Chat Function
# -----------------------------------------------------------------------------
def chat_with_model(config: ChatAppConfig) -> None:
    """
    Driver function for the chat application.

    Start an interactive chat session:
    1) Keep track of the conversation history.
    2) If user types 'quit', exit the loop.
    3) Upon exit, save the conversation to a local text file with timestamp.
    4) Use only the latest user input for retrieval, but preserve full context
    in the prompt generation so the assistant can handle follow-up queries.
    """
    rag_model = config.rag_configs.get_rag_model()

    # Keep the conversation as list of (role, text)
    conversation_history: list[tuple[str, str]] = []

    while True:
        user_input = input('You: ')

        # Check for 'quit' to exit
        if user_input.strip().lower() == 'quit':
            print('Exiting the chat...')
            break

        # Add the user's turn to the conversation
        conversation_history.append(('User', user_input))

        # We create a custom prompt template that includes
        # the entire conversation so far plus the newly retrieved context.
        conversation_template = ConversationPromptTemplate(
            conversation_history,
        )

        # Ask the RAG model to generate a response
        response_list = rag_model.generate(
            texts=[user_input],  # retrieve only on the new user input
            prompt_template=conversation_template,
            retrieval_top_k=20,
            retrieval_score_threshold=0.1,
        )
        # There's only one element in response_list
        response = response_list[0]

        # Add the model's response to the conversation
        conversation_history.append(('Assistant', response))

        # Print the model's response
        print(
            f'Model: {response} \n --------------------------------------- \n',
        )

    # -------------------------------------------------------------------------
    # Write conversation history to a file with timestamp.
    # -------------------------------------------------------------------------
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(config.save_conversation_path, exist_ok=True)
    filename = (
        f'{config.save_conversation_path}/conversation_{timestamp_str}.txt'
    )
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for speaker, text in conversation_history:
                f.write(f'{speaker}: {text}\n')
        print(f'Conversation saved to {filename}')
    except Exception as e:
        print(f'Error writing conversation to file: {e}')


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    # Load the configuration
    config = ChatAppConfig.from_yaml(args.config)

    # Start the interactive chat
    chat_with_model(config)
