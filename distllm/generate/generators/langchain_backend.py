"""Langchain backend for generating text."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field

from distllm.utils import BaseConfig


class LangChainGeneratorConfig(BaseConfig):
    """Configuration for the LangChainGenerator."""

    name: Literal['langchain'] = 'langchain'  # type: ignore[assignment]
    llm: str = Field(
        'gpt-3.5-turbo',
        description='The model type to use for the generator.',
    )
    temperature: int = Field(
        0,
        description='The temperature parameter for the generator.',
    )
    verbose: bool = Field(
        True,
        description='Whether to print verbose output.',
    )
    dotenv_path: Path = Field(
        default=Path('~/.env'),
        description='Path to the .env file. Contains API keys: '
        'OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY',
    )


class LangChainGenerator:
    """Create simple language chains for inference."""

    def __init__(self, config: LangChainGeneratorConfig) -> None:
        """Initialize the LangChainGenerator."""
        from langchain.chains import LLMChain
        from langchain.chat_models import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import GoogleGenerativeAI

        # Load environment variables from .env file containing
        # API keys for the language models
        load_dotenv(config.dotenv_path)

        # Define the possible chat models
        chat_models = {
            'gpt-3.5-turbo': ChatOpenAI,
            'gemini-pro': GoogleGenerativeAI,
            'claude-3-opus-20240229': ChatAnthropic,
        }

        # Get the chat model based on the configuration
        chat_model = chat_models.get(config.llm)
        if not chat_model:
            raise ValueError(f'Invalid chat model: {config.llm}')

        # Initialize the language model
        llm = chat_model(model=config.llm, verbose=config.verbose)

        # Create the prompt template (input only)
        prompt = ChatPromptTemplate.from_template('{input}')

        # Initialize the chain
        self.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=config.verbose,
        )

    def generate(self, prompts: str | list[str]) -> list[str]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : str | list[str]
            The prompts to generate text from.

        Returns
        -------
        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Format the inputs for the chain
        inputs = [{'input': prompt} for prompt in prompts]

        # Generate the outputs
        raw_outputs = self.chain.batch(inputs)

        # Extract the text from the outputs
        outputs = [output['text'] for output in raw_outputs]
        return outputs
