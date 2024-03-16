"""Interface for all language model generators to follow."""

from __future__ import annotations

from typing import Protocol

from distllm.utils import BaseConfig


class LLMResult(BaseConfig):
    """Configuration for the LLMResult."""

    # The prompt text
    prompt: str
    # The response text
    response: str


class LLMGenerator(Protocol):
    """Generator protocol for all generators to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the generator with the configuration."""
        ...

    @property
    def unique_id(self) -> str:
        """Get the unique identifier of the generator."""
        ...

    def generate(self, prompts: str | list[str]) -> list[LLMResult]:
        """Generate response text from prompts."""
        ...
