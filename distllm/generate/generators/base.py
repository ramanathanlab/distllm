"""Interface for all language model generators to follow."""

from __future__ import annotations

from typing import Protocol

from distllm.utils import BaseConfig


class LLMGenerator(Protocol):
    """Generator protocol for all generators to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the generator with the configuration."""
        ...

    def generate(self, prompts: str | list[str]) -> list[str]:
        """Generate response text from prompts.

        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        ...
