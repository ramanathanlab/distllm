"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

import os
from typing import Literal
from uuid import uuid4

from distllm.generate.generators.base import LLMResult
from distllm.utils import BaseConfig


class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the LLMGenerator."""

    name: Literal['vllm'] = 'vllm'  # type: ignore[assignment]
    # Path to the hugingface cache (by default it uses the home directory)
    hf_cache_path: str = ''
    # The name of the vllm LLM model, see
    # https://docs.vllm.ai/en/latest/models/supported_models.html
    llm_name: str
    # Whether to trust remote code
    trust_remote_code: bool = True
    # Temperature for sampling
    temperature: float = 0.7
    # Top p for sampling
    top_p: float = 0.95


class VLLMGenerator:
    """Language model generator using vllm backend."""

    def __init__(self, config: VLLMGeneratorConfig) -> None:
        """Initialize the LLMGenerator.

        Parameters
        ----------
        config : vLLMGeneratorConfig
            The configuration for the LLMGenerator.
        """
        from vllm import LLM
        from vllm import SamplingParams

        # Specify the huggingface cache path
        if config.hf_cache_path:
            os.environ['HF_HOME'] = config.hf_cache_path

        # Create the sampling params to use
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
        )

        # Create an LLM instance
        self.llm = LLM(
            model=config.llm_name,
            trust_remote_code=config.trust_remote_code,
        )

        # Set the unique id
        self.unique_id = str(uuid4())

    def generate(self, prompts: str | list[str]) -> list[LLMResult]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : str | list[str]
            The prompts to generate text from.

        Returns
        -------
        list[LLMResult]
            A list of LLMResult with the prompt and response. For example:
            [
                LLMResult(prompt='What is two plus two?', response='four'),
                ...
            ]
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Generate texts from the prompts. The output is a list of
        # RequestOutput objects that contain the prompt, generated text,
        # and other information.
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Extract the prompt and response from the outputs
        results = [
            LLMResult(prompt=output.prompt, response=output.outputs[0].text)
            for output in outputs
        ]

        return results
