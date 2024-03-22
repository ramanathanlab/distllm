"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

import os
from typing import Literal

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
    temperature: float = 0.5
    # Min p for sampling
    min_p: float = 0.1
    # Top p for sampling (off by default)
    top_p: float = 0.0
    # Max tokens to generate
    max_tokens: int = 2000
    # Whether to use beam search
    use_beam_search: bool = False


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
        sampling_kwargs = {}
        if config.top_p:
            sampling_kwargs['top_p'] = config.top_p
        else:
            sampling_kwargs['min_p'] = config.min_p

        # Create the sampling params to use
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            use_beam_search=config.use_beam_search,
            **sampling_kwargs,
        )

        # Create an LLM instance
        self.llm = LLM(
            model=config.llm_name,
            trust_remote_code=config.trust_remote_code,
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

        # Generate responses from the prompts. The output is a list of
        # RequestOutput objects that contain the prompt, generated text,
        # and other information.
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Extract the response from the outputs
        responses = [output.outputs[0].text for output in outputs]

        return responses
