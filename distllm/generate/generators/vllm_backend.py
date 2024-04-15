"""Module for the vllm backend LLMGenerator."""

from __future__ import annotations

from typing import Literal
from flops_profiler.profiler import get_model_profile
from distllm.utils import BaseConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the LLMGenerator."""

    name: Literal['vllm'] = 'vllm'  # type: ignore[assignment]
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
        print(f"LLM name: {config.llm_name}")
        self.llm = LLM(
            model=config.llm_name,
            trust_remote_code=config.trust_remote_code,
            dtype='bfloat16',
        )

        self.h_llm_model = AutoModelForCausalLM.from_pretrained(config.llm_name)

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
        step_profile = True 
        if step_profile:
            self.h_llm_model = self.h_llm_model.to('cuda')
            flops, latency, tflops = get_model_profile(model=self.h_llm_model, # model
                    kwargs=dict(prompts, self.sampling_params), # dictionary of keyword arguments to the model.
                    print_profile=True, # prints the model graph
                    detailed=False, # print the detailed profile
                    as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                    func_name='generate') # the function name to profile, "forward" by default
            print(f"*********:flops:{flops} latency:{latency} tflops:{tflops}")
            total_iters = total_iters + 1

        # Generate responses from the prompts. The output is a list of
        # RequestOutput objects that contain the prompt, generated text,
        # and other information.
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Extract the response from the outputs
        responses = [output.outputs[0].text for output in outputs]

        return responses
