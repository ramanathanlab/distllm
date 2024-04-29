"""Module for the hugging face generator backend."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from distllm.utils import BaseConfig
from distllm.utils import batch_data


class HuggingFaceGeneratorConfig(BaseConfig):
    """Configuration for the HuggingFaceGenerator."""

    name: Literal['huggingface'] = 'huggingface'  # type: ignore[assignment]
    pretrained_model_name_or_path: str = Field(
        ...,
        description='The model id or path to the pretrained model.',
    )
    half_precision: bool = Field(
        False,
        description='Whether to use half precision.',
    )
    eval_mode: bool = Field(
        True,
        description='Whether to set the model to evaluation mode.',
    )
    compile_model: bool = Field(
        False,
        description='Whether to compile the model for faster inference.',
    )
    quantization: bool = Field(
        True,
        description='Whether to use quantization.',
    )
    top_p: float = Field(
        0.95,
        description='The top p for sampling.',
    )
    num_beams: int = Field(
        10,
        description='The number of beams for sampling.',
    )
    do_sample: bool = Field(
        True,
        description='Whether to use sampling.',
    )
    batch_size: int = Field(
        1,
        description='The number of prompts to process at once.',
    )


class HuggingFaceGenerator:
    """Language model generator using hugging face backend."""

    def __init__(self, config: HuggingFaceGeneratorConfig) -> None:
        """Initialize the HuggingFaceGenerator."""
        import torch
        from transformers import AutoModel
        from transformers import AutoTokenizer

        model_kwargs = {}

        # Use quantization
        if config.quantization:
            from transformers import BitsAndBytesConfig

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model_kwargs['quantization_config'] = nf4_config

        # Load model and tokenizer
        model = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

        # Set the model max length for proper truncation
        tokenizer.model_max_length = model.config.max_position_embeddings

        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load the model onto the device
        if not config.quantization:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu',
            )
            model.to(device)

        # Compile the model for faster inference
        if config.compile_model:
            model = torch.compile(model, fullgraph=True)

        # Set persistent attributes
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _generate_batch(self, prompts: str | list[str]) -> list[str]:
        """Generate text from a batch encoding.

        Parameters
        ----------
        prompts : str | list[str]
            The prompts to generate text from.

        Returns
        -------
        list[str]
            A list of generated responses.
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(
            prompts,
            padding='longest',
            return_tensors='pt',
        )

        # Move the batch_encoding to the device
        batch_encoding = batch_encoding.to(self.model.device)

        # Generate text using top-p sampling
        generated_text = self.model.generate(
            **batch_encoding,
            top_p=self.config.top_p,
            num_return_sequences=1,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
        )

        # Decode the generated text
        responses = self.tokenizer.batch_decode(
            generated_text,
            skip_special_tokens=True,
        )

        return responses

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

        # Generate responses from the prompts
        responses = []
        for batch in batch_data(prompts, self.config.batch_size):
            responses.extend(self._generate_batch(batch))

        return responses
