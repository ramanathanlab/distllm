"""Embedder for the auto model."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig


class AutoEmbedderConfig(BaseConfig):
    """Config for the transformers AutoModel embedder."""

    # The name of the embedder
    name: Literal['auto'] = 'auto'  # type: ignore[assignment]
    # The model id
    pretrained_model_name_or_path: str
    # Use the model in half precision
    half_precision: bool = True
    # Set the model to evaluation mode
    eval_mode: bool = True
    # Compile the model for faster inference
    compile_model: bool = True
    # Use quantization
    quantization: bool = True


class AutoEmbedder:
    """Embedder for the transformers AutoModel."""

    def __init__(self, config: AutoEmbedderConfig):
        """Initialize the embedder."""
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
        self._tokenizer = tokenizer

        # Set the maximum sequence length
        self.model.max_length = self.model.config.max_position_embeddings

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the embedder."""
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the embedder."""
        return self.model.device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the embedder."""
        return self.model.config.hidden_size

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the embedder."""
        return self._tokenizer

    def embed(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Embed the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence (containing the input_ids,
            attention_mask, and token_type_ids).

        Returns
        -------
        torch.Tensor
            The embeddings of the sequence extracted from the last hidden state
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        # Get the model outputs with a forward pass
        outputs = self.model(**batch_encoding, output_hidden_states=True)

        # Get the last hidden states
        return outputs.hidden_states[-1]
