"""Full sequence Embedder."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import Field
from torch.utils.data import DataLoader
from tqdm import tqdm

from distllm.embed.embedders.base import EmbedderResult
from distllm.embed.encoders.base import Encoder
from distllm.embed.poolers.base import Pooler
from distllm.utils import BaseConfig
from flops_profiler.profiler import get_model_profile

def _flops_to_string(flops):
    return str(round(flops / 10.0**12, 2)) + ' TFLOPS'

@torch.no_grad()
def compute_embeddings(
    dataloader: DataLoader,
    encoder: Encoder,
    pooler: Pooler,
    normalize: bool = False,
) -> np.ndarray:
    """Compute pooled hidden embeddings.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to use for batching the data.
    encoder : Encoder
        The encoder to use for inference.
    pooler : Pooler
        The pooler to use for pooling the embeddings.
    normalize : bool, optional
        Whether to normalize the embeddings, by default False.

    Returns
    -------
    np.ndarray
        A numpy array of pooled hidden embeddings.
    """
    # Get the number of embeddings and the embedding size
    num_embeddings = len(dataloader.dataset)

    # Initialize a torch tensor for storing embeddings in host memory
    all_embeddings = torch.empty(
        (num_embeddings, encoder.embedding_size),
        dtype=encoder.dtype,
    )

    # Index for storing embeddings
    idx = 0
    step_profile = True
    if step_profile:
        avg_tflops = 0
        peak_flops = 0
        total_iters = 0
    for batch in tqdm(dataloader):
        # Move the batch to the model device
        inputs = batch.to(encoder.device)
        
        if step_profile:
            print(f"--------- BS: {inputs.attention_mask.shape[0]} ----- idx = {idx}")
            flops, latency, tflops = get_model_profile(model=encoder.model, # model
                        #input_shape=inputs, 
                        #args=None, # list of positional arguments to the model.
                        kwargs=dict(inputs), # dictionary of keyword arguments to the model.
                        print_profile=True, # prints the model graph
                        detailed=False, # print the detailed profile
                        #module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                        #top_modules=1, # the number of top modules to print aggregated profile
                        #warm_up=10, # the number of warm-ups before measuring the time of each module
                        as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                        #output_file=None, # path to the output file. If None, the profiler prints to stdout.
                        #ignore_modules=None, # the list of modules to ignore in the profiling
                        func_name='forward') # the function name to profile, "forward" by default
            print(f"*********:flops:{flops} latency:{latency} tflops:{tflops}")
            total_iters = total_iters + 1
            avg_tflops = avg_tflops + tflops
            if peak_flops < tflops:
                peak_flops = tflops
    
        # Get the model outputs with a forward pass
        embeddings = encoder.encode(inputs)

        # Compute the pooled embeddings
        pooled_embeds = pooler.pool(embeddings, inputs.attention_mask)

        # Normalize the embeddings
        if normalize:
            pooled_embeds = F.normalize(pooled_embeds, p=2, dim=-1)

        # Get the batch size
        batch_size = inputs.attention_mask.shape[0]

        # Store the pooled embeddings in the output buffer
        all_embeddings[idx : idx + batch_size, :] = pooled_embeds.cpu()

        # Increment the output buffer index by the batch size
        idx += batch_size
    
    if step_profile:
        print(f"average tflops={_flops_to_string(avg_tflops / total_iters)} peak_tflops={_flops_to_string(peak_flops)}")
    return all_embeddings.numpy()


class FullSequenceEmbedderConfig(BaseConfig):
    """Configuration for the full sequence embedder."""

    name: Literal['full_sequence'] = 'full_sequence'  # type: ignore[assignment]
    normalize_embeddings: bool = Field(
        False,
        description='Whether to return normalized the embeddings.',
    )


class FullSequenceEmbedder:
    """Embedder for full sequence embeddings."""

    def __init__(self, config: FullSequenceEmbedderConfig) -> None:
        """Initialize the embedder with the configuration."""
        self.config = config

    def embed(
        self,
        dataloader: DataLoader,
        encoder: Encoder,
        pooler: Pooler,
    ) -> EmbedderResult:
        """Embed the sequences.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader to use for batching the data.
        encoder : Encoder
            The encoder to use for inference.
        pooler : Pooler
            The pooler to use for pooling the embeddings.

        Returns
        -------
        EmbedderResult
            Dataclass with the embeddings, text, and optional metadata.
        """
        print("compute embeddings for full sequence")
        embeddings = compute_embeddings(
            dataloader=dataloader,
            encoder=encoder,
            pooler=pooler,
            normalize=self.config.normalize_embeddings,
        )

        # Return the result
        return EmbedderResult(
            embeddings=embeddings,
            text=dataloader.dataset.data,
            metadata=dataloader.dataset.metadata,
        )
