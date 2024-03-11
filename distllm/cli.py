"""CLI for distllm."""

from __future__ import annotations

from pathlib import Path

import typer
from tqdm import tqdm

app = typer.Typer()


@app.command()
def embed(  # noqa: PLR0913
    model_name: str = typer.Option(
        ...,
        '--model_name',
        '-mn',
        help='The name of the model architecture to use for '
        ' generating the embeddings [auto, esm2].',
    ),
    pretrained_model_name_or_path: str = typer.Option(
        ...,
        '--pretrained_model_name_or_path',
        '-m',
        help='The model weights to use for generating the embeddings.',
    ),
    data_path: Path = typer.Option(  # noqa: B008
        ...,
        '--data_path',
        '-d',
        help='The directory to the data files to embed.',
    ),
    data_extension: str = typer.Option(
        ...,
        '--data_extension',
        '-de',
        help='The extension of the data files to glob.',
    ),
    output_path: Path = typer.Option(  # noqa: B008
        ...,
        '--output_path',
        '-o',
        help='The directory to save the embeddings to (will use the '
        'same file name as data_path).',
    ),
    dataset_name: str = typer.Option(
        'jsonl',
        '--dataset_name',
        '-dn',
        help='The name of the dataset to use for generating the embeddings '
        '[jsonl, fasta, sequence_per_line].',
    ),
    batch_size: int = typer.Option(
        1,
        '--batch_size',
        '-b',
        help='The batch size to use for generating the embeddings.',
    ),
    pooler_name: str = typer.Option(
        'mean',
        '--pooler_name',
        '-pn',
        help='The name of the pooler to use for generating the embeddings '
        '[mean, last_token].',
    ),
    half_precision: bool = typer.Option(
        False,
        '--half_precision',
        '-hp',
        help='Use half precision for the model.',
    ),
    eval_mode: bool = typer.Option(
        False,
        '--eval_mode',
        '-em',
        help='Set the model to evaluation mode.',
    ),
    compile_model: bool = typer.Option(
        False,
        '--compile_model',
        '-cm',
        help='Compile the model for faster inference.',
    ),
    quantization: bool = typer.Option(
        False,
        '--quantization',
        '-q',
        help='Quantize the model for faster inference.',
    ),
) -> None:
    """Generate embeddings for a single file."""
    from distllm.distributed_embedding import embed_and_save_file

    # The dataset kwargs
    dataset_kwargs = {
        # The name of the dataset to use
        'name': dataset_name,
        # The batch size to use for generating the embeddings
        'batch_size': batch_size,
    }

    # The embedder kwargs
    embedder_kwargs = {
        # The name of the model architecture to use
        'name': model_name,
        # The model id to use for generating the embeddings
        'pretrained_model_name_or_path': pretrained_model_name_or_path,
        # Use the model in half precision
        'half_precision': half_precision,
        # Set the model to evaluation mode
        'eval_mode': eval_mode,
        # Compile the model for faster inference
        # Note: This can actually slow down the inference
        # if the number of queries is small
        'compile_model': compile_model,
        # Use quantization
        'quantization': quantization,
    }

    # The pooler kwargs
    pooler_kwargs = {
        # The name of the pooler to use
        'name': pooler_name,
    }

    # Get the data files
    data_files = list(data_path.glob(f'*.{data_extension}'))
    if not data_files:
        raise ValueError(
            f'No files found in {data_path} with extension {data_extension}',
        )

    # Embed and save the files
    for data_file in tqdm(data_files):
        embed_and_save_file(
            file=data_file,
            output_dir=output_path,
            dataset_kwargs=dataset_kwargs,
            embedder_kwargs=embedder_kwargs,
            pooler_kwargs=pooler_kwargs,
        )


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
