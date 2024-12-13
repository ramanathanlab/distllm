"""CLI for distllm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def embed(  # noqa: PLR0913
    encoder_name: str = typer.Option(
        ...,
        '--encoder_name',
        '-mn',
        help='The name of the encoder architecture to use for '
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
        '[jsonl, jsonl_chunk, fasta, sequence_per_line].',
    ),
    batch_size: int = typer.Option(
        1,
        '--batch_size',
        '-b',
        help='The batch size to use for generating the embeddings.',
    ),
    chunk_batch_size: int = typer.Option(
        1,
        '--chunk_batch_size',
        '-cb',
        help='The batch size to use for chunked text within semantic '
        'chunking.',
    ),
    buffer_size: int = typer.Option(
        1,
        '--buffer_size',
        '-bs',
        help='The buffer size to use for semantic chunking.',
    ),
    pooler_name: str = typer.Option(
        'mean',
        '--pooler_name',
        '-pn',
        help='The name of the pooler to use for generating the embeddings '
        '[mean, last_token].',
    ),
    embedder_name: str = typer.Option(
        'full_sequence',
        '--embedder_name',
        '-en',
        help='The name of the embedder to use for generating the embeddings '
        '[full_sequence, semantic_chunk].',
    ),
    writer_name: str = typer.Option(
        'huggingface',
        '--writer_name',
        '-wn',
        help='The name of the writer to use for saving the embeddings '
        '[huggingface, numpy].',
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
    from distllm.distributed_embedding import embedding_worker

    # The dataset kwargs
    dataset_kwargs = {
        # The name of the dataset to use
        'name': dataset_name,
        # The batch size to use for generating the embeddings
        'batch_size': batch_size,
    }

    # If the dataset is jsonl_chunk, set the buffer size
    if dataset_name == 'jsonl_chunk':
        dataset_kwargs['buffer_size'] = buffer_size

    # The encoder kwargs
    encoder_kwargs = {
        # The name of the model architecture to use
        'name': encoder_name,
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

    # The embedder kwargs
    embedder_kwargs: dict[str, Any] = {
        # The name of the embedder to use
        'name': embedder_name,
    }

    if embedder_name == 'semantic_chunk':
        # Set the batch size to use for chunked text within semantic chunking
        embedder_kwargs['chunk_batch_size'] = chunk_batch_size

    # The writer kwargs
    writer_kwargs = {
        # The name of the writer to use
        'name': writer_name,
    }

    # Get the data files
    data_files = list(data_path.glob(f'*.{data_extension}'))
    if not data_files:
        raise ValueError(
            f'No files found in {data_path} with extension {data_extension}',
        )

    # Embed and save the files
    for data_file in tqdm(data_files):
        embedding_worker(
            input_path=data_file,
            output_dir=output_path,
            dataset_kwargs=dataset_kwargs,
            encoder_kwargs=encoder_kwargs,
            pooler_kwargs=pooler_kwargs,
            embedder_kwargs=embedder_kwargs,
            writer_kwargs=writer_kwargs,
        )


@app.command()
def merge(
    writer_name: str = typer.Option(
        'huggingface',
        '--writer_name',
        '-wn',
        help='The name of the writer to use for saving datasets '
        '[huggingface, numpy].',
    ),
    num_proc: int = typer.Option(
        None,
        '--num_proc',
        '-np',
        help='The number of processes to use for merging the datasets. '
        'Only works with huggingface writer.',
    ),
    dataset_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--dataset_dir',
        '-d',
        help='The directory containing the dataset subdirectories '
        'to merge (will glob * this directory).',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The dataset directory to save the merged datasets to.',
    ),
) -> None:
    """Merge datasets from multiple directories output by `generate`."""
    from distllm.generate import get_writer

    # The writer kwargs
    writer_kwargs: dict[str, Any] = {
        # The name of the writer to use
        'name': writer_name,
    }

    # If the writer is huggingface, set the number of processes
    if writer_name == 'huggingface':
        writer_kwargs['num_proc'] = num_proc

    # Initialize the writer
    writer = get_writer(writer_kwargs)

    # Get the dataset directories
    dataset_dirs = list(dataset_dir.glob('*'))

    # Merge the datasets
    writer.merge(dataset_dirs, output_dir)


@app.command()
def generate(  # noqa: PLR0913
    input_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--input_dir',
        '-i',
        help='The directory containing the input sub-files/directories '
        'containing inputs for generation (will glob * this directory).',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The dataset directory to save the merged datasets to.',
    ),
    prompt_name: str = typer.Option(
        'question_chunk',
        '--prompt_name',
        '-pn',
        help='The name of the prompt to use for generating the text '
        '[question_chunk].',
    ),
    reader_name: str = typer.Option(
        'huggingface',
        '--reader_name',
        '-rn',
        help='The name of the reader to use for reading the input files '
        '[huggingface].',
    ),
    writer_name: str = typer.Option(
        'huggingface',
        '--writer_name',
        '-wn',
        help='The name of the writer to use for saving datasets '
        '[huggingface].',
    ),
    generator_name: str = typer.Option(
        'vllm',
        '--generator_name',
        '-gn',
        help='The name of the generator to use for generating the text '
        '[vllm, huggingface].',
    ),
    llm_name: str = typer.Option(
        'mistralai/Mistral-7B-Instruct-v0.2',
        '--llm_name',
        '-vmn',
        help='The name of the VLLM model to use for generating the text '
        '[mistralai/Mistral-7B-Instruct-v0.2]. '
        'See: https://docs.vllm.ai/en/latest/models/supported_models.html',
    ),
    temperature: float = typer.Option(
        0.5,
        '--temperature',
        '-t',
        help='Temperature for sampling.',
    ),
    min_p: float = typer.Option(
        0.1,
        '--min_p',
        '-mp',
        help='Min p for sampling.',
    ),
    top_p: float = typer.Option(
        0.0,
        '--top_p',
        '-tp',
        help='Top p for sampling (off by default).',
    ),
    max_tokens: int = typer.Option(
        2000,
        '--max_tokens',
        '-mt',
        help='Max tokens to generate.',
    ),
    use_beam_search: bool = typer.Option(
        False,
        '--use_beam_search',
        '-bs',
        help='Whether to use beam search.',
    ),
    batch_size: int = typer.Option(
        1,
        '--batch_size',
        '-b',
        help='The batch size to use for generating the text'
        ' (for huggingface).',
    ),
    quantization: bool = typer.Option(
        False,
        '--quantization',
        '-q',
        help='Quantize the model for faster inference.',
    ),
) -> None:
    """Merge datasets from multiple directories output by `embed` command."""
    from distllm.distributed_generation import generate_worker

    # The prompt kwargs
    prompt_kwargs: dict[str, Any] = {
        # The name of the prompt to use
        'name': prompt_name,
    }

    # The reader kwargs
    reader_kwargs: dict[str, Any] = {
        # The name of the reader to use
        'name': reader_name,
    }

    # The writer kwargs
    writer_kwargs: dict[str, Any] = {
        # The name of the writer to use
        'name': writer_name,
    }

    # The generator kwargs
    generator_kwargs: dict[str, Any] = {
        # The name of the generator to use
        'name': generator_name,
    }

    # vllm backend specific kwargs
    if generator_name == 'vllm':
        # The name of the VLLM model to use
        generator_kwargs['llm_name'] = llm_name
        # Temperature for sampling
        generator_kwargs['temperature'] = temperature
        # Min p for sampling
        generator_kwargs['min_p'] = min_p
        # Top p for sampling (off by default)
        generator_kwargs['top_p'] = top_p
        # Max tokens to generate
        generator_kwargs['max_tokens'] = max_tokens
        # Whether to use beam search
        generator_kwargs['use_beam_search'] = use_beam_search

    # huggingface backend specific kwargs
    elif generator_name == 'huggingface':
        # The name of the HuggingFace model to use
        generator_kwargs['pretrained_model_name_or_path'] = llm_name
        # Top p for sampling
        generator_kwargs['top_p'] = top_p
        # The batch size to use for generating the text
        generator_kwargs['batch_size'] = batch_size
        # Use quantization
        generator_kwargs['quantization'] = quantization

    # Get the dataset directories
    input_paths = list(input_dir.glob('*'))

    for input_path in tqdm(input_paths):
        generate_worker(
            input_path=input_path,
            output_dir=output_dir,
            prompt_kwargs=prompt_kwargs,
            reader_kwargs=reader_kwargs,
            writer_kwargs=writer_kwargs,
            generator_kwargs=generator_kwargs,
        )


@app.command()
def tokenize(  # noqa: PLR0913
    input_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--input_dir',
        '-i',
        help='The directory containing the input sub-files/directories '
        'containing inputs for tokenization (will glob * this directory).',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The directory to save the tokenized text to.',
    ),
    text_field: str = typer.Option(
        'text',
        '--text_field',
        '-tf',
        help='The name of the text field in the jsonl file.',
    ),
    tokenizer_name: str = typer.Option(
        'meta-llama/Llama-2-70b-chat-hf',
        '--tokenizer_name',
        '-tn',
        help='The name of the tokenizer to use.',
    ),
    dotenv_path: Path = typer.Option(  # noqa: B008
        Path('~/.env'),
        '--dotenv_path',
        '-dp',
        help='Path to the .env file with HF_TOKEN for huggingface hub.',
    ),
    save_labels: bool = typer.Option(
        False,
        '--save_labels',
        '-sl',
        help='Whether to store a separate labels field in the dataset.',
    ),
) -> None:
    """Tokenize a directory of jsonl files and save the datasets to disk."""
    from distllm.distributed_tokenization import tokenizer_worker

    # The tokenizer kwargs
    tokenizer_kwargs: dict[str, Any] = {
        # The name of the text field in the jsonl file
        'text_field': text_field,
        # The name of the tokenizer to use
        'tokenizer_name': tokenizer_name,
        # Path to the .env file
        'dotenv_path': dotenv_path,
        # Whether to save labels
        'save_labels': save_labels,
    }

    # Get the dataset directories
    input_paths = list(input_dir.glob('*'))

    for input_path in tqdm(input_paths):
        tokenizer_worker(
            input_path=input_path,
            output_dir=output_dir,
            tokenizer_kwargs=tokenizer_kwargs,
        )


@app.command()
def chunk_fasta_file(
    input_file: Path = typer.Option(  # noqa: B008
        ...,
        '--input_file',
        '-i',
        help='The fasta file to chunk.',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The directory to save the chunked fasta files to.',
    ),
    num_chunks: int = typer.Option(
        ...,
        '--chunk_size',
        '-c',
        help='The number of smaller files to chunk the fasta file into.',
    ),
) -> None:
    """Chunk a fasta file into smaller fasta files."""
    from distllm.embed.datasets.fasta import read_fasta
    from distllm.embed.datasets.fasta import write_fasta
    from distllm.utils import batch_data

    # Read the fasta file
    sequences = read_fasta(input_file)

    # Chunk the sequences
    chunks = batch_data(sequences, len(sequences) // num_chunks)

    # Make the output directory
    output_dir.mkdir(parents=True)

    # Save the chunked fasta files
    for i, chunk in tqdm(enumerate(chunks), desc='Writing chunks'):
        filename = f'{input_file.stem}_{i:04}{input_file.suffix}'
        write_fasta(chunk, output_dir / filename)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
