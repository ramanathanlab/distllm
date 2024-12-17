"""Perform protein search using FAISS."""

from __future__ import annotations

import argparse
from pathlib import Path

from distllm.embed.datasets.fasta import read_fasta
from distllm.embed.encoders import Esm2EncoderConfig
from distllm.embed.encoders import EsmCambrianEncoderConfig
from distllm.embed.poolers import MeanPoolerConfig
from distllm.rag.search import FaissIndexV2Config
from distllm.rag.search import RetrieverConfig

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform semantic search using FAISS.',
    )
    parser.add_argument(
        '--fasta_path',
        type=Path,
        help='The path to the FASTA file containing query protein sequences.',
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        help='The path to the HF dataset directory containing the document '
        'text and fp32 embeddings.',
    )
    parser.add_argument(
        '--dataset_chunk_dir',
        type=Path,
        default=None,
        help='The path to a directory containing dataset chunks, each '
        'containing an HF dataset with the document text and fp32 embeddings.',
    )
    parser.add_argument(
        '--faiss_index_path',
        type=str,
        help='The path to the FAISS index.',
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='facebook/esm2_t33_650M_UR50D',
        help='The model name or path to the LLM.',
    )
    parser.add_argument(
        '--precision',
        default='ubinary',
        type=str,
        help='The desired precision for the embeddings'
        ' [float32, uint8, ubinary].',
    )
    parser.add_argument(
        '--search_algorithm',
        default='exact',
        type=str,
        help='The FAISS search algorithm [exact, hnsw].',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help='The number of nearest neighbors to retrieve.',
    )
    parser.add_argument(
        '--rescore_multiplier',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--num_quantization_workers',
        type=int,
        default=1,
        help='The number of quantization workers to use.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='The batch size for the embedder model.',
    )
    parser.add_argument(
        '--use_faesm',
        action='store_true',
        help='Whether to use the FAESM model.',
    )
    args = parser.parse_args()

    # Get the paths to the dataset chunks if they exist
    dataset_chunk_paths = None
    if args.dataset_chunk_dir is not None:
        dataset_chunk_paths = list(args.dataset_chunk_dir.glob('*'))

    # Configure the FAISS index
    faiss_config = FaissIndexV2Config(
        dataset_dir=args.dataset_dir,
        faiss_index_path=args.faiss_index_path,
        dataset_chunk_paths=dataset_chunk_paths,
        precision=args.precision,
        search_algorithm=args.search_algorithm,
        rescore_multiplier=args.rescore_multiplier,
        num_quantization_workers=args.num_quantization_workers,
    )

    # Encoding strategy
    if 'esm2' in args.model_name_or_path:
        encoder_config = Esm2EncoderConfig(
            pretrained_model_name_or_path=args.model_name_or_path,
            faesm=args.use_faesm,
        )
    else:
        assert 'esmc' in args.model_name_or_path
        encoder_config = EsmCambrianEncoderConfig(
            pretrained_model_name_or_path=args.model_name_or_path,
        )

    # Pooling strategy
    pooler_config = MeanPoolerConfig()

    # Configure the retriever
    config = RetrieverConfig(
        faiss_config=faiss_config,
        encoder_config=encoder_config,
        pooler_config=pooler_config,
        batch_size=args.batch_size,
    )

    # Build the retriever
    retriever = config.get_retriever()

    # Read the FASTA file
    sequence_data = read_fasta(args.fasta_path)
    query_sequences = [x.sequence for x in sequence_data]

    # Query the retriever
    result, embeddings = retriever.search(query=query_sequences)

    print(embeddings.shape)

    for seq, idx, scores in zip(
        sequence_data,
        result.total_indices,
        result.total_scores,
    ):
        search_tags = retriever.get(idx, key='tags')
        print('Query sequence tag:', seq.tag)
        print('Query sequence:', seq.sequence)
        print('idx:', idx)
        print('scores:', scores)
        print('tags:', search_tags)
        for tag in search_tags:
            print(
                f'Uniprot link: https://www.uniprot.org/uniprotkb/{tag}/entry',
            )
        print('paths:', retriever.get(idx, key='paths'))
        print('search sequence:', retriever.get(idx, key='text'))
        print()
