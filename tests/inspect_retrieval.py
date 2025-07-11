#!/usr/bin/env python3
"""
Standalone script to inspect retrieval results from HF dataset.

This script demonstrates how to examine the exact content and attributes
of documents retrieved from your HF dataset during RAG.
"""
from __future__ import annotations

from pathlib import Path

from distllm.chat import ChatAppConfig
from distllm.chat import inspect_retrieval_results
from distllm.chat import print_retrieval_inspection


def main():
    """Main function to demonstrate retrieval inspection."""
    # You'll need to provide your actual config file path
    config_path = Path(
        'your_config.yaml',
    )  # Replace with your actual config path

    if not config_path.exists():
        print(f'❌ Config file not found at {config_path}')
        print(
            'Please update the config_path variable with your actual config file path.',
        )
        return

    # Load the configuration
    config = ChatAppConfig.from_yaml(config_path)

    # Initialize just the retriever (no need for the full RAG model)
    retriever = config.rag_configs.retriever_config.get_retriever()

    # Example queries to test retrieval
    test_queries = [
        'gene expression regulation',
        'flux balance analysis',
        'protein folding mechanisms',
        'cell metabolism pathways',
    ]

    print('🔍 Testing retrieval inspection with different queries...\n')

    for query in test_queries:
        print(f"Testing query: '{query}'")

        # Inspect retrieval results
        results = inspect_retrieval_results(
            retriever=retriever,
            query=query,
            top_k=3,  # Just get top 3 for this demo
            score_threshold=0.1,
        )

        # Print the detailed inspection
        print_retrieval_inspection(results)

        # Show programmatic access to the results
        print('📊 Programmatic access example:')
        print(f'  - Query: {results["query"]}')
        print(f'  - Available columns: {results["dataset_columns"]}')
        print(f'  - Number of results: {results["num_results"]}')

        if results['retrieved_documents']:
            doc = results['retrieved_documents'][0]  # First document
            print(f'  - Top result index: {doc["dataset_index"]}')
            print(f'  - Top result score: {doc["score"]:.4f}')
            print(f'  - Top result path: {doc["attributes"]["path"]}')
            print(
                f'  - Top result text preview: {doc["attributes"]["text"][:100]}...',
            )

        print('\n' + '=' * 50 + '\n')


if __name__ == '__main__':
    main()
