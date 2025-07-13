#!/usr/bin/env python3

"""
RAG Argonium Advanced Question Grader v2.0 (Parallel) - With Chunk ID Logging and RAG Toggle

NEW in V2:
- Chunk ID logging for all retrieved documents during RAG
- Chunk IDs generated using the same format as mcqa_generation_distllm_v2.py
- Detailed retrieval information stored in results
- Enhanced traceability from questions to retrieved chunks
- RAG toggle functionality (--no-rag) to disable RAG completely
- File-level metadata logging for configuration tracking

Usage:
    python rag_argonium_score_parallel_v2.py <questions_file.json> --model <model_shortname> --grader <grader_shortname> [--config <config_file>] [--rag-config <rag_config_file>] [--parallel <num_workers>] [--format auto|mc|qa] [--random <num_questions>] [--seed <random_seed>] [--save-incorrect] [--use-context-field] [--retrieval-top-k <k>] [--retrieval-score-threshold <threshold>] [--log-chunks] [--no-rag]

Where:
    - questions_file.json: A JSON file with an array of objects, each having "question", "answer", and optionally "text" fields
    - model_shortname: The shortname of the model to test from model_servers.yaml
    - grader_shortname: The shortname of the model to use for grading from model_servers.yaml
    - config_file: Configuration file to use for model settings (default: model_servers.yaml)
    - rag_config_file: RAG configuration file (YAML) for retrieval settings (optional)
    - parallel: Number of concurrent workers for parallel processing (default: 1)
    - format: Format of questions (auto, mc, qa) (default: auto)
    - random: Randomly select N questions from the dataset (optional)
    - seed: Random seed for reproducible question selection (optional, only used with --random)
    - save-incorrect: Save incorrectly answered questions to a separate JSON file (optional)
    - use-context-field: Use the "text" field from JSON as context instead of retrieval (optional, ignored if --no-rag is used)
    - retrieval-top-k: Number of documents to retrieve (default: 5)
    - retrieval-score-threshold: Minimum retrieval score threshold (default: 0.0)
    - log-chunks: Enable detailed chunk logging (always enabled in v2)
    - no-rag: Disable RAG completely - questions are asked directly to the model without any context

Examples:
    # With RAG (default)
    python rag_argonium_score_parallel_v2.py frg_mc_100.json --model llama --grader gpt41 --parallel 4
    python rag_argonium_score_parallel_v2.py frg_mc_100.json --model llama --grader gpt41 --rag-config rag_config.yaml

    # Without RAG - direct question answering
    python rag_argonium_score_parallel_v2.py frg_mc_100.json --model llama --grader gpt41 --no-rag

    # With context field (ignored if --no-rag is used)
    python rag_argonium_score_parallel_v2.py frg_mc_100.json --model llama --grader gpt41 --use-context-field

The script:
1) Can operate in RAG mode (default) or direct question answering mode (--no-rag)
2) In RAG mode: Uses retrieval-augmented generation to enhance question answering
3) In direct mode: Asks questions directly to the model without any context
4) Logs every retrieved chunk with its chunk ID for full traceability (RAG mode only)
5) Supports both VLLM and Argo generators
6) Records configuration metadata in the output file
7) Uses the specified MODEL to generate answers and GRADER to evaluate them
8) Reports detailed accuracy metrics and exports results with full configuration tracking
9) Processes multiple questions in parallel when --parallel > 1
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import backoff
import numpy as np
import openai
import requests
import yaml
from dotenv import load_dotenv
from pydantic import Field
from tqdm import tqdm

from distllm.generate.prompts import (
    IdentityPromptTemplate,
    IdentityPromptTemplateConfig,
    PromptTemplate,
)
from distllm.rag.search import Retriever, RetrieverConfig
from distllm.utils import BaseConfig

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# Chunk ID Generation Functions (from mcqa_generation_distllm_v2.py)
# -----------------------------------------------------------------------------


def generate_chunk_id(dataset_index: int, path: str) -> str:
    """
    Generate a chunk ID from dataset index and file path.
    Format: {file_id}_{chunk_index:04d}
    """
    # Create file_id from path
    file_id = hashlib.sha256(path.encode()).hexdigest()[:16]
    chunk_index = dataset_index
    return f'{file_id}_{chunk_index:04d}'


def reverse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    """
    Reverse engineer a chunk_id to get file_id and chunk_index.

    Args:
        chunk_id: The chunk ID in format {file_id}_{chunk_index:04d}

    Returns:
        Tuple[str, int]: (file_id, chunk_index)
    """
    parts = chunk_id.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f'Invalid chunk_id format: {chunk_id}')

    file_id = parts[0]
    try:
        chunk_index = int(parts[1])
    except ValueError:
        raise ValueError(f'Invalid chunk index in chunk_id: {chunk_id}')

    return file_id, chunk_index


def get_original_path_from_chunk_id(
    chunk_id: str, path_mapping: Dict[str, str]
) -> Optional[str]:
    """
    Get the original file path from a chunk_id using the path mapping.

    Args:
        chunk_id: The chunk ID to look up
        path_mapping: Dictionary mapping file_id to original_path

    Returns:
        Optional[str]: The original file path, or None if not found
    """
    try:
        file_id, _ = reverse_chunk_id(chunk_id)
        return path_mapping.get(file_id)
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# Helper functions (from argonium_score_parallel_v9.py)
# -----------------------------------------------------------------------------

# Global client cache for OpenAI clients (thread-safe)
_client_cache = {}
_client_cache_lock = threading.Lock()


def get_openai_client(api_key, api_base, timeout=120.0):
    """Get or create a cached OpenAI client for the given configuration."""
    cache_key = (api_key, api_base, timeout)

    with _client_cache_lock:
        if cache_key not in _client_cache:
            _client_cache[cache_key] = openai.OpenAI(
                api_key=api_key, base_url=api_base, timeout=timeout
            )
        return _client_cache[cache_key]


def load_model_config(model_shortname, config_file='model_servers.yaml'):
    """Load model configuration from the specified configuration file."""
    if os.path.isabs(config_file):
        yaml_path = config_file
    else:
        yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_file
        )

    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'Model config file not found: {yaml_path}')
    except yaml.YAMLError as e:
        raise ValueError(f'Error parsing YAML file: {e}')

    # Find the model configuration
    model_config = None
    for server in config.get('servers', []):
        if server.get('shortname') == model_shortname:
            model_config = server
            break

    if model_config is None:
        available_models = [
            server.get('shortname') for server in config.get('servers', [])
        ]
        raise ValueError(
            f'Model "{model_shortname}" not found in config. Available models: {available_models}'
        )

    return model_config


def detect_question_format(questions):
    """Detect the format of questions (mc or qa)."""
    mc_count = 0
    qa_count = 0

    for question in questions[: min(10, len(questions))]:  # Sample first 10
        question_text = question.get('question', '')
        # Check for multiple choice patterns
        mc_patterns = [
            r'(?:^|\n)\s*([A-E])[.):]\s',
            r'(?:^|\n)\s*([1-5])[.):]\s',
        ]
        is_mc = any(
            re.search(pattern, question_text, re.IGNORECASE)
            for pattern in mc_patterns
        )
        if is_mc:
            mc_count += 1
        else:
            qa_count += 1

    return 'mc' if mc_count > qa_count else 'qa'


def detect_choice_identifier_type(question_text):
    """Detect if the question uses letters (A-E) or numbers (1-5) for choices."""
    letter_match = re.search(
        r'(?:^|\n)\s*([A-E])[.):]\s', question_text, re.IGNORECASE
    )
    number_match = re.search(r'(?:^|\n)\s*([1-5])[.):]\s', question_text)

    if letter_match:
        return 'letter'
    elif number_match:
        return 'number'
    else:
        return 'unknown'


def extract_choice_identifier(answer_text):
    """Extract the choice identifier (A, B, C, D, E or 1, 2, 3, 4, 5) from the answer text."""
    # Try letter patterns first
    letter_patterns = [
        r'(?:^|\s|answer\s+is\s+|correct\s+answer\s+is\s+|answer:\s*)([A-E])(?:\s|$|[.):,])',
        r'(?:^|\s)([A-E])[.):]\s',
        r'(?:^|\s)option\s+([A-E])(?:\s|$|[.):,])',
        r'(?:^|\s)choice\s+([A-E])(?:\s|$|[.):,])',
        r'(?:^|\s)([A-E])(?:\s|$)',
    ]

    for pattern in letter_patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Try number patterns
    number_patterns = [
        r'(?:^|\s|answer\s+is\s+|correct\s+answer\s+is\s+|answer:\s*)([1-5])(?:\s|$|[.):,])',
        r'(?:^|\s)([1-5])[.):]\s',
        r'(?:^|\s)option\s+([1-5])(?:\s|$|[.):,])',
        r'(?:^|\s)choice\s+([1-5])(?:\s|$|[.):,])',
        r'(?:^|\s)([1-5])(?:\s|$)',
    ]

    for pattern in number_patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


# -----------------------------------------------------------------------------
# Enhanced RAG classes with chunk logging and no-RAG support
# -----------------------------------------------------------------------------


class RAGPromptTemplate:
    """Prompt template for RAG that includes context."""

    def __init__(self, question_format: str = 'mc'):
        self.question_format = question_format

    def preprocess(
        self,
        text: Union[str, List[str]],
        contexts: Optional[List[List[str]]] = None,
        scores: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Preprocess the texts into prompts with context."""
        if isinstance(text, str):
            text = [text]

        if contexts is None:
            return text

        prompts = []
        for i, question in enumerate(text):
            if i < len(contexts) and contexts[i]:
                context_text = '\n\n'.join(contexts[i])
                if self.question_format == 'mc':
                    prompt = f"""Context:
{context_text}

Question:
{question}

Based on the context provided, answer the question by selecting the correct option."""
                else:
                    prompt = f"""Context:
{context_text}

Question:
{question}

Based on the context provided, answer the question."""
            else:
                prompt = question
            prompts.append(prompt)

        return prompts

    def postprocess(self, responses: List[str]) -> List[str]:
        """Postprocess the responses."""
        return responses


class DirectPromptTemplate:
    """Prompt template for direct question answering without context."""

    def __init__(self, question_format: str = 'mc'):
        self.question_format = question_format

    def preprocess(
        self,
        text: Union[str, List[str]],
        contexts: Optional[List[List[str]]] = None,
        scores: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Preprocess the texts into direct prompts without any context."""
        if isinstance(text, str):
            text = [text]
        # Always return texts as-is, ignoring any context
        return text

    def postprocess(self, responses: List[str]) -> List[str]:
        """Postprocess the responses."""
        return responses


class VLLMGeneratorConfig(BaseConfig):
    """Configuration for the vLLM generator."""

    server: str = Field(
        ..., description='Cels machine you are running on, e.g, rbdgx1'
    )
    port: int = Field(..., description='The port vLLM is listening to.')
    api_key: str = Field(
        ..., description='The API key for vLLM server, e.g., CELS'
    )
    model: str = Field(
        ..., description='The model that vLLM server is running.'
    )
    temperature: float = Field(0.0, description='Temperature for generation.')
    max_tokens: int = Field(
        1024, description='Maximum number of tokens to generate.'
    )

    def get_generator(self) -> 'VLLMGenerator':
        """Get the vLLM generator."""
        return VLLMGenerator(self)


class VLLMGenerator:
    """Generator for vLLM server."""

    def __init__(self, config: VLLMGeneratorConfig) -> None:
        self.config = config
        self.base_url = f'http://{config.server}:{config.port}'
        self.api_key = config.api_key
        self.model = config.model

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using vLLM server."""
        # Use config defaults if not provided
        if temperature is None:
            temperature = self.config.temperature
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        url = f'{self.base_url}/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        data = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens,
        }

        try:
            response = requests.post(
                url, headers=headers, json=data, timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            result = f'Error: {e!s}'
        except (KeyError, IndexError) as e:
            result = f'Error: {e!s}'

        return result


class ArgoGeneratorConfig(BaseConfig):
    """Configuration for the Argo generator using OpenAI client."""

    model: str = Field(
        default_factory=lambda: os.getenv('MODEL', 'argo:gpt-4o'),
        description='The model name for Argo proxy.',
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            'BASE_URL', 'http://localhost:56267'
        ),
        description='The base URL for the Argo proxy server.',
    )
    api_key: str = Field(
        'whatever+random',
        description='The API key for Argo proxy (can be any string).',
    )
    temperature: float = Field(0.0, description='Temperature for generation.')
    max_tokens: int = Field(
        1024, description='Maximum number of tokens to generate.'
    )

    def get_generator(self) -> 'ArgoGenerator':
        """Get the Argo generator."""
        return ArgoGenerator(self)


class ArgoGenerator:
    """Generator for Argo proxy using OpenAI client."""

    def __init__(self, config: ArgoGeneratorConfig) -> None:
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Argo proxy."""
        # Use config defaults if not provided
        if temperature is None:
            temperature = self.config.temperature
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            result = f'Error: {e!s}'

        return result


class RagGeneratorWithChunkLogging:
    """RAG generator for generating responses to queries with chunk logging and no-RAG support."""

    def __init__(
        self,
        generator: Union[VLLMGenerator, ArgoGenerator],
        retriever: Optional[Retriever] = None,
        verbose: bool = False,
        use_rag: bool = True,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.verbose = verbose
        self.use_rag = use_rag  # New flag to control RAG usage

    def generate(
        self,
        texts: Union[str, List[str]],
        prompt_template: Optional[Any] = None,
        retrieval_top_k: int = 5,
        retrieval_score_threshold: float = 0.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        contexts: Optional[List[List[str]]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Generate responses to the given queries and return retrieval info."""
        if isinstance(texts, str):
            texts = [texts]

        # Use the identity prompt template if none is provided
        if prompt_template is None:
            prompt_template = IdentityPromptTemplate(
                IdentityPromptTemplateConfig()
            )

        # Initialize retrieval info
        retrieval_info = {
            'retrieved_chunks': [],
            'retrieval_scores': [],
            'retrieval_indices': [],
            'context_source': 'none',
            'rag_enabled': self.use_rag,
            'error': None,
        }

        # Default: no context
        retrieved_contexts, scores = None, None

        # Only use RAG if enabled
        if self.use_rag:
            # Use provided contexts (from JSON text field) or retrieve if retriever is available
            if contexts is not None:
                retrieved_contexts = contexts
                scores = None
                retrieval_info['context_source'] = 'provided'
            elif self.retriever is not None:
                try:
                    results, _ = self.retriever.search(
                        texts,
                        top_k=retrieval_top_k,
                        score_threshold=retrieval_score_threshold,
                    )
                    retrieved_contexts = [
                        self.retriever.get_texts(indices)
                        for indices in results.total_indices
                    ]
                    scores = results.total_scores
                    retrieval_info['context_source'] = 'retrieved'
                    retrieval_info['retrieval_indices'] = results.total_indices
                    retrieval_info['retrieval_scores'] = results.total_scores

                    # Extract chunk information for each query
                    for query_idx, indices in enumerate(results.total_indices):
                        if indices:
                            # Get additional information about each chunk
                            chunk_info = []
                            for idx in indices:
                                chunk_data = {
                                    'dataset_index': idx,
                                    'text': self.retriever.get([idx], 'text')[
                                        0
                                    ],
                                    'score': scores[query_idx][
                                        indices.index(idx)
                                    ]
                                    if scores
                                    else 0.0,
                                }

                                # Try to get path information if available
                                try:
                                    path = self.retriever.get([idx], 'path')[0]
                                    chunk_data['path'] = path
                                    chunk_data['chunk_id'] = generate_chunk_id(
                                        idx, path
                                    )
                                except (KeyError, IndexError):
                                    # If path is not available, use a placeholder
                                    chunk_data['path'] = f'unknown_path_{idx}'
                                    chunk_data['chunk_id'] = generate_chunk_id(
                                        idx, f'unknown_path_{idx}'
                                    )

                                chunk_info.append(chunk_data)

                            retrieval_info['retrieved_chunks'].append(
                                chunk_info
                            )
                        else:
                            retrieval_info['retrieved_chunks'].append([])

                except Exception as e:
                    retrieval_info['error'] = str(e)
                    print(f'Error during retrieval: {e}')
        else:
            # No RAG mode - explicitly set context source
            retrieval_info['context_source'] = 'disabled'

        # Build the final prompts
        prompts = prompt_template.preprocess(texts, retrieved_contexts, scores)

        # If verbose, print contexts and chunk information
        if (
            self.verbose
            and self.use_rag
            and retrieved_contexts
            and retrieved_contexts[0]
        ):
            print('Retrieved contexts:')
            for i, context in enumerate(retrieved_contexts[0]):
                print(f'  Context {i + 1}: {context[:200]}...')

            # Print chunk information if available
            if retrieval_info['retrieved_chunks']:
                print('Retrieved chunks:')
                for i, chunk in enumerate(
                    retrieval_info['retrieved_chunks'][0]
                ):
                    print(
                        f'  Chunk {i + 1}: ID={chunk["chunk_id"]}, Score={chunk["score"]:.4f}, Path={chunk["path"]}'
                    )

        # Generate responses
        responses = []
        for prompt in prompts:
            result = self.generator.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            responses.append(result)

        return responses, retrieval_info


class RetrievalAugmentedGenerationConfig(BaseConfig):
    """Configuration for the retrieval-augmented generation model."""

    generator_config: Union[VLLMGeneratorConfig, ArgoGeneratorConfig] = Field(
        ..., description='Settings for the generator (VLLM or Argo)'
    )
    retriever_config: Optional[RetrieverConfig] = Field(
        None, description='Settings for the retriever'
    )
    verbose: bool = Field(False, description='Whether to print verbose output')
    use_rag: bool = Field(
        True, description='Whether to use RAG or direct question answering'
    )

    def get_rag_model(self) -> RagGeneratorWithChunkLogging:
        """Get the retrieval-augmented generation model."""
        # Get the generator based on the config type
        if isinstance(self.generator_config, VLLMGeneratorConfig):
            generator = self.generator_config.get_generator()
        elif isinstance(self.generator_config, ArgoGeneratorConfig):
            generator = self.generator_config.get_generator()
        else:
            raise ValueError(
                f'Unsupported generator config type: {type(self.generator_config)}'
            )

        # Initialize the retriever (only if RAG is enabled)
        retriever = None
        if self.use_rag and self.retriever_config is not None:
            retriever = self.retriever_config.get_retriever()

        # Initialize the RAG model
        rag_model = RagGeneratorWithChunkLogging(
            generator=generator,
            retriever=retriever,
            verbose=self.verbose,
            use_rag=self.use_rag,
        )

        return rag_model


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    giveup=lambda e: 'Invalid authentication' in str(e),
    max_time=300,
)
def generate_rag_answer(
    question: str,
    rag_model: RagGeneratorWithChunkLogging,
    question_format: str = 'auto',
    context_text: Optional[str] = None,
    retrieval_top_k: int = 5,
    retrieval_score_threshold: float = 0.0,
    use_rag: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Generate an answer to a question using RAG and return chunk information."""
    # Auto-detect question format if not specified
    actual_format = question_format
    if question_format == 'auto':
        mc_patterns = [
            r'(?:^|\n)\s*([A-E])[.):]\s',
            r'(?:^|\n)\s*([1-5])[.):]\s',
        ]
        is_mc = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in mc_patterns
        )
        actual_format = 'mc' if is_mc else 'qa'

    # Create appropriate prompt template based on RAG usage
    if use_rag:
        prompt_template = RAGPromptTemplate(question_format=actual_format)
    else:
        prompt_template = DirectPromptTemplate(question_format=actual_format)

    # Prepare contexts if using context field (ignored if RAG is disabled)
    contexts = None
    if use_rag and context_text:
        contexts = [[context_text]]

    # Add a small random delay to avoid overwhelming the server
    jitter = random.uniform(0.1, 1.0)
    time.sleep(jitter)

    try:
        # Generate answer using RAG or direct generation
        responses, retrieval_info = rag_model.generate(
            texts=[question],
            prompt_template=prompt_template,
            retrieval_top_k=retrieval_top_k,
            retrieval_score_threshold=retrieval_score_threshold,
            contexts=contexts,
        )
        return responses[0], retrieval_info
    except Exception as e:
        print(f'Error in generate_rag_answer (will retry): {str(e)}')
        raise


def _evaluate_answer_with_retry(
    question, reference_answer, model_answer, config, question_format='auto'
):
    """Wrapper for evaluate_answer with custom retry logic for JSON parsing issues."""
    try:
        return _evaluate_answer_core(
            question, reference_answer, model_answer, config, question_format
        )
    except json.JSONDecodeError as e:
        print(f'JSON parsing error (will retry): {str(e)}')
        # Try again with a different approach
        try:
            return _evaluate_answer_core(
                question,
                reference_answer,
                model_answer,
                config,
                question_format,
                retry_count=1,
            )
        except json.JSONDecodeError as e2:
            print(f'JSON parsing error on retry (will retry again): {str(e2)}')
            # Final attempt with simplified format
            return _evaluate_answer_core(
                question,
                reference_answer,
                model_answer,
                config,
                question_format,
                retry_count=2,
            )


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    giveup=lambda e: 'Invalid authentication' in str(e),
    max_time=300,
)
def _evaluate_answer_core(
    question,
    reference_answer,
    model_answer,
    config,
    question_format='auto',
    retry_count=0,
):
    """Core evaluation logic with retry handling."""
    # Auto-detect question format if not specified
    actual_format = question_format
    if question_format == 'auto':
        mc_patterns = [
            r'(?:^|\n)\s*([A-E])[.):]\s',
            r'(?:^|\n)\s*([1-5])[.):]\s',
        ]
        is_mc = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in mc_patterns
        )
        actual_format = 'mc' if is_mc else 'qa'

    # Get the OpenAI client
    client = get_openai_client(
        config['openai_api_key'],
        config['openai_api_base'],
        config.get('timeout', 120.0),
    )

    # Build the evaluation prompt based on format and retry count
    if actual_format == 'mc':
        if retry_count == 0:
            # Standard JSON format
            system_prompt = '''You are an expert evaluator for multiple choice questions. You must evaluate whether the model's answer matches the reference answer.

Your task:
1. Extract the letter choice (A, B, C, D, E) or number choice (1, 2, 3, 4, 5) from both the reference answer and model answer
2. Compare the extracted choices
3. Provide a detailed evaluation

Return your response in valid JSON format with these fields:
- "reference_choice": the choice from the reference answer
- "model_choice": the choice from the model answer  
- "score": 1 if choices match, 0 if they don't match
- "reasoning": explanation of your evaluation
- "format": "mc"'''
        elif retry_count == 1:
            # More explicit JSON format
            system_prompt = """You are an expert evaluator. Extract the choice from each answer and compare them.

IMPORTANT: Return ONLY valid JSON with these exact fields:
{
  "reference_choice": "extracted choice from reference",
  "model_choice": "extracted choice from model",
  "score": 1 or 0,
  "reasoning": "your explanation",
  "format": "mc"
}

Do not include any text outside the JSON."""
        else:
            # Simplified format for final retry
            system_prompt = """Extract the choice letter/number from each answer and compare them. Return only JSON format."""

        user_prompt = f"""Question: {question}

Reference Answer: {reference_answer}

Model Answer: {model_answer}

Evaluate if the model's choice matches the reference choice."""

    else:  # qa format
        if retry_count == 0:
            system_prompt = '''You are an expert evaluator for question-answering tasks. You must evaluate whether the model's answer is correct compared to the reference answer.

Your task:
1. Compare the semantic meaning of both answers
2. Determine if the model answer is correct, partially correct, or incorrect
3. Provide a detailed evaluation

Return your response in valid JSON format with these fields:
- "score": number between 0 and 1 (1 = perfect match, 0.5 = partially correct, 0 = incorrect)
- "reasoning": detailed explanation of your evaluation
- "format": "qa"'''
        elif retry_count == 1:
            system_prompt = """You are an expert evaluator. Compare the answers and return ONLY valid JSON:

{
  "score": 0.0 to 1.0,
  "reasoning": "your explanation",
  "format": "qa"
}

Do not include any text outside the JSON."""
        else:
            system_prompt = """Compare the answers. Return only JSON format with score and reasoning."""

        user_prompt = f"""Question: {question}

Reference Answer: {reference_answer}

Model Answer: {model_answer}

Evaluate the correctness of the model's answer."""

    # Make the API call
    response = client.chat.completions.create(
        model=config['openai_model'],
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0.0,
        max_tokens=1000,
    )

    response_text = response.choices[0].message.content.strip()

    # Parse the JSON response
    try:
        evaluation = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                evaluation = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                raise json.JSONDecodeError(
                    'Could not parse JSON from response', response_text, 0
                )
        else:
            raise json.JSONDecodeError(
                'No JSON found in response', response_text, 0
            )

    # Ensure required fields are present
    if 'score' not in evaluation:
        evaluation['score'] = 0
    if 'reasoning' not in evaluation:
        evaluation['reasoning'] = 'No reasoning provided'
    if 'format' not in evaluation:
        evaluation['format'] = actual_format

    # For MC questions, also extract the choices if present
    if actual_format == 'mc':
        if 'reference_choice' not in evaluation:
            evaluation['reference_choice'] = extract_choice_identifier(
                reference_answer
            )
        if 'model_choice' not in evaluation:
            evaluation['model_choice'] = extract_choice_identifier(
                model_answer
            )

    return evaluation


def evaluate_answer(
    question, reference_answer, model_answer, config, question_format='auto'
):
    """Evaluate a model's answer against the reference answer."""
    return _evaluate_answer_with_retry(
        question, reference_answer, model_answer, config, question_format
    )


def process_question(
    item,
    rag_model: RagGeneratorWithChunkLogging,
    grader_config: Dict[str, Any],
    question_format: str = 'auto',
    verbose: bool = False,
    use_context_field: bool = False,
    retrieval_top_k: int = 5,
    retrieval_score_threshold: float = 0.0,
    use_rag: bool = True,
):
    """Process a single question - generate answer with RAG and evaluate it."""
    i, qa_pair = item
    question = qa_pair.get('question', '')
    reference_answer = qa_pair.get('answer', '')
    context_text = (
        qa_pair.get('text', '') if use_context_field and use_rag else None
    )

    if not question or not reference_answer:
        return {
            'question_id': i,
            'error': 'Missing question or answer',
            'skipped': True,
        }

    try:
        if verbose:
            print(f'\nProcessing question {i}...')

        # Generate model answer with RAG or direct generation
        start_time = time.time()
        model_answer, retrieval_info = generate_rag_answer(
            question=question,
            rag_model=rag_model,
            question_format=question_format,
            context_text=context_text,
            retrieval_top_k=retrieval_top_k,
            retrieval_score_threshold=retrieval_score_threshold,
            use_rag=use_rag,
        )
        model_time = time.time() - start_time

        if verbose:
            print(
                f'\n--- {"RAG" if use_rag else "DIRECT"} RESPONSE FOR QUESTION {i} ---'
            )
            print(model_answer)
            print(f'--- END {"RAG" if use_rag else "DIRECT"} RESPONSE ---')
            print(f'Generated answer for question {i} in {model_time:.2f}s')

            # Print retrieval information if available
            if use_rag and retrieval_info['retrieved_chunks']:
                print(f'--- RETRIEVED CHUNKS FOR QUESTION {i} ---')
                for chunk_list in retrieval_info['retrieved_chunks']:
                    for chunk in chunk_list:
                        print(f'  Chunk ID: {chunk["chunk_id"]}')
                        print(f'  Score: {chunk["score"]:.4f}')
                        print(f'  Path: {chunk["path"]}')
                        print(f'  Text: {chunk["text"][:200]}...')
                        print()
                print('--- END RETRIEVED CHUNKS ---')

        # Evaluate the answer
        start_time = time.time()
        evaluation = evaluate_answer(
            question,
            reference_answer,
            model_answer,
            grader_config,
            question_format,
        )
        eval_time = time.time() - start_time

        if verbose:
            print(f'Evaluated answer for question {i} in {eval_time:.2f}s')

        # Get the score and format
        score = evaluation.get('score', 0)
        format_type = evaluation.get('format', question_format)
        if format_type == 'auto':
            format_type = 'mc' if 'correct_letter' in evaluation else 'qa'

        # Prepare detailed result
        result = {
            'question_id': i,
            'question': question,
            'reference_answer': reference_answer,
            'model_answer': model_answer,
            'evaluation': evaluation,
            'score': score,
            'format': format_type,
            'model_time_seconds': model_time,
            'evaluation_time_seconds': eval_time,
            'retrieval_info': retrieval_info,  # Include retrieval information
            'skipped': False,
        }

        # Add context information if used
        if use_context_field and context_text and use_rag:
            result['context_used'] = context_text

        return result

    except Exception as e:
        return {
            'question_id': i,
            'error': str(e),
            'skipped': True,
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced Question Grader with RAG and Chunk Logging'
    )
    parser.add_argument(
        'questions_file',
        help='Path to the JSON file containing questions',
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model shortname from the config file',
    )
    parser.add_argument(
        '--grader',
        required=True,
        help='Grader model shortname from the config file',
    )
    parser.add_argument(
        '--config',
        default='model_servers.yaml',
        help='Path to the model configuration file',
    )
    parser.add_argument(
        '--rag-config',
        help='Path to the RAG configuration file (YAML)',
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel workers',
    )
    parser.add_argument(
        '--format',
        choices=['auto', 'mc', 'qa'],
        default='auto',
        help='Question format',
    )
    parser.add_argument(
        '--random',
        type=int,
        help='Randomly select N questions from the dataset',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible question selection',
    )
    parser.add_argument(
        '--save-incorrect',
        action='store_true',
        help='Save incorrectly answered questions to a separate file',
    )
    parser.add_argument(
        '--use-context-field',
        action='store_true',
        help='Use the "text" field from JSON as context instead of retrieval (ignored if --no-rag is used)',
    )
    parser.add_argument(
        '--retrieval-top-k',
        type=int,
        default=5,
        help='Number of documents to retrieve',
    )
    parser.add_argument(
        '--retrieval-score-threshold',
        type=float,
        default=0.0,
        help='Minimum retrieval score threshold',
    )
    parser.add_argument(
        '--log-chunks',
        action='store_true',
        help='Enable detailed chunk logging (always enabled in v2)',
    )
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG completely - questions are asked directly to the model without any context',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )

    return parser.parse_args()


def create_metadata(
    args, model_config, grader_config, questions, rag_config=None
):
    """Create metadata for the evaluation run."""
    metadata = {
        'evaluation_metadata': {
            'script_version': '2.0',
            'script_name': 'rag_argonium_score_parallel_v2.py',
            'timestamp': datetime.now().isoformat(),
            'arguments': vars(args),
            'rag_enabled': not args.no_rag,
            'model_configuration': {
                'model_shortname': args.model,
                'model_name': model_config.get('model', 'unknown'),
                'server': model_config.get('server', 'unknown'),
                'port': model_config.get('port', 'unknown'),
            },
            'grader_configuration': {
                'grader_shortname': args.grader,
                'grader_model': grader_config.get('openai_model', 'unknown'),
                'grader_api_base': grader_config.get(
                    'openai_api_base', 'unknown'
                ),
            },
            'question_statistics': {
                'total_questions': len(questions),
                'selected_questions': len(questions)
                if not args.random
                else args.random,
                'question_format': args.format,
                'random_seed': args.seed,
            },
            'rag_configuration': {
                'enabled': not args.no_rag,
                'use_context_field': args.use_context_field
                and not args.no_rag,
                'retrieval_top_k': args.retrieval_top_k,
                'retrieval_score_threshold': args.retrieval_score_threshold,
                'rag_config_file': args.rag_config,
                'rag_config_details': rag_config if rag_config else None,
            },
            'processing_configuration': {
                'parallel_workers': args.parallel,
                'verbose': args.verbose,
                'chunk_logging_enabled': True,  # Always enabled in v2
            },
        }
    }

    # Add dataset path if available from RAG config
    if rag_config and 'retriever_config' in rag_config:
        retriever_config = rag_config['retriever_config']
        if 'faiss_config' in retriever_config:
            faiss_config = retriever_config['faiss_config']
            if 'dataset_dir' in faiss_config:
                metadata['evaluation_metadata']['rag_configuration'][
                    'dataset_path'
                ] = str(faiss_config['dataset_dir'])

    return metadata


def main():
    """Main function."""
    args = parse_arguments()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Load questions
    with open(args.questions_file, 'r') as f:
        questions = json.load(f)

    # Randomly select questions if specified
    if args.random:
        if args.random < len(questions):
            questions = random.sample(questions, args.random)
            print(f'Randomly selected {len(questions)} questions')
        else:
            print(
                f'Requested {args.random} questions but only {len(questions)} available'
            )

    # Load model configurations
    model_config = load_model_config(args.model, args.config)
    grader_config = load_model_config(args.grader, args.config)

    # Load RAG configuration if provided
    rag_config = None
    if args.rag_config:
        with open(args.rag_config, 'r') as f:
            rag_config = yaml.safe_load(f)
        print(f'Using RAG configuration from {args.rag_config}')

    # Auto-detect question format
    question_format = args.format
    if question_format == 'auto':
        question_format = detect_question_format(questions)

    # Determine if RAG should be used
    use_rag = not args.no_rag

    # Create RAG model
    if use_rag and rag_config:
        # Use provided RAG configuration
        rag_model_config = RetrievalAugmentedGenerationConfig(**rag_config)
        rag_model_config.use_rag = True
    elif use_rag and not args.use_context_field:
        # Create basic RAG model without retrieval
        if 'argo' in args.model.lower():
            generator_config = ArgoGeneratorConfig(
                model=model_config['openai_model'],
                base_url=model_config['openai_api_base'],
                api_key=model_config['openai_api_key'],
            )
        else:
            generator_config = VLLMGeneratorConfig(
                server=model_config['server'],
                port=model_config['port'],
                api_key=model_config['api_key'],
                model=model_config['model'],
            )

        rag_model_config = RetrievalAugmentedGenerationConfig(
            generator_config=generator_config,
            retriever_config=None,
            verbose=args.verbose,
            use_rag=True,
        )
    else:
        # Create model for context field usage or no-RAG mode
        if 'argo' in args.model.lower():
            generator_config = ArgoGeneratorConfig(
                model=model_config['openai_model'],
                base_url=model_config['openai_api_base'],
                api_key=model_config['openai_api_key'],
            )
        else:
            generator_config = VLLMGeneratorConfig(
                server=model_config['server'],
                port=model_config['port'],
                api_key=model_config['api_key'],
                model=model_config['model'],
            )

        rag_model_config = RetrievalAugmentedGenerationConfig(
            generator_config=generator_config,
            retriever_config=None,
            verbose=args.verbose,
            use_rag=use_rag,
        )

    # Get the RAG model
    rag_model = rag_model_config.get_rag_model()

    # Print configuration
    print(f'Testing model: {args.model} ({model_config["model"]})')
    print(f'Grading with: {args.grader} ({grader_config["openai_model"]})')
    print(f'Parallel workers: {args.parallel}')
    print(f'RAG enabled: {use_rag}')
    if use_rag:
        print(f'Using context field: {args.use_context_field}')
        print(f'Retrieval top-k: {args.retrieval_top_k}')
        print(f'Retrieval score threshold: {args.retrieval_score_threshold}')
        print(f'Chunk logging: Enabled')

        if rag_config:
            print('Using provided RAG configuration')
        elif not args.use_context_field:
            print('Using basic RAG model without retrieval')
        else:
            print('Using context field from JSON')
    else:
        print('RAG DISABLED - Direct question answering mode')
        print(
            'Questions will be asked directly to the model without any context'
        )

    # Create metadata
    metadata = create_metadata(
        args, model_config, grader_config, questions, rag_config
    )

    # Process questions
    print(
        f'\nProcessing {len(questions)} questions with {"RAG" if use_rag else "direct generation"}...'
    )

    if args.parallel > 1:
        print(f'Using {args.parallel} parallel workers...')
        print(
            'This may take some time. Each model call has built-in retries and waiting.'
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.parallel
        ) as executor:
            futures = [
                executor.submit(
                    process_question,
                    (i, question),
                    rag_model,
                    grader_config,
                    question_format,
                    args.verbose,
                    args.use_context_field,
                    args.retrieval_top_k,
                    args.retrieval_score_threshold,
                    use_rag,
                )
                for i, question in enumerate(questions)
            ]

            results = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc='Processing questions',
            ):
                results.append(future.result())
    else:
        results = []
        for i, question in enumerate(
            tqdm(questions, desc='Processing questions')
        ):
            result = process_question(
                (i, question),
                rag_model,
                grader_config,
                question_format,
                args.verbose,
                args.use_context_field,
                args.retrieval_top_k,
                args.retrieval_score_threshold,
                use_rag,
            )
            results.append(result)

    # Sort results by question_id
    results.sort(key=lambda x: x['question_id'])

    # Calculate statistics
    processed_results = [r for r in results if not r.get('skipped', False)]
    total_questions = len(questions)
    processed_questions = len(processed_results)
    skipped_questions = total_questions - processed_questions

    if processed_questions > 0:
        scores = [r['score'] for r in processed_results]
        accuracy = sum(scores) / len(scores)

        # Format-specific statistics
        mc_results = [r for r in processed_results if r.get('format') == 'mc']
        qa_results = [r for r in processed_results if r.get('format') == 'qa']

        print(f'\n--- RESULTS ---')
        print(f'Total questions: {total_questions}')
        print(f'Processed questions: {processed_questions}')
        print(f'Skipped questions: {skipped_questions}')
        print(f'Overall accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)')

        if mc_results:
            mc_accuracy = sum(r['score'] for r in mc_results) / len(mc_results)
            print(
                f'Multiple choice accuracy: {mc_accuracy:.4f} ({mc_accuracy * 100:.2f}%) ({len(mc_results)} questions)'
            )

        if qa_results:
            qa_accuracy = sum(r['score'] for r in qa_results) / len(qa_results)
            print(
                f'Question-answer accuracy: {qa_accuracy:.4f} ({qa_accuracy * 100:.2f}%) ({len(qa_results)} questions)'
            )

        # Chunk retrieval statistics (only if RAG is enabled)
        if use_rag:
            total_chunks_retrieved = 0
            questions_with_retrieval = 0

            for result in processed_results:
                if result.get('retrieval_info', {}).get('retrieved_chunks'):
                    questions_with_retrieval += 1
                    for chunk_list in result['retrieval_info'][
                        'retrieved_chunks'
                    ]:
                        total_chunks_retrieved += len(chunk_list)

            if questions_with_retrieval > 0:
                print(f'\n--- RETRIEVAL STATISTICS ---')
                print(f'Questions with retrieval: {questions_with_retrieval}')
                print(f'Total chunks retrieved: {total_chunks_retrieved}')
                print(
                    f'Average chunks per question: {total_chunks_retrieved / questions_with_retrieval:.2f}'
                )

    else:
        print('\nNo questions were successfully processed.')
        return

    # Save results with metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'rag_results_{args.model}_{timestamp}.json'

    # Create final output structure with metadata
    output_data = {
        'metadata': metadata,
        'results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f'Results saved to {output_file}')

    # Save incorrect answers if requested
    if args.save_incorrect:
        incorrect_results = [r for r in processed_results if r['score'] < 1.0]
        if incorrect_results:
            incorrect_file = f'incorrect_answers_{args.model}_{timestamp}.json'
            incorrect_data = {
                'metadata': metadata,
                'results': incorrect_results,
            }
            with open(incorrect_file, 'w') as f:
                json.dump(incorrect_data, f, indent=2)
            print(f'Incorrect answers saved to {incorrect_file}')

    # Print sample chunk logging if available
    if (
        use_rag
        and processed_results
        and processed_results[0]
        .get('retrieval_info', {})
        .get('retrieved_chunks')
    ):
        print(f'\n--- SAMPLE CHUNK LOGGING ---')
        sample_result = processed_results[0]
        print(f'Question ID: {sample_result["question_id"]}')
        print(f'Retrieved chunks:')
        for chunk_list in sample_result['retrieval_info']['retrieved_chunks']:
            for chunk in chunk_list:
                print(f'  - Chunk ID: {chunk["chunk_id"]}')
                print(f'    Score: {chunk["score"]:.4f}')
                print(f'    Path: {chunk["path"]}')
                print(f'    Text preview: {chunk["text"][:100]}...')
                print()


if __name__ == '__main__':
    main()
