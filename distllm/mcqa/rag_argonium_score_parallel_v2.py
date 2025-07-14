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
- Source chunk retrieval tracking to verify if original source was retrieved
- Pydantic configuration management with YAML support

Usage:
    # Recommended: Use YAML config file (everything specified in config)
    python rag_argonium_score_parallel_v2.py --config <mcqa_config.yaml>

    # Alternative: Command-line arguments (for backward compatibility)
    python rag_argonium_score_parallel_v2.py <questions_file.json> --model <model_shortname> --grader <grader_shortname> [--config <mcqa_config.yaml>]

Where:
    - config: MCQA configuration file (YAML) containing all evaluation settings including questions file, model, and grader
    - questions_file.json: A JSON file with an array of objects, each having "question", "answer", and optionally "text" fields
    - model_shortname: The shortname of the model to test from model_servers.yaml
    - grader_shortname: The shortname of the model to use for grading from model_servers.yaml

Examples:
    # Recommended: Everything in YAML config
    python rag_argonium_score_parallel_v2.py --config mcqa_config.yaml

    # Override specific settings from config
    python rag_argonium_score_parallel_v2.py --config mcqa_config.yaml --no-rag

    # Traditional command-line usage (requires all arguments)
    python rag_argonium_score_parallel_v2.py questions.json --model llama --grader gpt41 --parallel 4

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
10) Tracks whether the source chunk of each question was retrieved (RAG mode only)
11) Uses Pydantic and YAML for clean configuration management
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
from pydantic import BaseModel, Field, field_validator
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
# Configuration Classes
# -----------------------------------------------------------------------------


class GeneratorConfig(BaseModel):
    """Base generator configuration."""

    generator_type: str = Field(
        ..., description="Generator type: 'vllm' or 'argo'"
    )


class VLLMGeneratorSettings(BaseModel):
    """VLLM-specific generator settings."""

    server: str = Field(..., description='Server hostname (e.g., rbdgx1)')
    model: str = Field(..., description='Model name/path')
    port: int = Field(..., description='Port number')
    api_key: str = Field(..., description='API key')
    temperature: float = Field(0.0, description='Generation temperature')
    max_tokens: int = Field(1024, description='Maximum tokens to generate')


class ArgoGeneratorSettings(BaseModel):
    """Argo-specific generator settings."""

    model: str = Field(
        ..., description="Argo model name (e.g., 'argo:gpt-4o')"
    )
    base_url: str = Field(..., description='Argo proxy base URL')
    api_key: str = Field('whatever+random', description='API key for Argo')
    temperature: float = Field(0.0, description='Generation temperature')
    max_tokens: int = Field(1024, description='Maximum tokens to generate')


class ModelConfiguration(BaseModel):
    """Model configuration settings."""

    generator: GeneratorConfig
    generator_settings: Union[VLLMGeneratorSettings, ArgoGeneratorSettings] = (
        Field(..., description='Generator-specific settings')
    )
    grader_shortname: str = Field(
        ..., description='Grader model shortname from model_servers.yaml'
    )
    model_config_file: str = Field(
        'model_servers.yaml', description='Model configuration file path'
    )

    @field_validator('generator_settings')
    @classmethod
    def validate_generator_settings(cls, v, values):
        generator_data = (
            values.data.get('generator', {}) if hasattr(values, 'data') else {}
        )
        generator_type = (
            getattr(generator_data, 'generator_type', None)
            if hasattr(generator_data, 'generator_type')
            else None
        )

        if generator_type == 'vllm' and not isinstance(
            v, VLLMGeneratorSettings
        ):
            raise ValueError(
                "generator_settings must be VLLMGeneratorSettings when generator_type is 'vllm'"
            )
        elif generator_type == 'argo' and not isinstance(
            v, ArgoGeneratorSettings
        ):
            raise ValueError(
                "generator_settings must be ArgoGeneratorSettings when generator_type is 'argo'"
            )

        return v


# Retriever Configuration Classes
class FaissIndexConfiguration(BaseModel):
    """Configuration for FAISS Index."""

    name: str = Field('faiss_index_v2', description='Index name')
    dataset_dir: str = Field(
        ..., description='Path to the HF dataset directory'
    )
    faiss_index_path: str = Field(..., description='Path to the FAISS index')
    dataset_chunk_paths: Optional[List[str]] = Field(
        None, description='Paths to dataset chunks'
    )
    precision: str = Field('float32', description='Embedding precision')
    search_algorithm: str = Field('exact', description='Search algorithm')
    rescore_multiplier: int = Field(
        2, description='Oversampling factor for rescoring'
    )
    num_quantization_workers: int = Field(
        1, description='Number of quantization workers'
    )


class EncoderConfiguration(BaseModel):
    """Configuration for text encoder."""

    name: str = Field('auto', description='Encoder name')
    pretrained_model_name_or_path: str = Field(
        ..., description='Pre-trained model name or path'
    )
    tokenizer_name: Optional[str] = Field(
        None, description='Optional tokenizer name'
    )
    half_precision: bool = Field(False, description='Use half precision')
    eval_mode: bool = Field(True, description='Set model to evaluation mode')
    compile_model: bool = Field(
        False, description='Compile model for faster inference'
    )
    quantization: bool = Field(True, description='Use quantization')


class PoolerConfiguration(BaseModel):
    """Configuration for pooler."""

    name: str = Field('mean', description='Pooler name (mean or last_token)')


class RetrieverConfiguration(BaseModel):
    """Configuration for the retriever."""

    faiss_config: FaissIndexConfiguration = Field(
        ..., description='FAISS index configuration'
    )
    encoder_config: EncoderConfiguration = Field(
        ..., description='Encoder configuration'
    )
    pooler_config: PoolerConfiguration = Field(
        ..., description='Pooler configuration'
    )
    batch_size: int = Field(4, description='Batch size for the embedder model')


class RAGConfiguration(BaseModel):
    """RAG-specific configuration settings."""

    enabled: bool = Field(True, description='Enable RAG functionality')
    rag_config_file: Optional[str] = Field(
        None, description='RAG configuration file (YAML)'
    )
    retriever_config: Optional[RetrieverConfiguration] = Field(
        None, description='Retriever configuration'
    )
    use_context_field: bool = Field(
        False, description="Use 'text' field from JSON as context"
    )
    retrieval_top_k: int = Field(
        5, description='Number of documents to retrieve'
    )
    retrieval_score_threshold: float = Field(
        0.0, description='Minimum retrieval score threshold'
    )
    chunk_logging_enabled: bool = Field(
        True, description='Enable detailed chunk logging'
    )


class ProcessingConfiguration(BaseModel):
    """Processing and execution configuration."""

    parallel_workers: int = Field(1, description='Number of parallel workers')
    question_format: str = Field(
        'auto', description='Question format: auto, mc, or qa'
    )
    verbose: bool = Field(False, description='Enable verbose output')
    random_selection: Optional[int] = Field(
        None, description='Randomly select N questions'
    )
    random_seed: Optional[int] = Field(
        None, description='Random seed for reproducible selection'
    )


class OutputConfiguration(BaseModel):
    """Output and result configuration."""

    save_incorrect: bool = Field(
        False, description='Save incorrectly answered questions'
    )
    output_directory: str = Field(
        '.', description='Output directory for results'
    )
    output_prefix: str = Field(
        'rag_results', description='Prefix for output files'
    )


class MCQAConfig(BaseModel):
    """Main MCQA evaluation configuration."""

    questions_file: str = Field(
        ..., description='Path to JSON file containing questions'
    )
    model: ModelConfiguration
    rag: RAGConfiguration = RAGConfiguration()
    processing: ProcessingConfiguration = ProcessingConfiguration()
    output: OutputConfiguration = OutputConfiguration()

    @field_validator('processing')
    def validate_processing(cls, v):
        if v.question_format not in ['auto', 'mc', 'qa']:
            raise ValueError("question_format must be 'auto', 'mc', or 'qa'")
        if v.parallel_workers < 1:
            raise ValueError('parallel_workers must be >= 1')
        return v

    @field_validator('rag')
    def validate_rag(cls, v):
        if v.retrieval_top_k < 1:
            raise ValueError('retrieval_top_k must be >= 1')
        if v.retrieval_score_threshold < 0:
            raise ValueError('retrieval_score_threshold must be >= 0')
        return v

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MCQAConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)


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


def check_source_chunk_retrieved(
    qa_pair: Dict[str, Any], retrieval_info: Dict[str, Any], use_rag: bool
) -> Optional[bool]:
    """
    Check if the source chunk of the question was included in the retrieved chunks.

    Args:
        qa_pair: Question-answer pair that might contain source information
        retrieval_info: Information about retrieved chunks
        use_rag: Whether RAG was used for this evaluation

    Returns:
        True if source chunk was retrieved, False if not, None if RAG not used or no source info
    """
    # If RAG was not used, return None
    if not use_rag:
        return None

    # If no retrieval info or no retrieved chunks, return None
    if not retrieval_info or not retrieval_info.get('retrieved_chunks'):
        return None

    # Look for source information in the question data
    # Check various possible field names for source information
    source_fields = [
        'source_chunk_id',
        'chunk_id',
        'source_id',
        'source_file',
        'source_path',
        'dataset_index',
        'source_dataset_index',
    ]

    source_info = None
    source_field_name = None

    for field in source_fields:
        if field in qa_pair:
            source_info = qa_pair[field]
            source_field_name = field
            break

    # If no source information found, return None
    if source_info is None:
        return None

    # Get all retrieved chunks
    retrieved_chunks = []
    for chunk_list in retrieval_info['retrieved_chunks']:
        retrieved_chunks.extend(chunk_list)

    # Check different types of source information
    if source_field_name in ['source_chunk_id', 'chunk_id']:
        # Direct chunk ID comparison
        retrieved_chunk_ids = [
            chunk.get('chunk_id') for chunk in retrieved_chunks
        ]
        return source_info in retrieved_chunk_ids

    elif source_field_name in ['source_file', 'source_path']:
        # Path-based comparison
        retrieved_paths = [chunk.get('path') for chunk in retrieved_chunks]
        return source_info in retrieved_paths

    elif source_field_name in ['dataset_index', 'source_dataset_index']:
        # Dataset index comparison
        retrieved_indices = [
            chunk.get('dataset_index') for chunk in retrieved_chunks
        ]
        return source_info in retrieved_indices

    elif source_field_name == 'source_id':
        # Generic source ID - try to match against chunk IDs or paths
        retrieved_chunk_ids = [
            chunk.get('chunk_id') for chunk in retrieved_chunks
        ]
        retrieved_paths = [chunk.get('path') for chunk in retrieved_chunks]
        return (
            source_info in retrieved_chunk_ids
            or source_info in retrieved_paths
        )

    # If we can't determine the match, return None
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
        2048, description='Maximum number of tokens to generate.'
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

        # Check if source chunk was retrieved
        source_chunk_retrieved = check_source_chunk_retrieved(
            qa_pair, retrieval_info, use_rag
        )

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
            'source_chunk_retrieved': source_chunk_retrieved,  # Track source retrieval
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
        description='Advanced Question Grader with RAG and Chunk Logging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration File Usage:
    Recommended: Specify everything in YAML config file
        python rag_argonium_score_parallel_v2.py --config mcqa_config.yaml
    
    Alternative: Command-line arguments (for backward compatibility)
        python rag_argonium_score_parallel_v2.py questions.json --model llama --grader gpt41
    
    See sample_mcqa_config.yaml for a complete configuration example.
    Command-line arguments will override YAML settings when provided.
        """,
    )

    # Arguments (can be specified in config file)
    parser.add_argument(
        'questions_file',
        nargs='?',  # Make optional
        help='Path to the JSON file containing questions (can be specified in config)',
    )
    parser.add_argument(
        '--model',
        help='Model shortname from the model configuration file (can be specified in config)',
    )
    parser.add_argument(
        '--grader',
        help='Grader model shortname from the model configuration file (can be specified in config)',
    )

    # Configuration files
    parser.add_argument(
        '--config',
        help='MCQA configuration file (YAML) with all evaluation settings',
    )
    parser.add_argument(
        '--model-config',
        default='model_servers.yaml',
        help='Model configuration file (default: model_servers.yaml)',
    )

    # Override arguments (can override YAML settings)
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG completely - direct question answering',
    )
    parser.add_argument(
        '--parallel',
        type=int,
        help='Number of parallel workers',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output',
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

    return parser.parse_args()


def create_config_from_args(args) -> MCQAConfig:
    """Create MCQAConfig from command-line arguments and YAML file."""

    # Start with defaults
    if args.config and os.path.exists(args.config):
        print(f'Loading configuration from {args.config}')
        config = MCQAConfig.from_yaml(args.config)
    else:
        if args.config:
            print(
                f'Warning: Configuration file {args.config} not found, using defaults'
            )

        # Ensure required arguments are provided when no config file
        if not args.questions_file:
            raise ValueError(
                'questions_file must be provided either as argument or in config file'
            )
        if not args.model:
            raise ValueError(
                '--model must be provided either as argument or in config file'
            )
        if not args.grader:
            raise ValueError(
                '--grader must be provided either as argument or in config file'
            )

        # Create default configuration based on model argument
        if 'argo' in args.model.lower():
            generator_config = GeneratorConfig(generator_type='argo')
            generator_settings = ArgoGeneratorSettings(
                model=f'argo:{args.model}',
                base_url=os.getenv('ARGO_BASE_URL', 'http://localhost:56267'),
                api_key=os.getenv('ARGO_API_KEY', 'whatever+random'),
            )
        else:
            generator_config = GeneratorConfig(generator_type='vllm')
            generator_settings = VLLMGeneratorSettings(
                server=os.getenv('VLLM_SERVER', 'rbdgx1'),
                model=args.model,
                port=int(os.getenv('VLLM_PORT', '8000')),
                api_key=os.getenv('VLLM_API_KEY', 'CELS'),
            )

        config = MCQAConfig(
            questions_file=args.questions_file,
            model=ModelConfiguration(
                generator=generator_config,
                generator_settings=generator_settings,
                grader_shortname=args.grader,
                model_config_file=args.model_config or 'model_servers.yaml',
            ),
        )

    # Override with command-line arguments if provided
    if args.questions_file:
        config.questions_file = args.questions_file

    if args.grader:
        config.model.grader_shortname = args.grader

    if args.model_config:
        config.model.model_config_file = args.model_config

    if args.no_rag:
        config.rag.enabled = False

    if args.parallel is not None:
        config.processing.parallel_workers = args.parallel

    if args.verbose:
        config.processing.verbose = True

    if args.random is not None:
        config.processing.random_selection = args.random

    if args.seed is not None:
        config.processing.random_seed = args.seed

    if args.save_incorrect:
        config.output.save_incorrect = True

    return config


def convert_mcqa_retriever_to_distllm_config(
    retriever_config: RetrieverConfiguration,
) -> Dict[str, Any]:
    """Convert MCQA retriever configuration to distllm format."""
    from pathlib import Path

    # Convert FAISS config - only supporting v2 format
    faiss_config = retriever_config.faiss_config.model_dump()
    faiss_config['dataset_dir'] = Path(faiss_config['dataset_dir'])
    faiss_config['faiss_index_path'] = Path(faiss_config['faiss_index_path'])
    if faiss_config.get('dataset_chunk_paths'):
        faiss_config['dataset_chunk_paths'] = [
            Path(p) for p in faiss_config['dataset_chunk_paths']
        ]

    # Convert encoder config
    encoder_config = retriever_config.encoder_config.model_dump()

    # Convert pooler config
    pooler_config = retriever_config.pooler_config.model_dump()

    return {
        'faiss_config': faiss_config,
        'encoder_config': encoder_config,
        'pooler_config': pooler_config,
        'batch_size': retriever_config.batch_size,
    }


def create_metadata(
    config: MCQAConfig,
    questions: List[Dict],
    rag_config: Optional[Dict] = None,
    config_file_used: Optional[str] = None,
) -> Dict:
    """Create metadata for the evaluation run."""
    metadata = {
        'evaluation_metadata': {
            'script_version': '2.0',
            'script_name': 'rag_argonium_score_parallel_v2.py',
            'timestamp': datetime.now().isoformat(),
            'configuration': config.dict(),
            'config_file_used': config_file_used,
            'question_statistics': {
                'total_questions': len(questions),
                'selected_questions': len(questions)
                if not config.processing.random_selection
                else config.processing.random_selection,
                'question_format': config.processing.question_format,
                'random_seed': config.processing.random_seed,
            },
            'rag_configuration': {
                'enabled': config.rag.enabled,
                'use_context_field': config.rag.use_context_field,
                'retrieval_top_k': config.rag.retrieval_top_k,
                'retrieval_score_threshold': config.rag.retrieval_score_threshold,
                'rag_config_file': config.rag.rag_config_file,
                'rag_config_details': rag_config if rag_config else None,
            },
            'processing_configuration': {
                'parallel_workers': config.processing.parallel_workers,
                'verbose': config.processing.verbose,
                'chunk_logging_enabled': config.rag.chunk_logging_enabled,
            },
        }
    }
    return metadata


def main():
    """Main function."""
    args = parse_arguments()

    # Create configuration from arguments and YAML
    config = create_config_from_args(args)

    # Set random seed if provided
    if config.processing.random_seed is not None:
        random.seed(config.processing.random_seed)

    # Load questions
    with open(config.questions_file, 'r') as f:
        questions = json.load(f)

    # Randomly select questions if specified
    if config.processing.random_selection:
        if config.processing.random_selection < len(questions):
            questions = random.sample(
                questions, config.processing.random_selection
            )
            print(f'Randomly selected {len(questions)} questions')
        else:
            print(
                f'Requested {config.processing.random_selection} questions but only {len(questions)} available'
            )

    # Load model configurations
    grader_config = load_model_config(
        config.model.grader_shortname, config.model.model_config_file
    )

    # Load RAG configuration if provided
    rag_config = None
    if config.rag.rag_config_file:
        with open(config.rag.rag_config_file, 'r') as f:
            rag_config = yaml.safe_load(f)
        print(f'Using RAG configuration from {config.rag.rag_config_file}')
    elif config.rag.retriever_config:
        # Convert MCQA retriever config to distllm format
        retriever_config_dict = convert_mcqa_retriever_to_distllm_config(
            config.rag.retriever_config
        )
        print('Using inline retriever configuration')
    else:
        retriever_config_dict = None

    # Auto-detect question format
    question_format = config.processing.question_format
    if question_format == 'auto':
        question_format = detect_question_format(questions)

    # Create RAG model
    use_rag = config.rag.enabled
    if use_rag and rag_config:
        # Use provided RAG configuration from file
        rag_model_config = RetrievalAugmentedGenerationConfig(**rag_config)
        rag_model_config.use_rag = True
    elif use_rag and retriever_config_dict:
        # Use inline retriever configuration
        # Create generator config based on model settings
        if config.model.generator.generator_type == 'argo':
            if not isinstance(
                config.model.generator_settings, ArgoGeneratorSettings
            ):
                raise ValueError(
                    "Generator type is 'argo' but settings are not ArgoGeneratorSettings"
                )
            argo_settings = config.model.generator_settings
            generator_config = ArgoGeneratorConfig(
                model=argo_settings.model,
                base_url=argo_settings.base_url,
                api_key=argo_settings.api_key,
                temperature=argo_settings.temperature,
                max_tokens=argo_settings.max_tokens,
            )
        else:
            if not isinstance(
                config.model.generator_settings, VLLMGeneratorSettings
            ):
                raise ValueError(
                    "Generator type is 'vllm' but settings are not VLLMGeneratorSettings"
                )
            vllm_settings = config.model.generator_settings
            generator_config = VLLMGeneratorConfig(
                server=vllm_settings.server,
                port=vllm_settings.port,
                api_key=vllm_settings.api_key,
                model=vllm_settings.model,
                temperature=vllm_settings.temperature,
                max_tokens=vllm_settings.max_tokens,
            )

        # Create full RAG configuration
        rag_model_config = RetrievalAugmentedGenerationConfig(
            generator_config=generator_config,
            retriever_config=RetrieverConfig(**retriever_config_dict),
            verbose=config.processing.verbose,
            use_rag=True,
        )
    elif use_rag and not config.rag.use_context_field:
        # Create basic RAG model without retrieval
        if config.model.generator.generator_type == 'argo':
            if not isinstance(
                config.model.generator_settings, ArgoGeneratorSettings
            ):
                raise ValueError(
                    "Generator type is 'argo' but settings are not ArgoGeneratorSettings"
                )
            argo_settings = config.model.generator_settings
            generator_config = ArgoGeneratorConfig(
                model=argo_settings.model,
                base_url=argo_settings.base_url,
                api_key=argo_settings.api_key,
                temperature=argo_settings.temperature,
                max_tokens=argo_settings.max_tokens,
            )
        else:
            if not isinstance(
                config.model.generator_settings, VLLMGeneratorSettings
            ):
                raise ValueError(
                    "Generator type is 'vllm' but settings are not VLLMGeneratorSettings"
                )
            vllm_settings = config.model.generator_settings
            generator_config = VLLMGeneratorConfig(
                server=vllm_settings.server,
                port=vllm_settings.port,
                api_key=vllm_settings.api_key,
                model=vllm_settings.model,
                temperature=vllm_settings.temperature,
                max_tokens=vllm_settings.max_tokens,
            )

        rag_model_config = RetrievalAugmentedGenerationConfig(
            generator_config=generator_config,
            retriever_config=None,
            verbose=config.processing.verbose,
            use_rag=True,
        )
    else:
        # Create model for context field usage or no-RAG mode
        if config.model.generator.generator_type == 'argo':
            if not isinstance(
                config.model.generator_settings, ArgoGeneratorSettings
            ):
                raise ValueError(
                    "Generator type is 'argo' but settings are not ArgoGeneratorSettings"
                )
            argo_settings = config.model.generator_settings
            generator_config = ArgoGeneratorConfig(
                model=argo_settings.model,
                base_url=argo_settings.base_url,
                api_key=argo_settings.api_key,
                temperature=argo_settings.temperature,
                max_tokens=argo_settings.max_tokens,
            )
        else:
            if not isinstance(
                config.model.generator_settings, VLLMGeneratorSettings
            ):
                raise ValueError(
                    "Generator type is 'vllm' but settings are not VLLMGeneratorSettings"
                )
            vllm_settings = config.model.generator_settings
            generator_config = VLLMGeneratorConfig(
                server=vllm_settings.server,
                port=vllm_settings.port,
                api_key=vllm_settings.api_key,
                model=vllm_settings.model,
                temperature=vllm_settings.temperature,
                max_tokens=vllm_settings.max_tokens,
            )

        rag_model_config = RetrievalAugmentedGenerationConfig(
            generator_config=generator_config,
            retriever_config=None,
            verbose=config.processing.verbose,
            use_rag=use_rag,
        )

    # Create RAG model
    rag_model = rag_model_config.get_rag_model()

    # Print configuration summary
    print(f'Configuration Summary:')
    print(f'  Generator Type: {config.model.generator.generator_type}')
    print(f'  Generator Model: {config.model.generator_settings.model}')
    print(f'  Grader: {config.model.grader_shortname}')
    print(f'  RAG Enabled: {config.rag.enabled}')
    print(f'  Parallel Workers: {config.processing.parallel_workers}')
    print(f'  Question Format: {question_format}')
    print(f'  Total Questions: {len(questions)}')

    # Prepare items for parallel processing
    items = [(i, qa_pair) for i, qa_pair in enumerate(questions, 1)]
    results = []

    # Process questions
    start_time = time.time()
    print(
        f'\nProcessing {len(items)} questions with {"RAG" if use_rag else "DIRECT"} mode...'
    )
    if config.processing.parallel_workers > 1:
        print(
            f'Using {config.processing.parallel_workers} parallel workers...'
        )
    print(
        'This may take some time. Each model call has built-in retries and waiting.'
    )

    # Track progress
    completed = 0
    total = len(items)
    results_lock = threading.Lock()

    def update_progress(result):
        nonlocal completed
        with results_lock:
            completed += 1
            results.append(result)
            if not config.processing.verbose:
                print(
                    f'Progress: {completed}/{total} ({completed / total * 100:.1f}%)',
                    end='\r',
                )

    def process_item(item):
        try:
            result = process_question(
                item,
                rag_model,
                grader_config,
                question_format,
                config.processing.verbose,
                config.rag.use_context_field,
                config.rag.retrieval_top_k,
                config.rag.retrieval_score_threshold,
                use_rag,
            )
            update_progress(result)
            return result
        except Exception as e:
            error_result = {
                'question_id': item[0],
                'error': str(e),
                'skipped': True,
            }
            update_progress(error_result)
            return error_result

    # Execute processing
    if config.processing.parallel_workers == 1:
        for item in items:
            process_item(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.processing.parallel_workers
        ) as executor:
            futures = [executor.submit(process_item, item) for item in items]
            concurrent.futures.wait(futures)

    total_time = time.time() - start_time
    print(f'\nCompleted processing in {total_time:.2f} seconds')

    # Filter out skipped results for statistics
    processed_results = [r for r in results if not r.get('skipped', False)]

    if processed_results:
        # Calculate statistics
        all_scores = [r['score'] for r in processed_results]
        mc_results = [r for r in processed_results if r.get('format') == 'mc']
        qa_results = [r for r in processed_results if r.get('format') == 'qa']

        mc_scores = [r['score'] for r in mc_results]
        qa_scores = [r['score'] for r in qa_results]

        overall_accuracy = (
            sum(all_scores) / len(all_scores) if all_scores else 0
        )
        mc_accuracy = sum(mc_scores) / len(mc_scores) if mc_scores else None
        qa_accuracy = sum(qa_scores) / len(qa_scores) if qa_scores else None

        print(f'\n=== EVALUATION RESULTS ===')
        print(
            f'Overall Accuracy: {overall_accuracy:.3f} ({sum(all_scores)}/{len(all_scores)})'
        )
        if mc_accuracy is not None:
            print(
                f'MC Accuracy: {mc_accuracy:.3f} ({sum(mc_scores)}/{len(mc_scores)})'
            )
        if qa_accuracy is not None:
            print(
                f'QA Accuracy: {qa_accuracy:.3f} ({sum(qa_scores)}/{len(qa_scores)})'
            )

        # Source chunk retrieval statistics
        if use_rag:
            source_retrieved_results = [
                r
                for r in processed_results
                if r.get('source_chunk_retrieved') is not None
            ]
            if source_retrieved_results:
                source_retrieved_count = sum(
                    1
                    for r in source_retrieved_results
                    if r.get('source_chunk_retrieved')
                )
                source_retrieval_rate = source_retrieved_count / len(
                    source_retrieved_results
                )
                print(
                    f'Source Chunk Retrieval Rate: {source_retrieval_rate:.3f} ({source_retrieved_count}/{len(source_retrieved_results)})'
                )

        # Create metadata
        metadata = create_metadata(config, questions, rag_config, args.config)

        # Save results with metadata
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = config.model.generator_settings.model.replace(
            '/', '_'
        ).replace(':', '_')
        output_file = os.path.join(
            config.output.output_directory,
            f'{config.output.output_prefix}_{model_name}_{timestamp}.json',
        )

        # Create final output structure with metadata
        output_data = {
            'metadata': metadata,
            'results': results,
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f'Results saved to {output_file}')

        # Save configuration file to output directory for reproducibility
        config_output_file = os.path.join(
            config.output.output_directory,
            f'{config.output.output_prefix}_config_{model_name}_{timestamp}.yaml',
        )

        # Create output directory if it doesn't exist
        os.makedirs(config.output.output_directory, exist_ok=True)

        try:
            config.to_yaml(config_output_file)
            print(f'Configuration saved to {config_output_file}')
        except Exception as e:
            print(f'Warning: Could not save configuration file: {e}')

        # Save incorrect answers if requested
        if config.output.save_incorrect:
            incorrect_results = [
                r for r in processed_results if r['score'] < 1.0
            ]
            if incorrect_results:
                incorrect_file = os.path.join(
                    config.output.output_directory,
                    f'incorrect_answers_{model_name}_{timestamp}.json',
                )
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
            for chunk_list in sample_result['retrieval_info'][
                'retrieved_chunks'
            ]:
                for chunk in chunk_list:
                    print(f'  - Chunk ID: {chunk["chunk_id"]}')
                    print(f'    Score: {chunk["score"]:.4f}')
                    print(f'    Path: {chunk["path"]}')
                    print(f'    Text preview: {chunk["text"][:100]}...')
                    print()

    else:
        print('\nNo questions were successfully processed.')
        return


if __name__ == '__main__':
    main()
