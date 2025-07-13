#!/usr/bin/env python3

"""
RAG Argonium Advanced Question Grader v1.0 (Parallel)

Usage:
    python rag_argonium_score_parallel_v1.py <questions_file.json> --model <model_shortname> --grader <grader_shortname> [--config <config_file>] [--rag-config <rag_config_file>] [--parallel <num_workers>] [--format auto|mc|qa] [--random <num_questions>] [--seed <random_seed>] [--save-incorrect] [--use-context-field] [--retrieval-top-k <k>] [--retrieval-score-threshold <threshold>]

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
    - use-context-field: Use the "text" field from JSON as context instead of retrieval (optional)
    - retrieval-top-k: Number of documents to retrieve (default: 5)
    - retrieval-score-threshold: Minimum retrieval score threshold (default: 0.0)

Examples:
    python rag_argonium_score_parallel_v1.py frg_mc_100.json --model llama --grader gpt41 --parallel 4
    python rag_argonium_score_parallel_v1.py frg_mc_100.json --model llama --grader gpt41 --rag-config rag_config.yaml
    python rag_argonium_score_parallel_v1.py frg_mc_100.json --model llama --grader gpt41 --use-context-field
    python rag_argonium_score_parallel_v1.py frg_mc_100.json --model argo --grader gpt41 --rag-config argo_config.yaml

The script:
1) Uses RAG (retrieval-augmented generation) to enhance question answering
2) Supports both VLLM and Argo generators
3) Can use external retrieval or context from the JSON file
4) Uses the specified MODEL to generate an answer to each question with RAG
5) Uses the specified GRADER to evaluate the model's answer against the reference answer
6) Reports detailed accuracy metrics and exports results
7) Processes multiple questions in parallel when --parallel > 1
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
)
from distllm.rag.search import Retriever, RetrieverConfig
from distllm.utils import BaseConfig

# Load environment variables
load_dotenv()

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
        with open(yaml_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        for server in config['servers']:
            if server['shortname'] == model_shortname:
                api_key = server['openai_api_key']
                if api_key.startswith('${') and api_key.endswith('}'):
                    env_var = api_key[2:-1]
                    api_key = os.environ.get(env_var, '')
                    if not api_key:
                        print(f'Error: Environment variable {env_var} not set')
                        sys.exit(1)

                return {
                    'api_key': api_key,
                    'api_base': server['openai_api_base'],
                    'model_name': server['openai_model'],
                }

        print(
            f"Error: Model '{model_shortname}' not found in model_servers.yaml"
        )
        print(
            'Available models:',
            ', '.join([s['shortname'] for s in config['servers']]),
        )
        sys.exit(1)

    except FileNotFoundError:
        print(f'Error: model_servers.yaml not found at {yaml_path}')
        sys.exit(1)
    except Exception as e:
        print(f'Error loading model configuration: {e}')
        sys.exit(1)


def detect_question_format(questions):
    """Detect whether the questions are in multiple-choice or free-form QA format."""
    mc_count = 0
    qa_count = 0

    mc_patterns = [
        r'(?:^|\n)\s*([A-E])[.):]\s',
        r'(?:^|\n)\s*([1-5])[.):]\s',
        r'\n\s*Option\s+[A-E][.:)]',
        r'\n\s*Choice\s+[A-E][.:)]',
        r'\n\s*Answer\s+[A-E][.:)]',
    ]

    for qa_pair in questions:
        question = qa_pair.get('question', '')
        is_mc = any(
            re.search(pattern, question, re.IGNORECASE)
            for pattern in mc_patterns
        )

        if is_mc:
            mc_count += 1
        else:
            qa_count += 1

    if len(questions) > 0:
        mc_ratio = mc_count / len(questions)
        return 'mc' if mc_ratio > 0.6 else 'qa'
    else:
        return 'qa'


def detect_choice_identifier_type(question_text):
    """Detect whether a question uses letter (A, B, C) or number (1, 2, 3) identifiers."""
    has_letter_option = bool(
        re.search(r'(?:^|\n)\s*([A-E])[.):,]\s', question_text)
    )
    has_number_option = bool(
        re.search(r'(?:^|\n)\s*([1-5])[.):,]\s', question_text)
    )

    if has_number_option:
        return 'number'
    elif has_letter_option:
        return 'letter'
    else:
        return 'letter'


def extract_choice_identifier(answer_text):
    """Extract the choice identifier from an answer text."""
    if not answer_text:
        return (None, None)

    # Multiple patterns to identify choice identifiers
    letter_patterns = [
        r'(?:^|\n)\s*([A-E])[.):]\s',
        r'(?:option|answer|choice)\s+([A-E])\b',
        r'(?:the\s+(?:correct\s+)?answer\s+is\s+([A-E]))|(?:\b([A-E])\s+is\s+(?:the\s+)?correct)',
    ]

    number_patterns = [
        r'(?:^|\n)\s*([1-5])[.):]\s',
        r'(?:option|answer|choice)\s+([1-5])\b',
        r'(?:the\s+(?:correct\s+)?answer\s+is\s+([1-5]))|(?:\b([1-5])\s+is\s+(?:the\s+)?correct)',
    ]

    # Try letter patterns first
    for pattern in letter_patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group:
                    return ('letter', group.upper())

    # Try number patterns
    for pattern in number_patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group:
                    return ('number', group)

    # Look at the beginning of the response
    first_line = answer_text.split('\n')[0].strip()

    letter_start = re.match(
        r'^([A-E])(?:[.):,]|\s|$)', first_line, re.IGNORECASE
    )
    if letter_start:
        return ('letter', letter_start.group(1).upper())

    number_start = re.match(
        r'^([1-5])(?:[.):,]|\s|$)', first_line, re.IGNORECASE
    )
    if number_start:
        return ('number', number_start.group(1))

    # For short answers, scan for valid options
    if len(answer_text) < 100:
        standalone_letter = re.search(
            r'\b([A-E])\b', answer_text, re.IGNORECASE
        )
        if standalone_letter:
            return ('letter', standalone_letter.group(1).upper())

        standalone_number = re.search(r'\b([1-5])\b', answer_text)
        if standalone_number:
            return ('number', standalone_number.group(1))

    return (None, None)


# -----------------------------------------------------------------------------
# RAG Components (adapted from chat.py and chat_argoproxy.py)
# -----------------------------------------------------------------------------


class PromptTemplate:
    """Base class for prompt templates."""

    def preprocess(
        self,
        texts: List[str],
        contexts: Optional[List[List[str]]] = None,
        scores: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Preprocess the texts before sending to the model."""
        raise NotImplementedError('Subclasses should implement this method')


class RAGPromptTemplate(PromptTemplate):
    """RAG prompt template for multiple choice questions with retrieval context."""

    def __init__(self, question_format: str = 'mc'):
        self.question_format = question_format

    def preprocess(
        self,
        texts: List[str],
        contexts: Optional[List[List[str]]] = None,
        scores: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Preprocess the texts with retrieval context."""
        if not texts:
            return ['']

        prompts = []
        for i, text in enumerate(texts):
            # Build the base prompt
            if self.question_format == 'mc':
                id_type_in_question = detect_choice_identifier_type(text)
                label_format = (
                    'number (1, 2, 3, etc.)'
                    if id_type_in_question == 'number'
                    else 'letter (A, B, C, etc.)'
                )

                prompt = f"""You are an expert at multiple-choice questions. Think through the question step by step, but provide only a concise final answer.

Your response should contain ONLY:
1. The correct {label_format}
2. A brief explanation (2-3 sentences) of why this choice is correct.

Do not include the question restatement or list of alternatives in your response.

"""
            else:
                prompt = """You are an expert at answering questions in various fields including science, mathematics, physics, computer science, and more. Provide concise, accurate, and thorough answers based on your knowledge. Focus on being factually correct.

"""

            # Add context if available
            if contexts and i < len(contexts) and contexts[i]:
                prompt += '\n[Retrieved Context]\n'
                for j, doc in enumerate(contexts[i]):
                    score_info = (
                        f' (Score: {scores[i][j]:.3f})'
                        if scores and i < len(scores) and j < len(scores[i])
                        else ''
                    )
                    prompt += f'Context {j + 1}{score_info}: {doc}\n'
                prompt += '\n'

            # Add the question
            if self.question_format == 'mc':
                prompt += f'Please answer this multiple-choice question using the provided context to help you:\n\n{text}'
            else:
                prompt += f'Please answer the following question using the provided context to help you:\n\n{text}'

            prompts.append(prompt)

        return prompts


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
        16384, description='The maximum number of tokens to generate.'
    )

    def get_generator(self) -> 'VLLMGenerator':
        """Get the vLLM generator."""
        return VLLMGenerator(config=self)


class VLLMGenerator:
    """A generator that calls a local or remote vLLM server."""

    def __init__(self, config: VLLMGeneratorConfig) -> None:
        self.server = config.server
        self.port = config.port
        self.api_key = config.api_key
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a prompt to the local vLLM server and return the completion."""
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        url = f'http://{self.server}.cels.anl.gov:{self.port}/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': temp_to_use,
            'max_tokens': tokens_to_use,
        }

        response = requests.post(
            url, headers=headers, data=json.dumps(payload)
        )
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
        else:
            print(f'Error: {response.status_code}')
            result = response.text

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
        16384, description='The maximum number of tokens to generate.'
    )

    def get_generator(self) -> 'ArgoGenerator':
        """Get the Argo generator."""
        return ArgoGenerator(config=self)


class ArgoGenerator:
    """A generator that calls the Argo proxy using OpenAI client."""

    def __init__(self, config: ArgoGeneratorConfig) -> None:
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=f'{config.base_url}/v1',
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a prompt to the Argo proxy and return the completion."""
        temp_to_use = self.temperature if temperature is None else temperature
        tokens_to_use = self.max_tokens if max_tokens is None else max_tokens

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp_to_use,
                max_tokens=tokens_to_use,
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f'Error calling Argo proxy: {e}')
            result = f'Error: {e!s}'

        return result


class RagGenerator:
    """RAG generator for generating responses to queries."""

    def __init__(
        self,
        generator: Union[VLLMGenerator, ArgoGenerator],
        retriever: Optional[Retriever] = None,
        verbose: bool = False,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.verbose = verbose

    def generate(
        self,
        texts: Union[str, List[str]],
        prompt_template: Optional[PromptTemplate] = None,
        retrieval_top_k: int = 5,
        retrieval_score_threshold: float = 0.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        contexts: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """Generate responses to the given queries."""
        if isinstance(texts, str):
            texts = [texts]

        # Use the identity prompt template if none is provided
        if prompt_template is None:
            prompt_template = IdentityPromptTemplate(
                IdentityPromptTemplateConfig()
            )

        # Default: no context
        retrieved_contexts, scores = None, None

        # Use provided contexts (from JSON text field) or retrieve if retriever is available
        if contexts is not None:
            retrieved_contexts = contexts
            scores = None
        elif self.retriever is not None:
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

        # Build the final prompts
        prompts = prompt_template.preprocess(texts, retrieved_contexts, scores)

        # If verbose, print contexts
        if self.verbose and retrieved_contexts and retrieved_contexts[0]:
            print('Retrieved contexts:')
            for i, context in enumerate(retrieved_contexts[0]):
                print(f'  Context {i + 1}: {context[:200]}...')

        # Generate responses
        responses = []
        for prompt in prompts:
            result = self.generator.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            responses.append(result)

        return responses


class RetrievalAugmentedGenerationConfig(BaseConfig):
    """Configuration for the retrieval-augmented generation model."""

    generator_config: Union[VLLMGeneratorConfig, ArgoGeneratorConfig] = Field(
        ..., description='Settings for the generator (VLLM or Argo)'
    )
    retriever_config: Optional[RetrieverConfig] = Field(
        None, description='Settings for the retriever'
    )
    verbose: bool = Field(
        default=False, description='Whether to print retrieved contexts.'
    )

    def get_rag_model(self) -> RagGenerator:
        """Instantiate the RAG model."""
        # Initialize the generator (either VLLM or Argo)
        if isinstance(self.generator_config, VLLMGeneratorConfig):
            generator = VLLMGenerator(self.generator_config)
        elif isinstance(self.generator_config, ArgoGeneratorConfig):
            generator = ArgoGenerator(self.generator_config)
        else:
            raise ValueError(
                f'Unsupported generator config type: {type(self.generator_config)}'
            )

        # Initialize the retriever
        retriever = None
        if self.retriever_config is not None:
            retriever = self.retriever_config.get_retriever()

        return RagGenerator(
            generator=generator, retriever=retriever, verbose=self.verbose
        )


# -----------------------------------------------------------------------------
# RAG Answer Generation and Evaluation
# -----------------------------------------------------------------------------


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    giveup=lambda e: 'Invalid authentication' in str(e),
    max_time=300,
)
def generate_rag_answer(
    question: str,
    rag_model: RagGenerator,
    question_format: str = 'auto',
    context_text: Optional[str] = None,
    retrieval_top_k: int = 5,
    retrieval_score_threshold: float = 0.0,
) -> str:
    """Generate an answer to a question using RAG."""
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

    # Create appropriate prompt template
    prompt_template = RAGPromptTemplate(question_format=actual_format)

    # Prepare contexts if using context field
    contexts = None
    if context_text:
        contexts = [[context_text]]

    # Add a small random delay to avoid overwhelming the server
    jitter = random.uniform(0.1, 1.0)
    time.sleep(jitter)

    try:
        # Generate answer using RAG
        responses = rag_model.generate(
            texts=[question],
            prompt_template=prompt_template,
            retrieval_top_k=retrieval_top_k,
            retrieval_score_threshold=retrieval_score_threshold,
            contexts=contexts,
        )
        return responses[0]
    except Exception as e:
        print(f'Error in generate_rag_answer (will retry): {str(e)}')
        raise


def _evaluate_answer_with_retry(
    question, reference_answer, model_answer, config, question_format='auto'
):
    """Wrapper for evaluate_answer with custom retry logic for JSON parsing issues."""
    try:
        return _evaluate_answer_core(
            question,
            reference_answer,
            model_answer,
            config,
            question_format,
            retry_count=0,
        )
    except Exception as e:
        if 'JSON parsing failed, retrying once' in str(e):
            try:
                print('Retrying evaluation due to JSON parsing issue...')
                return _evaluate_answer_core(
                    question,
                    reference_answer,
                    model_answer,
                    config,
                    question_format,
                    retry_count=1,
                )
            except Exception as e2:
                print(f'Evaluation failed after retry: {str(e2)}')
                return {
                    'score': 0,
                    'confidence': 0.1,
                    'match': False,
                    'format': question_format
                    if question_format != 'auto'
                    else 'qa',
                    'reasoning': 'Failed to evaluate after retry',
                    'parse_error': True,
                }
        else:
            raise


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
    """Evaluate a model's answer against the reference answer using the specified grader model."""
    api_key = config['api_key']
    api_base = config['api_base']
    model_name = config['model_name']

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

    # Based on format, determine appropriate evaluation approach
    if actual_format == 'mc':
        id_type_in_question = detect_choice_identifier_type(question)
        option_type = 'number' if id_type_in_question == 'number' else 'letter'
        option_examples = (
            '(1, 2, 3, etc.)' if option_type == 'number' else '(A, B, C, etc.)'
        )

        system_message = (
            'You are an expert evaluator for multiple-choice questions. '
            "Your task is to determine if a model's answer matches the correct answer, "
            f'focusing specifically on which option {option_examples} was selected.'
        )

        # Extract choice identifiers for additional context
        ref_id_type, ref_id = extract_choice_identifier(reference_answer)
        model_id_type, model_id = extract_choice_identifier(model_answer)

        analysis_context = ''
        if ref_id:
            analysis_context += (
                f'\nPRE-ANALYSIS CONTEXT (for reference only):\n'
            )
            analysis_context += f'- Detected correct choice identifier: {ref_id} ({ref_id_type})\n'
        if model_id:
            analysis_context += f'- Detected model choice identifier: {model_id} ({model_id_type})\n'
        if analysis_context:
            analysis_context += (
                f'- Question uses {id_type_in_question} format\n'
            )
            analysis_context += f'\nNOTE: This pre-analysis is for context only. Please make your own independent evaluation.\n'

        user_message = f"""
Please evaluate whether the model selected the correct answer choice for this multiple-choice question.

QUESTION:
{question}

CORRECT ANSWER (with explanation):
{reference_answer}

MODEL'S ANSWER:
{model_answer}
{analysis_context}
EVALUATION STEPS:
1. First, identify the correct {option_type} from the reference answer {option_examples}.
2. Next, identify which {option_type} the model selected in its answer.
3. Check if these {option_type}s match.
4. IMPORTANT: Even if the {option_type}s don't exactly match (e.g., model says "A" but correct is "1"), check if the CONTENT of the model's reasoning matches the CONTENT of the correct option from the question.
5. Also handle cases where the model gives a number when the question uses letters, or vice versa (A=1, B=2, C=3, D=4, E=5).

SPECIAL INSTRUCTIONS:
- If the model says "A" and the correct answer is "1", check if A corresponds to option 1 in the question
- If the model says "3" and the correct answer is "C", check if option 3 corresponds to choice C in the question  
- The model should be considered correct if it chose the right CONTENT, even if using a different numbering/lettering system
- Consider both the choice identifier AND the reasoning content when determining correctness

Please respond with a JSON object in the following format:
{{
  "correct_choice": "The {option_type} of the correct answer {option_examples}",
  "model_choice": "The {option_type} the model selected (or equivalent if different format)",
  "match": true/false (Do the choices match, considering content and format conversion?),
  "confidence": A number from 0 to 1 representing your confidence in your evaluation,
  "score": 1 if match is true, 0 if match is false,
  "content_consistent": true/false (Does the model's reasoning match the correct option content?),
  "reasoning": "Brief explanation of your evaluation"
}}
"""
    else:  # Free-form QA format
        system_message = (
            'You are an expert evaluator for question answering. '
            "Your task is to determine if a model's answer to a question "
            'is factually correct and sufficiently addresses the question.'
        )

        user_message = f"""
Please evaluate whether the model's answer correctly addresses this question.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

MODEL'S ANSWER:
{model_answer}

EVALUATION STEPS:
1. Carefully read the reference answer and understand the key information required.
2. Read the model's answer and evaluate its factual correctness and completeness.
3. Determine if the model's answer contains the essential information found in the reference answer.

Please respond with a JSON object in the following format:
{{
  "match": true/false (Is the model's answer correct?),
  "confidence": A number from 0 to 1 representing your confidence in your evaluation,
  "score": A number from 0 to 1 representing the quality of the answer (1=perfect, 0=completely wrong),
  "reasoning": "A brief explanation of your evaluation"
}}
"""

    try:
        # Add a small random delay to avoid overwhelming the grader
        jitter = random.uniform(0.1, 1.0)
        time.sleep(jitter)

        # Get cached client instance (thread-safe)
        client = get_openai_client(api_key, api_base, timeout=120.0)

        # Check if we need to skip temperature (for reasoning models)
        skip_temperature = any(
            name in model_name.lower() for name in ['o3', 'o4-mini', 'o4mini']
        )

        # Prepare parameters
        params = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': user_message},
            ],
        }

        if not skip_temperature:
            params['temperature'] = 0.0

        # Call the API
        response = client.chat.completions.create(**params)

        if hasattr(response, 'choices'):
            evaluation_text = response.choices[0].message.content.strip()
        else:
            evaluation_text = response['choices'][0]['message'][
                'content'
            ].strip()

        # Parse JSON response
        try:
            evaluation = json.loads(evaluation_text)

            if actual_format == 'mc':
                # Handle both old and new JSON field names
                if 'correct_choice' in evaluation:
                    correct_choice = evaluation['correct_choice']
                elif 'correct_letter' in evaluation:
                    correct_choice = evaluation['correct_letter']
                else:
                    correct_choice = None

                if 'model_choice' in evaluation:
                    model_choice = evaluation['model_choice']
                elif 'model_letter' in evaluation:
                    model_choice = evaluation['model_letter']
                else:
                    model_choice = None

                # Ensure required fields exist
                if 'score' not in evaluation:
                    if 'match' in evaluation and isinstance(
                        evaluation['match'], bool
                    ):
                        evaluation['score'] = 1 if evaluation['match'] else 0
                    else:
                        evaluation['score'] = 0

                # Standardize field names
                if (
                    'correct_choice' in evaluation
                    and 'correct_letter' not in evaluation
                ):
                    evaluation['correct_letter'] = evaluation['correct_choice']
                if (
                    'model_choice' in evaluation
                    and 'model_letter' not in evaluation
                ):
                    evaluation['model_letter'] = evaluation['model_choice']

                # Store extracted identifiers
                if ref_id:
                    evaluation['extracted_correct_choice'] = ref_id
                    evaluation['correct_choice_type'] = ref_id_type
                if model_id:
                    evaluation['extracted_model_choice'] = model_id
                    evaluation['model_choice_type'] = model_id_type

                evaluation['format'] = actual_format
            else:
                # Ensure required fields exist for QA
                if 'score' not in evaluation and 'match' in evaluation:
                    evaluation['score'] = 1.0 if evaluation['match'] else 0.0
                evaluation['format'] = 'qa'

            return evaluation

        except json.JSONDecodeError:
            print(f'Warning: Grader returned invalid JSON: {evaluation_text}')

            # Fallback parsing
            score_match = re.search(
                r'score["\s:]+([01](?:\.\d+)?)', evaluation_text
            )
            match_text = re.search(
                r'match["\s:]+(\w+)', evaluation_text, re.IGNORECASE
            )

            if score_match:
                score = float(score_match.group(1))
                match_value = score > 0.5

                if match_text:
                    match_str = match_text.group(1).lower()
                    if match_str in ['true', 'yes', '1']:
                        match_value = True
                    elif match_str in ['false', 'no', '0']:
                        match_value = False

                return {
                    'score': score,
                    'confidence': 0.5,
                    'match': match_value,
                    'format': actual_format,
                }
            else:
                if retry_count > 0:
                    print(
                        f'Warning: Could not parse evaluation JSON after retry: {evaluation_text}'
                    )
                    return {
                        'score': 0,
                        'confidence': 0.1,
                        'match': False,
                        'format': actual_format,
                        'reasoning': 'Failed to parse grader response',
                        'parse_error': True,
                    }
                else:
                    print(
                        f'Warning: Could not parse evaluation JSON, will retry once: {evaluation_text}'
                    )
                    raise Exception('JSON parsing failed, retrying once')

    except Exception as e:
        print(f'Error in _evaluate_answer_core (will retry): {str(e)}')
        raise


def evaluate_answer(
    question, reference_answer, model_answer, config, question_format='auto'
):
    """Public interface for evaluating answers."""
    return _evaluate_answer_with_retry(
        question, reference_answer, model_answer, config, question_format
    )


def process_question(
    item,
    rag_model: RagGenerator,
    grader_config: Dict[str, Any],
    question_format: str = 'auto',
    verbose: bool = False,
    use_context_field: bool = False,
    retrieval_top_k: int = 5,
    retrieval_score_threshold: float = 0.0,
):
    """Process a single question - generate answer with RAG and evaluate it."""
    i, qa_pair = item
    question = qa_pair.get('question', '')
    reference_answer = qa_pair.get('answer', '')
    context_text = qa_pair.get('text', '') if use_context_field else None

    if not question or not reference_answer:
        return {
            'question_id': i,
            'error': 'Missing question or answer',
            'skipped': True,
        }

    try:
        if verbose:
            print(f'\nProcessing question {i}...')

        # Generate model answer with RAG
        start_time = time.time()
        model_answer = generate_rag_answer(
            question=question,
            rag_model=rag_model,
            question_format=question_format,
            context_text=context_text,
            retrieval_top_k=retrieval_top_k,
            retrieval_score_threshold=retrieval_score_threshold,
        )
        model_time = time.time() - start_time

        if verbose:
            print(f'\n--- RAG RESPONSE FOR QUESTION {i} ---')
            print(model_answer)
            print('--- END RAG RESPONSE ---')
            print(f'Generated answer for question {i} in {model_time:.2f}s')

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
            'skipped': False,
            'used_context_field': use_context_field,
        }

        # Add context information if used
        if use_context_field and context_text:
            result['context_text'] = context_text

        # Print progress if verbose
        if verbose:
            if format_type == 'mc':
                correct_letter = evaluation.get('correct_letter', '?')
                model_letter = evaluation.get('model_letter', '?')
                confidence = evaluation.get('confidence', 0)
                id_type = evaluation.get('identifier_type', 'letter')
                id_name = 'number' if id_type == 'number' else 'letter'

                print(
                    f'Q{i} (MC) Result: {"✓" if score == 1 else "✗"} Score: {score}/1 '
                    f'Confidence: {confidence:.2f} (Model chose {id_name}: {model_letter}, '
                    f'Correct {id_name}: {correct_letter})'
                )
            else:
                confidence = evaluation.get('confidence', 0)
                score_str = (
                    f'{score:.2f}'
                    if isinstance(score, float) and score < 1
                    else f'{int(score)}/1'
                )
                print(
                    f'Q{i} (QA) Result: {"✓" if score >= 0.5 else "✗"} Score: {score_str} '
                    f'Confidence: {confidence:.2f}'
                )

        return result

    except Exception as e:
        print(f'\nUnhandled error processing question {i}: {str(e)}')
        return {
            'question_id': i,
            'question': question,
            'reference_answer': reference_answer,
            'error': str(e),
            'skipped': True,
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RAG Advanced Question Grader (Parallel)'
    )
    parser.add_argument(
        'questions_file', help='JSON file with questions and answers'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model shortname from model_servers.yaml to test',
    )
    parser.add_argument(
        '--grader',
        required=True,
        help='Model shortname from model_servers.yaml to use for grading',
    )
    parser.add_argument(
        '--config',
        default='model_servers.yaml',
        help='Configuration file for model settings',
    )
    parser.add_argument(
        '--rag-config',
        help='RAG configuration file (YAML) for retrieval settings',
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of concurrent workers (default: 1)',
    )
    parser.add_argument(
        '--format',
        choices=['auto', 'mc', 'qa'],
        default='auto',
        help='Format of questions',
    )
    parser.add_argument(
        '--output', help='Output JSON file for detailed results'
    )
    parser.add_argument(
        '--verbose', action='store_true', help='Print verbose output'
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
        help='Save incorrectly answered questions',
    )
    parser.add_argument(
        '--use-context-field',
        action='store_true',
        help="Use the 'text' field from JSON as context instead of retrieval",
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
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load model configs
    model_config = load_model_config(args.model, args.config)
    grader_config = load_model_config(args.grader, args.config)

    print(f'Testing model: {args.model} ({model_config["model_name"]})')
    print(f'Grading with: {args.grader} ({grader_config["model_name"]})')
    print(f'Parallel workers: {args.parallel}')
    print(f'Using context field: {args.use_context_field}')
    print(f'Retrieval top-k: {args.retrieval_top_k}')
    print(f'Retrieval score threshold: {args.retrieval_score_threshold}')

    # Load RAG configuration if provided
    rag_model = None
    if args.rag_config:
        try:
            with open(args.rag_config, 'r') as f:
                rag_config_dict = yaml.safe_load(f)

            # Create RAG configuration from YAML
            rag_config = RetrievalAugmentedGenerationConfig(
                **rag_config_dict['rag_configs']
            )
            rag_model = rag_config.get_rag_model()
            print(f'Loaded RAG config from: {args.rag_config}')
        except Exception as e:
            print(f'Error loading RAG config: {e}')
            sys.exit(1)
    else:
        # Create a basic RAG model without retrieval
        if args.model == 'argo':
            generator_config = ArgoGeneratorConfig()
        else:
            # Assume VLLM for other models
            generator_config = VLLMGeneratorConfig(
                server=model_config['api_base'].split('//')[1].split('.')[0],
                port=int(
                    model_config['api_base'].split(':')[-1].split('/')[0]
                ),
                api_key=model_config['api_key'],
                model=model_config['model_name'],
            )

        rag_config = RetrievalAugmentedGenerationConfig(
            generator_config=generator_config,
            retriever_config=None,
            verbose=args.verbose,
        )
        rag_model = rag_config.get_rag_model()
        print('Using basic RAG model without retrieval')

    # Load question-answer pairs
    try:
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading questions file: {e}')
        sys.exit(1)

    # Randomly select questions if specified
    original_count = len(data)
    if args.random:
        if args.random <= 0:
            print(
                f'Error: --random must be a positive number, got {args.random}'
            )
            sys.exit(1)
        elif args.random < len(data):
            print(
                f'Randomly selecting {args.random} questions from {original_count} total questions...'
            )
            if args.seed is not None:
                random.seed(args.seed)
                print(f'Using random seed: {args.seed}')
            data = random.sample(data, args.random)
            print(f'Selected {len(data)} questions.')
        else:
            print(
                f'Requested {args.random} questions, but dataset only has {original_count}. Using all questions.'
            )

    # Determine question format
    question_format = args.format
    if question_format == 'auto':
        detected_format = detect_question_format(data)
        print(f'Auto-detected question format: {detected_format}')
        question_format = detected_format
    else:
        print(f'Using specified question format: {question_format}')

    # Check if context field is available when requested
    if args.use_context_field:
        has_context = any('text' in item for item in data)
        if not has_context:
            print(
                "Warning: --use-context-field specified but no 'text' field found in questions"
            )
            print(
                'Available fields in first question:',
                list(data[0].keys()) if data else 'None',
            )

    # Prepare items for parallel processing
    items = [(i, qa_pair) for i, qa_pair in enumerate(data, 1)]
    results = []

    # Process questions
    start_time = time.time()
    print(f'\nProcessing {len(items)} questions with RAG...')
    if args.parallel > 1:
        print(f'Using {args.parallel} parallel workers...')
    print(
        'This may take some time. Each model call has built-in retries and waiting.'
    )

    # Track progress
    completed = 0
    total = len(items)
    results_lock = threading.Lock()

    if args.parallel > 1:
        # Parallel processing with progress bar
        progress_bar = None
        if not args.verbose:
            progress_bar = tqdm(total=total, desc='Processing questions')

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.parallel
        ) as executor:
            futures = [
                executor.submit(
                    process_question,
                    item,
                    rag_model,
                    grader_config,
                    question_format,
                    args.verbose,
                    args.use_context_field,
                    args.retrieval_top_k,
                    args.retrieval_score_threshold,
                )
                for item in items
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    with results_lock:
                        results.append(result)
                        completed += 1
                        current_completed = completed
                        current_results = results.copy()

                    if progress_bar:
                        progress_bar.update(1)

                    if not args.verbose and not progress_bar:
                        valid_results = [
                            r
                            for r in current_results
                            if not r.get('skipped', False)
                        ]
                        if valid_results:
                            total_score = sum(
                                r.get('score', 0) for r in valid_results
                            )
                            accuracy = total_score / len(valid_results)
                            print(
                                f'Completed {current_completed}/{total} questions | '
                                f'Running accuracy: {accuracy:.1%}',
                                end='\r',
                            )

                except Exception as e:
                    print(f'\nUnhandled error in worker: {e}')
                    with results_lock:
                        results.append(
                            {
                                'question_id': 'unknown',
                                'error': str(e),
                                'skipped': True,
                            }
                        )
                        completed += 1

                    if progress_bar:
                        progress_bar.update(1)

            if progress_bar:
                progress_bar.close()
    else:
        # Sequential processing
        total_score = 0
        valid_count = 0

        for item in items:
            result = process_question(
                item,
                rag_model,
                grader_config,
                question_format,
                args.verbose,
                args.use_context_field,
                args.retrieval_top_k,
                args.retrieval_score_threshold,
            )
            results.append(result)
            completed += 1

            if not result.get('skipped', False):
                total_score += result.get('score', 0)
                valid_count += 1

            if not args.verbose:
                if valid_count > 0:
                    accuracy = total_score / valid_count
                    print(
                        f'Completed {completed}/{total} questions | '
                        f'Running accuracy: {accuracy:.1%}',
                        end='\r',
                    )
                else:
                    print(f'Completed {completed}/{total} questions', end='\r')

    if not args.verbose:
        print()

    # Sort results by question_id
    results.sort(key=lambda x: x.get('question_id', 0))

    # Calculate statistics
    total_time = time.time() - start_time
    valid_results = [r for r in results if not r.get('skipped', False)]

    mc_results = [
        r for r in valid_results if r.get('format', question_format) == 'mc'
    ]
    qa_results = [
        r for r in valid_results if r.get('format', question_format) == 'qa'
    ]

    mc_scores = [r.get('score', 0) for r in mc_results]
    qa_scores = [r.get('score', 0) for r in qa_results]
    all_scores = [r.get('score', 0) for r in valid_results]

    # Fix inconsistencies in MC results
    inconsistencies_fixed = 0
    for result in mc_results:
        evaluation = result.get('evaluation', {})
        correct_letter = evaluation.get('correct_letter', '?')
        model_letter = evaluation.get('model_letter', '?')

        letters_match = (
            model_letter == correct_letter
            if model_letter and correct_letter
            else None
        )
        score = result.get('score', 0)

        if letters_match is not None and (
            (letters_match and score == 0)
            or (not letters_match and score == 1)
        ):
            inconsistencies_fixed += 1
            score = 1 if letters_match else 0
            result['score'] = score
            if 'evaluation' in result:
                result['evaluation']['score'] = score

    if inconsistencies_fixed > 0:
        mc_scores = [r.get('score', 0) for r in mc_results]
        all_scores = [r.get('score', 0) for r in valid_results]

    # Display results
    print('\n' + '=' * 60)
    print('RAG EVALUATION RESULTS')
    print('=' * 60)

    # Overall statistics
    if all_scores:
        overall_accuracy = sum(all_scores) / len(all_scores)
        print(
            f'Overall accuracy: {overall_accuracy:.2%} ({sum(all_scores):.1f}/{len(all_scores)})'
        )
    else:
        print('No valid questions processed.')

    # Format-specific statistics
    if mc_scores:
        mc_accuracy = sum(mc_scores) / len(mc_scores)
        print(
            f'Multiple-choice questions: {mc_accuracy:.2%} accuracy ({sum(mc_scores)}/{len(mc_scores)})'
        )

        mc_confidences = [
            r.get('evaluation', {}).get('confidence', 0) for r in mc_results
        ]
        avg_mc_confidence = (
            sum(mc_confidences) / len(mc_confidences) if mc_confidences else 0
        )
        print(f'Average MC confidence: {avg_mc_confidence:.2f}')

    if qa_scores:
        qa_accuracy = sum(qa_scores) / len(qa_scores)
        print(
            f'Free-form QA questions: {qa_accuracy:.2%} score ({sum(qa_scores):.1f}/{len(qa_scores)})'
        )

        qa_confidences = [
            r.get('evaluation', {}).get('confidence', 0) for r in qa_results
        ]
        avg_qa_confidence = (
            sum(qa_confidences) / len(qa_confidences) if qa_confidences else 0
        )
        print(f'Average QA confidence: {avg_qa_confidence:.2f}')

    print(f'Total processing time: {total_time:.2f} seconds')

    # Report any inconsistencies fixed
    if inconsistencies_fixed > 0:
        print(f'Fixed {inconsistencies_fixed} scoring inconsistencies.')

    print('=' * 60)

    # Save incorrect answers if requested
    if args.save_incorrect:
        incorrect_results = []
        for result in valid_results:
            score = result.get('score', 0)
            format_type = result.get('format', question_format)
            is_incorrect = score < 1 if format_type == 'mc' else score < 0.5

            if is_incorrect:
                incorrect_results.append(result)

        if incorrect_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            incorrect_file = f'incorrect_rag_{args.model}_{timestamp}.json'

            incorrect_data = {
                'metadata': {
                    'questions_file': args.questions_file,
                    'test_model': args.model,
                    'test_model_name': model_config['model_name'],
                    'grader_model': args.grader,
                    'grader_model_name': grader_config['model_name'],
                    'timestamp': datetime.now().isoformat(),
                    'total_incorrect': len(incorrect_results),
                    'total_processed': len(valid_results),
                    'incorrect_rate': len(incorrect_results)
                    / len(valid_results)
                    if valid_results
                    else 0,
                    'selection_criteria': 'MC: score < 1, QA: score < 0.5',
                    'used_context_field': args.use_context_field,
                    'retrieval_top_k': args.retrieval_top_k,
                    'retrieval_score_threshold': args.retrieval_score_threshold,
                },
                'incorrect_answers': incorrect_results,
            }

            with open(incorrect_file, 'w', encoding='utf-8') as f:
                json.dump(incorrect_data, f, indent=2)

            print(
                f'Saved {len(incorrect_results)} incorrect answers to: {incorrect_file}'
            )
        else:
            print(
                'No incorrect answers to save (all questions answered correctly).'
            )

    # Save detailed results
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'rag_results_{args.model}_{timestamp}.json'
    else:
        output_file = args.output

    # Calculate statistics for metadata
    overall_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_confidence = (
        sum(
            [
                r.get('evaluation', {}).get('confidence', 0)
                for r in valid_results
            ]
        )
        / len(valid_results)
        if valid_results
        else 0
    )
    mc_accuracy = sum(mc_scores) / len(mc_scores) if mc_scores else None
    qa_accuracy = sum(qa_scores) / len(qa_scores) if qa_scores else None

    output_data = {
        'metadata': {
            'questions_file': args.questions_file,
            'test_model': args.model,
            'test_model_name': model_config['model_name'],
            'grader_model': args.grader,
            'grader_model_name': grader_config['model_name'],
            'parallel_workers': args.parallel,
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'overall_accuracy': overall_accuracy,
            'average_confidence': avg_confidence,
            'mc_accuracy': mc_accuracy,
            'qa_accuracy': qa_accuracy,
            'total_questions': len(data),
            'original_dataset_size': original_count,
            'random_selection': args.random if args.random else None,
            'random_seed': args.seed if args.seed is not None else None,
            'processed_questions': len(valid_results),
            'mc_questions': len(mc_results),
            'qa_questions': len(qa_results),
            'rag_config_file': args.rag_config,
            'used_context_field': args.use_context_field,
            'retrieval_top_k': args.retrieval_top_k,
            'retrieval_score_threshold': args.retrieval_score_threshold,
        },
        'results': results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f'\nDetailed results saved to: {output_file}')


if __name__ == '__main__':
    main()
