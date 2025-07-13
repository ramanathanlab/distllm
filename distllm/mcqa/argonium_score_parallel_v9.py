#!/usr/bin/env python3

"""
Argonium Advanced Question Grader v9.0 (Parallel)

Usage:
    python argonium_score_parallel_v9.py <questions_file.json> --model <model_shortname> --grader <grader_shortname> [--config <config_file>] [--parallel <num_workers>] [--format auto|mc|qa] [--random <num_questions>] [--seed <random_seed>] [--save-incorrect]

Where:
    - questions_file.json: A JSON file with an array of objects, each having "question" and "answer" fields
    - model_shortname: The shortname of the model to test from model_servers.yaml
    - grader_shortname: The shortname of the model to use for grading from model_servers.yaml
    - config_file: Configuration file to use for model settings (default: model_servers.yaml)
    - parallel: Number of concurrent workers for parallel processing (default: 1)
    - format: Format of questions (auto, mc, qa) (default: auto)
    - random: Randomly select N questions from the dataset (optional)
    - seed: Random seed for reproducible question selection (optional, only used with --random)
    - save-incorrect: Save incorrectly answered questions to a separate JSON file (optional)

Examples:
    python argonium_score_parallel_v9.py frg_mc_100.json --model llama --grader gpt41 --parallel 4
    python argonium_score_parallel_v9.py frg_mc_100.json --model llama --grader gpt41 --config custom_models.yaml
    python argonium_score_parallel_v9.py frg_mc_100.json --model llama --grader gpt41 --random 20
    python argonium_score_parallel_v9.py frg_mc_100.json --model llama --grader gpt41 --random 20 --seed 42
    python argonium_score_parallel_v9.py frg_mc_100.json --model llama --grader gpt41 --save-incorrect

The script:
1) Auto-detects question format (multiple-choice or free-form QA) by default
2) Uses the specified MODEL to generate an answer to each question
3) Uses the specified GRADER to evaluate the model's answer against the reference answer
4) Reports detailed accuracy metrics and exports results
5) Processes multiple questions in parallel when --parallel > 1
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

import backoff
import openai
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

# Global client cache for OpenAI clients (thread-safe)
_client_cache = {}
_client_cache_lock = threading.Lock()


def get_openai_client(api_key, api_base, timeout=120.0):
    """
    Get or create a cached OpenAI client for the given configuration.
    Thread-safe client caching to avoid creating new clients for every request.

    Args:
        api_key (str): OpenAI API key
        api_base (str): API base URL
        timeout (float): Request timeout in seconds

    Returns:
        OpenAI client instance
    """
    # Create a cache key based on the configuration
    cache_key = (api_key, api_base, timeout)

    with _client_cache_lock:
        if cache_key not in _client_cache:
            import openai as openai_module

            _client_cache[cache_key] = openai_module.OpenAI(
                api_key=api_key, base_url=api_base, timeout=timeout
            )
        return _client_cache[cache_key]


def load_model_config(model_shortname, config_file="model_servers.yaml"):
    """
    Load model configuration from the specified configuration file.
    Returns a dictionary with api_key, api_base, and model_name.
    """
    if os.path.isabs(config_file):
        yaml_path = config_file
    else:
        yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_file
        )

    try:
        with open(yaml_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

        # Look for the model by shortname
        for server in config["servers"]:
            if server["shortname"] == model_shortname:
                api_key = server["openai_api_key"]
                # Handle environment variable in api key if present
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    api_key = os.environ.get(env_var, "")
                    if not api_key:
                        print(f"Error: Environment variable {env_var} not set")
                        sys.exit(1)

                return {
                    "api_key": api_key,
                    "api_base": server["openai_api_base"],
                    "model_name": server["openai_model"],
                }

        # If not found
        print(f"Error: Model '{model_shortname}' not found in model_servers.yaml")
        print(
            "Available models:", ", ".join([s["shortname"] for s in config["servers"]])
        )
        sys.exit(1)

    except FileNotFoundError:
        print(f"Error: model_servers.yaml not found at {yaml_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        sys.exit(1)


def detect_choice_identifier_type(question_text):
    """
    Detect whether a question uses letter (A, B, C) or number (1, 2, 3) identifiers.

    Args:
        question_text (str): The question text to analyze

    Returns:
        str: 'letter' if using A,B,C format, 'number' if using 1,2,3 format, 'letter' as default
    """
    # Look for standard multiple choice patterns like "A)" or "1."
    has_letter_option = bool(re.search(r"(?:^|\n)\s*([A-E])[.):,]\s", question_text))
    has_number_option = bool(re.search(r"(?:^|\n)\s*([1-5])[.):,]\s", question_text))

    # Default to the input format: if numbers are found, use numbers; if letters are found, use letters
    if has_number_option:
        return "number"
    elif has_letter_option:
        return "letter"
    else:
        # If neither is clearly detected, default to 'letter'
        return "letter"


def detect_question_format(questions):
    """
    Detect whether the questions are in multiple-choice or free-form QA format.
    Returns 'mc' for multiple-choice or 'qa' for free-form QA.

    Detection is based on:
    1. The presence of numbering/lettering in the question (A., B., C. or 1., 2., 3.)
    2. The ratio of questions containing these patterns
    """
    mc_count = 0
    qa_count = 0

    # Patterns for multiple choice questions
    mc_patterns = [
        r"(?:^|\n)\s*([A-E])[.):]\s",  # Letter options: A., A), A:
        r"(?:^|\n)\s*([1-5])[.):]\s",  # Number options: 1., 1), 1:
        r"\n\s*Option\s+[A-E][.:)]",  # Option A., Option B., etc.
        r"\n\s*Choice\s+[A-E][.:)]",  # Choice A., Choice B., etc.
        r"\n\s*Answer\s+[A-E][.:)]",  # Answer A., Answer B., etc.
    ]

    for qa_pair in questions:
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")

        # Check for multiple choice patterns in the question
        is_mc = False
        for pattern in mc_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                is_mc = True
                break

        if is_mc:
            mc_count += 1
        else:
            qa_count += 1

    # If more than 60% are MC, we'll consider it an MC dataset
    if len(questions) > 0:
        mc_ratio = mc_count / len(questions)
        if mc_ratio > 0.6:
            return "mc"
        else:
            return "qa"
    else:
        return "qa"  # Default to free-form QA if no questions


def extract_choice_identifier(answer_text):
    """
    Extract the choice identifier (A, B, C, D, E, etc. or 1, 2, 3, 4, 5, etc.) from an answer text.
    Uses multiple regex patterns to handle different formats.

    Returns a tuple of (identifier_type, identifier) where identifier_type is either 'letter' or 'number'
    and identifier is the letter (A-Z) or number (1-9) that was identified.
    """
    # First, check if this is a multiple-choice format text
    if not answer_text:
        return (None, None)

    # More strict patterns for multiple choice options (look for standard patterns)
    # This looks for options clearly marked, like "A)" or "A." at beginning of lines or text
    letter_mc_pattern = r"(?:^|\n)\s*([A-E])[.):]\s"
    number_mc_pattern = r"(?:^|\n)\s*([1-5])[.):]\s"

    # Also look for explicit mentions of options
    letter_explicit = r"(?:option|answer|choice)\s+([A-E])\b"
    number_explicit = r"(?:option|answer|choice)\s+([1-5])\b"

    # Look for standard statements about correct answers
    letter_statement = r"(?:the\s+(?:correct\s+)?answer\s+is\s+([A-E]))|(?:\b([A-E])\s+is\s+(?:the\s+)?correct)"
    number_statement = r"(?:the\s+(?:correct\s+)?answer\s+is\s+([1-5]))|(?:\b([1-5])\s+is\s+(?:the\s+)?correct)"

    # Try the strict patterns first
    for pattern in [letter_mc_pattern, letter_explicit, letter_statement]:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group:
                    return ("letter", group.upper())

    for pattern in [number_mc_pattern, number_explicit, number_statement]:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            for group in match.groups():
                if group:
                    return ("number", group)

    # If strict patterns don't match, look at the beginning of the response
    # This pattern searches for an answer that starts with a letter/number indicator
    first_line = answer_text.split("\n")[0].strip()

    # Look for responses that begin with "A:", "A.", "A)", or just "A"
    letter_start = re.match(r"^([A-E])(?:[.):,]|\s|$)", first_line, re.IGNORECASE)
    if letter_start:
        return ("letter", letter_start.group(1).upper())

    # Look for responses that begin with "1:", "1.", "1)", or just "1"
    number_start = re.match(r"^([1-5])(?:[.):,]|\s|$)", first_line, re.IGNORECASE)
    if number_start:
        return ("number", number_start.group(1))

    # For short answers, scan for a valid option letter (A-E) or number (1-5)
    if len(answer_text) < 100:  # Only for short answers to avoid false matches
        # Look for any standalone A, B, C, D, E in the text
        standalone_letter = re.search(r"\b([A-E])\b", answer_text, re.IGNORECASE)
        if standalone_letter:
            return ("letter", standalone_letter.group(1).upper())

        # Look for any standalone 1, 2, 3, 4, 5 in the text
        standalone_number = re.search(r"\b([1-5])\b", answer_text)
        if standalone_number:
            return ("number", standalone_number.group(1))

    # If no pattern matches, return None for both
    return (None, None)


def normalize_choice_identifier(identifier, identifier_type, target_type):
    """
    Normalize a choice identifier to the target type.

    Args:
        identifier (str): The identifier (A, B, C... or 1, 2, 3...)
        identifier_type (str): 'letter' or 'number'
        target_type (str): 'letter' or 'number'

    Returns:
        str: The normalized identifier, or None if conversion fails
    """
    if not identifier or not identifier_type or not target_type:
        return None

    if identifier_type == target_type:
        return identifier

    try:
        if identifier_type == "letter" and target_type == "number":
            # Convert A->1, B->2, etc.
            return str(ord(identifier.upper()) - 64)
        elif identifier_type == "number" and target_type == "letter":
            # Convert 1->A, 2->B, etc.
            num = int(identifier)
            if 1 <= num <= 26:
                return chr(64 + num)
    except (ValueError, TypeError):
        pass

    return None


def extract_option_content(question, choice_identifier, identifier_type):
    """
    Extract the content of a specific option from a multiple-choice question.

    Args:
        question (str): The full question text
        choice_identifier (str): The choice identifier (A, B, 1, 2, etc.)
        identifier_type (str): 'letter' or 'number'

    Returns:
        str: The content of the option, or None if not found
    """
    if not question or not choice_identifier:
        return None

    # Escape the identifier for regex
    escaped_id = re.escape(choice_identifier)

    # Pattern to match the option and capture its content
    # Looks for the identifier followed by punctuation, then captures until next option or end
    pattern = r"(?:^|\n)\s*" + escaped_id + r"[.):,]\s+(.+?)(?=\n\s*[A-E1-5][.):,]|$)"

    match = re.search(pattern, question, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # Clean up the content (remove extra whitespace, newlines)
        content = re.sub(r"\s+", " ", content)
        return content

    return None


def check_content_consistency(model_answer, correct_option_content):
    """
    Check if the model's answer content is consistent with the correct option content.

    Args:
        model_answer (str): The model's full answer
        correct_option_content (str): The content of the correct option

    Returns:
        bool: True if the content seems consistent, False otherwise
    """
    if not model_answer or not correct_option_content:
        return False

    # Clean and normalize both texts for comparison
    model_clean = re.sub(r"\s+", " ", model_answer.lower().strip())
    option_clean = re.sub(r"\s+", " ", correct_option_content.lower().strip())

    # If the correct option content is substantial (>15 chars), look for significant overlap
    if len(option_clean) > 15:
        # Look for at least 60% of the option content words in the model answer
        option_words = set(re.findall(r"\w+", option_clean))
        model_words = set(re.findall(r"\w+", model_clean))

        if len(option_words) > 0:
            overlap = len(option_words.intersection(model_words))
            overlap_ratio = overlap / len(option_words)
            return overlap_ratio >= 0.6

    # For shorter content, look for exact substring match
    return option_clean in model_clean


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    giveup=lambda e: "Invalid authentication" in str(e),
    max_time=300,
)
def generate_answer(question, config, question_format="auto"):
    """
    Generate an answer to a question using the specified model.
    Handles both multiple-choice and free-form QA formats.
    Returns the model's raw answer text.

    This function uses exponential backoff to handle rate limits and transient errors.
    It will retry up to 5 times with increasing delays between attempts or until max_time is reached.

    Args:
        question (str): The question to answer
        config (dict): Configuration for the model API
        question_format (str): 'mc' for multiple-choice, 'qa' for free-form QA, 'auto' for auto-detect
    """
    # Configure the OpenAI client (thread-safe approach)
    api_key = config["api_key"]
    api_base = config["api_base"]
    model_name = config["model_name"]

    # Auto-detect question format if not specified
    actual_format = question_format
    if question_format == "auto":
        # Look for multiple-choice indicators
        mc_patterns = [
            r"(?:^|\n)\s*([A-E])[.):]\s",  # Letter options: A., A), A:
            r"(?:^|\n)\s*([1-5])[.):]\s",  # Number options: 1., 1), 1:
        ]

        is_mc = False
        for pattern in mc_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                is_mc = True
                break

        actual_format = "mc" if is_mc else "qa"

    # Based on format, determine appropriate prompts
    if actual_format == "mc":
        # Detect choice identifier type (letter/number) using unified logic
        id_type_in_question = detect_choice_identifier_type(question)

        # Determine label format for prompts
        if id_type_in_question == "number":
            label_format = "number (1, 2, 3, etc.)"
        else:
            label_format = "letter (A, B, C, etc.)"

        # System prompt designed for multiple-choice questions
        system_message = (
            "You are an expert at multiple-choice questions. "
            "Think through the question step by step, but provide only a concise final answer. "
            "Your response should contain ONLY:\n"
            f"1. The correct {label_format}\n"
            "2. A brief explanation (2-3 sentences) of why this choice is correct.\n\n"
            "Do not include the question restatement or list of alternatives in your response."
        )

        user_message = (
            f"Please answer this multiple-choice question. Think through it carefully:\n"
            f"- First, restate the question to yourself\n"
            f"- Then, consider all the alternative answers provided\n"
            f"- Finally, provide your response with ONLY the correct {label_format} "
            f"followed by 2-3 sentences explaining why this choice is correct.\n\n"
            f"Question:\n{question}"
        )
    else:  # Free-form QA format
        # System prompt designed for general questions
        system_message = (
            "You are an expert at answering questions in various fields including science, mathematics, "
            "physics, computer science, and more. Provide concise, accurate, and thorough answers "
            "based on your knowledge. Focus on being factually correct."
        )

        user_message = (
            f"Please answer the following question thoroughly and accurately:\n\n"
            f"{question}"
        )

    try:
        # Add a small random delay to avoid sending too many requests simultaneously
        jitter = random.uniform(0.1, 1.0)
        time.sleep(jitter)

        # Get cached client instance (thread-safe)
        client = get_openai_client(api_key, api_base, timeout=120.0)

        # Check if we need to skip temperature (for reasoning models like o3 and o4mini)
        # OpenAI doesn't support temperature parameter for reasoning models
        skip_temperature = any(
            name in model_name.lower() for name in ["o3", "o4-mini", "o4mini"]
        )

        # Prepare parameters
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        }

        # Add temperature only for models that support it
        if not skip_temperature:
            params["temperature"] = 0.0  # Use low temperature for consistent results

        # Call the API with the prepared parameters
        response = client.chat.completions.create(**params)

        # Handle the response based on the OpenAI client version
        if hasattr(response, "choices"):
            # New OpenAI client
            generated_text = response.choices[0].message.content.strip()
        else:
            # Legacy dict-style response
            generated_text = response["choices"][0]["message"]["content"].strip()
        return generated_text
    except Exception as e:
        # Propagate the exception to trigger backoff
        print(f"Error in generate_answer (will retry): {str(e)}")
        raise


def _evaluate_answer_with_retry(
    question, reference_answer, model_answer, config, question_format="auto"
):
    """
    Wrapper for evaluate_answer with custom retry logic for JSON parsing issues.
    """
    # First attempt
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
        if "JSON parsing failed, retrying once" in str(e):
            # Retry once for JSON parsing issues
            try:
                print("Retrying evaluation due to JSON parsing issue...")
                return _evaluate_answer_core(
                    question,
                    reference_answer,
                    model_answer,
                    config,
                    question_format,
                    retry_count=1,
                )
            except Exception as e2:
                # If it fails again, return a default result
                print(f"Evaluation failed after retry: {str(e2)}")
                return {
                    "score": 0,
                    "confidence": 0.1,
                    "match": False,
                    "format": question_format if question_format != "auto" else "qa",
                    "reasoning": "Failed to evaluate after retry",
                    "parse_error": True,
                }
        else:
            # For other errors, re-raise to trigger normal backoff
            raise


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    giveup=lambda e: "Invalid authentication" in str(e),
    max_time=300,
)
def _evaluate_answer_core(
    question,
    reference_answer,
    model_answer,
    config,
    question_format="auto",
    retry_count=0,
):
    """
    Evaluate a model's answer against the reference answer using the specified grader model.
    Handles both multiple-choice and free-form QA formats.
    Returns a dictionary with score, correct_letter, model_letter, and confidence.

    This function uses exponential backoff to handle rate limits and transient errors.
    It will retry up to 5 times with increasing delays between attempts or until max_time is reached.

    Args:
        question (str): The question being answered
        reference_answer (str): The reference answer
        model_answer (str): The model's answer
        config (dict): Configuration for the model API
        question_format (str): 'mc' for multiple-choice, 'qa' for free-form QA, 'auto' for auto-detect
    """
    # Configure the OpenAI client for the grader (thread-safe approach)
    api_key = config["api_key"]
    api_base = config["api_base"]
    model_name = config["model_name"]

    # Auto-detect question format if not specified
    actual_format = question_format
    if question_format == "auto":
        # Look for multiple-choice indicators
        mc_patterns = [
            r"(?:^|\n)\s*([A-E])[.):]\s",  # Letter options: A., A), A:
            r"(?:^|\n)\s*([1-5])[.):]\s",  # Number options: 1., 1), 1:
        ]

        is_mc = False
        for pattern in mc_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                is_mc = True
                break

        actual_format = "mc" if is_mc else "qa"

    # Based on format, determine appropriate evaluation approach
    if actual_format == "mc":
        # Detect choice identifier type (letter/number) using unified logic
        id_type_in_question = detect_choice_identifier_type(question)

        # Extract choice identifiers to provide context to the grader, but don't use for direct scoring
        ref_id_type, ref_id = extract_choice_identifier(reference_answer)
        model_id_type, model_id = extract_choice_identifier(model_answer)

        # Extract option content to provide additional context to grader
        correct_option_content = None
        if ref_id and question:
            correct_option_content = extract_option_content(
                question, ref_id, ref_id_type
            )

        # All evaluation will be done by the grader model - no direct regex-based scoring

        # For high accuracy grading, we'll use a detailed prompt with multiple checks
        # Determine whether to ask for letters or numbers in evaluation based on the question format
        option_type = "number" if id_type_in_question == "number" else "letter"
        option_examples = (
            "(1, 2, 3, etc.)" if option_type == "number" else "(A, B, C, etc.)"
        )

        system_message = (
            "You are an expert evaluator for multiple-choice questions. "
            "Your task is to determine if a model's answer matches the correct answer, "
            f"focusing specifically on which option {option_examples} was selected."
        )

        # Build additional context for the grader
        analysis_context = ""
        if ref_id:
            analysis_context += f"\nPRE-ANALYSIS CONTEXT (for reference only):\n"
            analysis_context += (
                f"- Detected correct choice identifier: {ref_id} ({ref_id_type})\n"
            )
            if correct_option_content:
                analysis_context += (
                    f'- Correct option content: "{correct_option_content}"\n'
                )
        if model_id:
            analysis_context += (
                f"- Detected model choice identifier: {model_id} ({model_id_type})\n"
            )
        if analysis_context:
            analysis_context += f"- Question uses {id_type_in_question} format\n"
            analysis_context += f"\nNOTE: This pre-analysis is for context only. Please make your own independent evaluation.\n"

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
        # For free-form QA, we don't need to extract choice identifiers

        # We need a different prompt for evaluating free-form answers
        system_message = (
            "You are an expert evaluator for question answering. "
            "Your task is to determine if a model's answer to a question "
            "is factually correct and sufficiently addresses the question."
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
        # Add a small random delay to avoid sending too many requests simultaneously
        jitter = random.uniform(0.1, 1.0)
        time.sleep(jitter)

        # Get cached client instance (thread-safe)
        client = get_openai_client(api_key, api_base, timeout=120.0)

        # Check if we need to skip temperature (for reasoning models like o3 and o4mini)
        # OpenAI doesn't support temperature parameter for reasoning models
        skip_temperature = any(
            name in model_name.lower() for name in ["o3", "o4-mini", "o4mini"]
        )

        # Prepare parameters
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        }

        # Add temperature only for models that support it
        if not skip_temperature:
            params["temperature"] = 0.0  # Use low temperature for precise evaluation

        # Call the API with the prepared parameters
        response = client.chat.completions.create(**params)

        # Handle the response based on the OpenAI client version
        if hasattr(response, "choices"):
            # New OpenAI client
            evaluation_text = response.choices[0].message.content.strip()
        else:
            # Legacy dict-style response
            evaluation_text = response["choices"][0]["message"]["content"].strip()

        # Try to parse the JSON response
        try:
            evaluation = json.loads(evaluation_text)

            if actual_format == "mc":  # For multiple-choice questions
                # Handle both old and new JSON field names
                if "correct_choice" in evaluation:
                    correct_choice = evaluation["correct_choice"]
                elif "correct_letter" in evaluation:
                    correct_choice = evaluation["correct_letter"]
                else:
                    correct_choice = None

                if "model_choice" in evaluation:
                    model_choice = evaluation["model_choice"]
                elif "model_letter" in evaluation:
                    model_choice = evaluation["model_letter"]
                else:
                    model_choice = None

                # Ensure required fields exist
                if "score" not in evaluation:
                    if "match" in evaluation and isinstance(evaluation["match"], bool):
                        evaluation["score"] = 1 if evaluation["match"] else 0
                    else:
                        evaluation["score"] = 0

                # Standardize field names for consistent output
                if (
                    "correct_choice" in evaluation
                    and "correct_letter" not in evaluation
                ):
                    evaluation["correct_letter"] = evaluation["correct_choice"]
                if "model_choice" in evaluation and "model_letter" not in evaluation:
                    evaluation["model_letter"] = evaluation["model_choice"]

                # Store any extracted identifiers for reference, but don't override grader decision
                if ref_id:
                    evaluation["extracted_correct_choice"] = ref_id
                    evaluation["correct_choice_type"] = ref_id_type
                if model_id:
                    evaluation["extracted_model_choice"] = model_id
                    evaluation["model_choice_type"] = model_id_type

                # Set the question format in the evaluation
                evaluation["format"] = actual_format

            else:  # For free-form QA questions
                # Ensure required fields exist for QA evaluation
                if "score" not in evaluation and "match" in evaluation:
                    # Binary scoring if missing score but match is provided
                    evaluation["score"] = 1.0 if evaluation["match"] else 0.0

                # Set the question format in the evaluation
                evaluation["format"] = "qa"

            return evaluation

        except json.JSONDecodeError:
            # If the model didn't return valid JSON, try to extract basic score information
            print(
                f"Warning: Grader returned invalid JSON, attempting to parse response: {evaluation_text}"
            )

            # Fallback for any format - try to extract score/match information
            score_match = re.search(r'score["\s:]+([01](?:\.\d+)?)', evaluation_text)
            match_text = re.search(r'match["\s:]+(\w+)', evaluation_text, re.IGNORECASE)

            if score_match:
                score = float(score_match.group(1))
                match_value = True if score > 0.5 else False

                # If we have a match value in text, use it instead
                if match_text:
                    match_str = match_text.group(1).lower()
                    if match_str in ["true", "yes", "1"]:
                        match_value = True
                    elif match_str in ["false", "no", "0"]:
                        match_value = False

                return {
                    "score": score,
                    "confidence": 0.5,  # Lower confidence in fallback
                    "match": match_value,
                    "format": actual_format,
                }
            else:
                # If we've already retried for JSON parsing, don't retry again
                if retry_count > 0:
                    print(
                        f"Warning: Could not parse evaluation JSON after retry: {evaluation_text}"
                    )
                    # Return a default low-confidence result instead of hanging
                    return {
                        "score": 0,
                        "confidence": 0.1,  # Very low confidence
                        "match": False,
                        "format": actual_format,
                        "reasoning": "Failed to parse grader response",
                        "parse_error": True,
                    }
                else:
                    print(
                        f"Warning: Could not parse evaluation JSON, will retry once: {evaluation_text}"
                    )
                    # Only retry once for JSON parsing issues
                    raise Exception("JSON parsing failed, retrying once")

    except Exception as e:
        print(f"Error in _evaluate_answer_core (will retry): {str(e)}")
        # Re-raise for retry logic via backoff decorator
        raise


def evaluate_answer(
    question, reference_answer, model_answer, config, question_format="auto"
):
    """
    Public interface for evaluating answers. Uses retry logic for JSON parsing issues.
    """
    return _evaluate_answer_with_retry(
        question, reference_answer, model_answer, config, question_format
    )


def process_question(
    item, model_config, grader_config, question_format="auto", verbose=False
):
    """
    Process a single question - generate answer and evaluate it.
    This function is designed to be called by the thread pool.

    Args:
        item (tuple): A tuple containing (index, question_answer_pair)
        model_config (dict): Configuration for the model API
        grader_config (dict): Configuration for the grader API
        question_format (str): 'mc' for multiple-choice, 'qa' for free-form QA, 'auto' for auto-detect
        verbose (bool): Whether to print detailed logs
    """
    i, qa_pair = item
    question = qa_pair.get("question", "")
    reference_answer = qa_pair.get("answer", "")

    if not question or not reference_answer:
        return {
            "question_id": i,
            "error": "Missing question or answer",
            "skipped": True,
        }

    try:
        # Only print detailed progress if verbose is enabled
        if verbose:
            print(f"\nProcessing question {i}...")

        # Generate model answer with retry logic
        start_time = time.time()
        model_answer = generate_answer(question, model_config, question_format)
        model_time = time.time() - start_time

        # Note: generate_answer uses @backoff decorator and raises exceptions rather than returning error strings

        # Show detailed model response only in verbose mode
        if verbose:
            print(f"\n--- MODEL RESPONSE FOR QUESTION {i} ---")
            print(model_answer)
            print("--- END MODEL RESPONSE ---")
            print(f"Generated answer for question {i} in {model_time:.2f}s")

        # Evaluate the answer with retry logic
        start_time = time.time()
        evaluation = evaluate_answer(
            question, reference_answer, model_answer, grader_config, question_format
        )
        eval_time = time.time() - start_time

        if verbose:
            print(f"Evaluated answer for question {i} in {eval_time:.2f}s")

        # Get the score and format
        score = evaluation.get("score", 0)
        format_type = evaluation.get("format", question_format)
        if format_type == "auto":  # If format was not set, default to auto-detection
            format_type = "mc" if "correct_letter" in evaluation else "qa"

        # Prepare detailed result
        result = {
            "question_id": i,
            "question": question,
            "reference_answer": reference_answer,
            "model_answer": model_answer,
            "evaluation": evaluation,
            "score": score,
            "format": format_type,
            "model_time_seconds": model_time,
            "evaluation_time_seconds": eval_time,
            "skipped": False,
        }

        # Print details if verbose
        if verbose:
            if format_type == "mc":
                correct_letter = evaluation.get("correct_letter", "?")
                model_letter = evaluation.get("model_letter", "?")
                confidence = evaluation.get("confidence", 0)
                # Get the identifier type (number or letter)
                id_type = evaluation.get("identifier_type", "letter")
                id_name = "number" if id_type == "number" else "letter"

                # Double-check identifiers and mark result
                identifiers_match = (
                    model_letter == correct_letter
                    if model_letter and correct_letter
                    else None
                )

                # If identifiers match but score is 0, or don't match but score is 1, there's an inconsistency
                if identifiers_match is not None and (
                    (identifiers_match and score == 0)
                    or (not identifiers_match and score == 1)
                ):
                    print(
                        f"WARNING: Score/{id_name} mismatch detected for Q{i}. Fixing..."
                    )
                    score = 1 if identifiers_match else 0
                    evaluation["score"] = score

                print(
                    f"Q{i} (MC) Result: {'✓' if score == 1 else '✗'} Score: {score}/1 "
                    + f"Confidence: {confidence:.2f} (Model chose {id_name}: {model_letter}, Correct {id_name}: {correct_letter})"
                )
            else:  # QA format
                confidence = evaluation.get("confidence", 0)
                score_str = (
                    f"{score:.2f}"
                    if isinstance(score, float) and score < 1
                    else f"{int(score)}/1"
                )
                print(
                    f"Q{i} (QA) Result: {'✓' if score >= 0.5 else '✗'} Score: {score_str} Confidence: {confidence:.2f}"
                )

        return result

    except Exception as e:
        print(f"\nUnhandled error processing question {i}: {str(e)}")
        # Return partial information but mark as failed
        return {
            "question_id": i,
            "question": question,
            "reference_answer": reference_answer,
            "error": str(e),
            "skipped": True,
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Question Grader (Parallel)")
    parser.add_argument("questions_file", help="JSON file with questions and answers")
    parser.add_argument(
        "--model", required=True, help="Model shortname from model_servers.yaml to test"
    )
    parser.add_argument(
        "--grader",
        required=True,
        help="Model shortname from model_servers.yaml to use for grading",
    )
    parser.add_argument(
        "--config",
        default="model_servers.yaml",
        help="Configuration file to use for model settings (default: model_servers.yaml)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent workers (default: 1)",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "mc", "qa"],
        default="auto",
        help="Format of questions: auto=auto-detect, mc=multiple-choice, qa=free-form QA (default: auto)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for detailed results (default: results_<timestamp>.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--random",
        type=int,
        help="Randomly select N questions from the dataset (must be positive, default: use all questions)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible question selection (only used with --random)",
    )
    parser.add_argument(
        "--save-incorrect",
        action="store_true",
        help="Save incorrectly answered questions to a separate JSON file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------


def main():
    args = parse_arguments()

    # Load model configs
    model_config = load_model_config(args.model, args.config)
    grader_config = load_model_config(args.grader, args.config)

    print(f"Testing model: {args.model} ({model_config['model_name']})")
    print(f"Grading with: {args.grader} ({grader_config['model_name']})")
    print(f"Parallel workers: {args.parallel}")

    # Load question-answer pairs
    try:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading questions file: {e}")
        sys.exit(1)

    # Randomly select questions if --random is specified
    original_count = len(data)
    if args.random:
        if args.random <= 0:
            print(f"Error: --random must be a positive number, got {args.random}")
            sys.exit(1)
        elif args.random < len(data):
            print(
                f"Randomly selecting {args.random} questions from {original_count} total questions..."
            )
            # Set random seed if specified for reproducible selection
            if args.seed is not None:
                random.seed(args.seed)
                print(f"Using random seed: {args.seed}")
            data = random.sample(data, args.random)
            print(f"Selected {len(data)} questions.")
        else:
            print(
                f"Requested {args.random} questions, but dataset only has {original_count}. Using all questions."
            )

    # Determine question format if set to auto
    question_format = args.format
    if question_format == "auto":
        detected_format = detect_question_format(data)
        print(f"Auto-detected question format: {detected_format}")
        question_format = detected_format
    else:
        print(f"Using specified question format: {question_format}")

    # Prepare items for parallel processing
    items = [(i, qa_pair) for i, qa_pair in enumerate(data, 1)]
    results = []

    # Process questions (in parallel if parallel > 1)
    start_time = time.time()

    # Set up synchronized printing for multiple threads
    print(
        f"\nProcessing {len(items)} questions"
        + (
            f" with {args.parallel} parallel workers..."
            if args.parallel > 1
            else " sequentially..."
        )
    )
    print("This may take some time. Each model call has built-in retries and waiting.")

    # Track completed questions for progress reporting - use thread-safe variables
    completed = 0
    total = len(items)
    results_lock = threading.Lock()  # Protect shared state

    if args.parallel > 1:
        # Only show progress bar in parallel mode and when not verbose
        progress_bar = None
        if not args.verbose:
            progress_bar = tqdm(total=total, desc="Processing questions")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.parallel
        ) as executor:
            # Submit all jobs to the executor
            futures = [
                executor.submit(
                    process_question,
                    item,
                    model_config,
                    grader_config,
                    question_format,
                    args.verbose,
                )
                for item in items
            ]

            # Wait for each future to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()

                    # Thread-safe updates to shared state
                    with results_lock:
                        results.append(result)
                        completed += 1
                        current_completed = completed

                        # Calculate running accuracy for display
                        current_results = results.copy()  # Safe copy for calculation

                    # Update progress bar if it exists (safe outside lock)
                    if progress_bar:
                        progress_bar.update(1)

                    # Calculate and display running percentage (only for non-verbose, non-progress-bar mode)
                    if not args.verbose and not progress_bar:
                        # Calculate running accuracy from completed results
                        valid_results = [
                            r for r in current_results if not r.get("skipped", False)
                        ]
                        if valid_results:
                            total_score = sum(r.get("score", 0) for r in valid_results)
                            accuracy = total_score / len(valid_results)
                            print(
                                f"Completed {current_completed}/{total} questions | Running accuracy: {accuracy:.1%}",
                                end="\r",
                            )

                except Exception as e:
                    # Find which item this future was processing
                    for i, submitted_future in enumerate(futures):
                        if submitted_future == future:
                            item_index = items[i][0]
                            break
                    else:
                        item_index = "unknown"

                    print(f"\nUnhandled error in worker for question {item_index}: {e}")

                    # Thread-safe error handling
                    with results_lock:
                        results.append(
                            {
                                "question_id": item_index,
                                "error": str(e),
                                "skipped": True,
                            }
                        )
                        completed += 1

                    # Update progress bar even for errors (safe outside lock)
                    if progress_bar:
                        progress_bar.update(1)

            # Close the progress bar when done
            if progress_bar:
                progress_bar.close()
    else:
        # Single-threaded processing without progress bar
        # Track running totals for efficiency
        total_score = 0
        valid_count = 0

        for item in items:
            result = process_question(
                item, model_config, grader_config, question_format, args.verbose
            )
            results.append(result)
            completed += 1

            # Update running totals efficiently
            if not result.get("skipped", False):
                total_score += result.get("score", 0)
                valid_count += 1

            # Simple progress counter with running accuracy
            if not args.verbose:
                if valid_count > 0:
                    accuracy = total_score / valid_count
                    print(
                        f"Completed {completed}/{total} questions | Running accuracy: {accuracy:.1%}",
                        end="\r",
                    )
                else:
                    print(f"Completed {completed}/{total} questions", end="\r")

    # Print newline after progress counter
    if not args.verbose:
        print()

    # Sort results by question_id
    results.sort(key=lambda x: x["question_id"])

    # Display results
    total_time = time.time() - start_time
    valid_results = [r for r in results if not r.get("skipped", False)]

    # Separate MC and QA results for statistics
    mc_results = [r for r in valid_results if r.get("format", question_format) == "mc"]
    qa_results = [r for r in valid_results if r.get("format", question_format) == "qa"]

    # Calculate scores for each format
    mc_scores = [r.get("score", 0) for r in mc_results]
    qa_scores = [r.get("score", 0) for r in qa_results]
    all_scores = [r.get("score", 0) for r in valid_results]

    # First pass to fix any inconsistencies in MC results
    inconsistencies_fixed = 0
    for result in mc_results:
        evaluation = result.get("evaluation", {})
        correct_letter = evaluation.get("correct_letter", "?")
        model_letter = evaluation.get("model_letter", "?")

        # Double-check for letter/score consistency and fix if needed
        letters_match = (
            model_letter == correct_letter if model_letter and correct_letter else None
        )
        score = result.get("score", 0)

        # If letters match but score is 0, or don't match but score is 1, fix the inconsistency
        if letters_match is not None and (
            (letters_match and score == 0) or (not letters_match and score == 1)
        ):
            inconsistencies_fixed += 1
            score = 1 if letters_match else 0
            result["score"] = score
            if "evaluation" in result:
                result["evaluation"]["score"] = score

    # Recalculate scores if we fixed any inconsistencies
    if inconsistencies_fixed > 0:
        mc_scores = [r.get("score", 0) for r in mc_results]
        all_scores = [r.get("score", 0) for r in valid_results]

    # Print detailed results only in verbose mode
    if args.verbose:
        for result in valid_results:
            i = result["question_id"]
            format_type = result.get("format", question_format)
            score = result.get("score", 0)

            if format_type == "mc":
                evaluation = result.get("evaluation", {})
                correct_letter = evaluation.get("correct_letter", "?")
                model_letter = evaluation.get("model_letter", "?")
                confidence = evaluation.get("confidence", 0)

                # Get the identifier type (number or letter)
                id_type = evaluation.get("identifier_type", "letter")
                id_name = "number" if id_type == "number" else "letter"

                # Print a shortened version of the question
                # Print just the first line of the question
                first_line = result["question"].split("\n")[0]
                print(f"\nQ{i} (MC): {first_line[:80]}...")

                print(
                    f"Result: {'✓' if score == 1 else '✗'} Score: {score}/1 Confidence: {confidence:.2f} "
                    + f"(Model chose {id_name}: {model_letter}, Correct {id_name}: {correct_letter})"
                )
            else:  # QA format
                evaluation = result.get("evaluation", {})
                confidence = evaluation.get("confidence", 0)

                # Print just the first line of the question
                first_line = result["question"].split("\n")[0]
                print(f"\nQ{i} (QA): {first_line[:80]}...")

                score_str = (
                    f"{score:.2f}"
                    if isinstance(score, float) and score < 1
                    else f"{int(score)}/1"
                )
                print(
                    f"Result: {'✓' if score >= 0.5 else '✗'} Score: {score_str} Confidence: {confidence:.2f}"
                )

        # Report any inconsistencies fixed in verbose mode
        if inconsistencies_fixed > 0:
            print(
                f"\nFixed {inconsistencies_fixed} score/letter inconsistencies in the MC results."
            )
    else:
        # In non-verbose mode, just show a brief summary before final results
        if inconsistencies_fixed > 0:
            print(f"\nFixed {inconsistencies_fixed} scoring inconsistencies.")

    # Calculate statistics
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS")
    print("=" * 60)

    # Overall statistics
    if all_scores:
        overall_accuracy = sum(all_scores) / len(all_scores)
        print(
            f"Overall accuracy: {overall_accuracy:.2%} ({sum(all_scores):.1f}/{len(all_scores)})"
        )
    else:
        print("No valid questions processed.")

    # MC statistics
    if mc_scores:
        mc_accuracy = sum(mc_scores) / len(mc_scores)
        print(
            f"Multiple-choice questions: {mc_accuracy:.2%} accuracy ({sum(mc_scores)}/{len(mc_scores)})"
        )

        # Calculate average confidence for MC
        mc_confidences = [
            r.get("evaluation", {}).get("confidence", 0) for r in mc_results
        ]
        avg_mc_confidence = (
            sum(mc_confidences) / len(mc_confidences) if mc_confidences else 0
        )
        print(f"Average MC confidence: {avg_mc_confidence:.2f}")

    # QA statistics
    if qa_scores:
        qa_accuracy = sum(qa_scores) / len(qa_scores)
        print(
            f"Free-form QA questions: {qa_accuracy:.2%} score ({sum(qa_scores):.1f}/{len(qa_scores)})"
        )

        # Calculate average confidence for QA
        qa_confidences = [
            r.get("evaluation", {}).get("confidence", 0) for r in qa_results
        ]
        avg_qa_confidence = (
            sum(qa_confidences) / len(qa_confidences) if qa_confidences else 0
        )
        print(f"Average QA confidence: {avg_qa_confidence:.2f}")

    print(f"Total processing time: {total_time:.2f} seconds")
    print("=" * 60)

    # Save incorrect answers if requested
    if args.save_incorrect:
        # Collect incorrect answers (score < 1 for MC, score < 0.5 for QA)
        incorrect_results = []
        for result in valid_results:
            score = result.get("score", 0)
            format_type = result.get("format", question_format)

            # Determine if this is incorrect based on format
            is_incorrect = False
            if format_type == "mc":
                is_incorrect = score < 1
            else:  # QA format
                is_incorrect = score < 0.5

            if is_incorrect:
                incorrect_results.append(result)

        if incorrect_results:
            # Generate filename for incorrect answers
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            incorrect_file = f"incorrect_{args.model}_{timestamp}.json"

            # Prepare data for incorrect answers file
            incorrect_data = {
                "metadata": {
                    "questions_file": args.questions_file,
                    "test_model": args.model,
                    "test_model_name": model_config["model_name"],
                    "grader_model": args.grader,
                    "grader_model_name": grader_config["model_name"],
                    "timestamp": datetime.now().isoformat(),
                    "total_incorrect": len(incorrect_results),
                    "total_processed": len(valid_results),
                    "incorrect_rate": (
                        len(incorrect_results) / len(valid_results)
                        if valid_results
                        else 0
                    ),
                    "selection_criteria": "MC: score < 1, QA: score < 0.5",
                },
                "incorrect_answers": incorrect_results,
            }

            with open(incorrect_file, "w", encoding="utf-8") as f:
                json.dump(incorrect_data, f, indent=2)

            print(
                f"Saved {len(incorrect_results)} incorrect answers to: {incorrect_file}"
            )
        else:
            print("No incorrect answers to save (all questions answered correctly).")

    # Save detailed results to output file
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{args.model}_{timestamp}.json"
    else:
        output_file = args.output

    # Calculate overall statistics for metadata
    overall_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_confidence = (
        sum([r.get("evaluation", {}).get("confidence", 0) for r in valid_results])
        / len(valid_results)
        if valid_results
        else 0
    )

    # Calculate format-specific statistics for metadata
    mc_accuracy = sum(mc_scores) / len(mc_scores) if mc_scores else None
    qa_accuracy = sum(qa_scores) / len(qa_scores) if qa_scores else None

    output_data = {
        "metadata": {
            "questions_file": args.questions_file,
            "test_model": args.model,
            "test_model_name": model_config["model_name"],
            "grader_model": args.grader,
            "grader_model_name": grader_config["model_name"],
            "parallel_workers": args.parallel,
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "overall_accuracy": overall_accuracy,
            "average_confidence": avg_confidence,
            "mc_accuracy": mc_accuracy,
            "qa_accuracy": qa_accuracy,
            "total_questions": len(data),
            "original_dataset_size": original_count,
            "random_selection": args.random if args.random else None,
            "random_seed": args.seed if args.seed is not None else None,
            "processed_questions": len(valid_results),
            "mc_questions": len(mc_results),
            "qa_questions": len(qa_results),
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
