"""Expose the distllm RAG chat experience through an OpenAI compatible API."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator
from typing import Iterable

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field

from distllm.chat_argoproxy import ChatAppConfig
from distllm.chat_argoproxy import ConversationPromptTemplate

# Load environment variables upfront so config paths/API keys can come from .env
load_dotenv()

CONFIG_ENV_VAR = 'DISTLLM_CHAT_CONFIG'
DEFAULT_TOP_K = int(os.getenv('DISTLLM_CHAT_RETRIEVAL_TOP_K', '20'))
DEFAULT_SCORE_THRESHOLD = float(
    os.getenv('DISTLLM_CHAT_SCORE_THRESHOLD', '0.1'),
)
DEFAULT_DEBUG_RETRIEVAL = os.getenv(
    'DISTLLM_CHAT_DEBUG_RETRIEVAL', '0'
).lower() in {
    '1',
    'true',
    'yes',
}

app = FastAPI(
    title='distllm-rag-server',
    description='OpenAI-compatible wrapper over distllm.chat_argoproxy',
    version='0.1.0',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

_rag_model = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, str]]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False
    top_k: int | None = Field(default=None, gt=0)
    score_threshold: float | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: dict[str, str]
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: list[ChatCompletionChoice]


def _load_rag_model():
    """Instantiate (or memoize) the RagGenerator."""
    global _rag_model  # noqa: PLW0603
    if _rag_model is not None:
        return _rag_model

    config_path = os.getenv(CONFIG_ENV_VAR)
    if not config_path:
        raise RuntimeError(
            f'Set {CONFIG_ENV_VAR} to the YAML config used by chat_argoproxy.py',
        )

    config = ChatAppConfig.from_yaml(Path(config_path))
    _rag_model = config.rag_configs.get_rag_model()
    return _rag_model


def _normalize_content(content: str | list[dict[str, str]]) -> str:
    """Support OpenAI multi-part message payloads by coalescing text segments."""
    if isinstance(content, str):
        return content

    texts: list[str] = []
    for block in content:
        if block.get('type') == 'text':
            texts.append(block.get('text', ''))
    return ''.join(texts)


def _build_conversation(
    messages: Iterable[ChatMessage],
) -> tuple[list[tuple[str, str]], str]:
    """Convert OpenAI-style messages into the format ChatArgoproxy expects."""
    conversation_history: list[tuple[str, str]] = []
    latest_user_message = ''

    for msg in messages:
        content = _normalize_content(msg.content).strip()
        if not content:
            # Skip empty content blocks entirely to avoid confusing retrieval
            continue

        role = msg.role.lower()
        if role == 'user':
            speaker = 'User'
            latest_user_message = content
        elif role == 'assistant':
            speaker = 'Assistant'
        elif role == 'system':
            speaker = 'System'
        else:
            speaker = role.title()
        conversation_history.append((speaker, content))

    if not latest_user_message:
        raise HTTPException(
            status_code=400,
            detail='At least one user message is required to run retrieval.',
        )

    return conversation_history, latest_user_message


def _build_response_payload(
    *,
    model: str,
    assistant_message: str,
) -> ChatCompletionResponse:
    choice = ChatCompletionChoice(
        index=0,
        message={'role': 'assistant', 'content': assistant_message},
        finish_reason='stop',
    )
    return ChatCompletionResponse(
        id=f'chatcmpl-{uuid.uuid4()}',
        created=int(time.time()),
        model=model,
        choices=[choice],
    )


def _stream_response(
    payload: ChatCompletionResponse,
) -> AsyncGenerator[str, None]:
    """Convert a JSON payload into an SSE stream with a single delta."""
    chunk = {
        'id': payload.id,
        'object': 'chat.completion.chunk',
        'created': payload.created,
        'model': payload.model,
        'choices': [
            {
                'index': 0,
                'delta': payload.choices[0].message,
                'finish_reason': None,
            },
        ],
    }
    final_chunk = {
        'id': payload.id,
        'object': 'chat.completion.chunk',
        'created': payload.created,
        'model': payload.model,
        'choices': [
            {
                'index': 0,
                'delta': {},
                'finish_reason': payload.choices[0].finish_reason,
            },
        ],
    }

    async def generator() -> AsyncGenerator[str, None]:
        yield f'data: {json.dumps(chunk)}\n\n'
        yield f'data: {json.dumps(final_chunk)}\n\n'
        yield 'data: [DONE]\n\n'

    return generator()


@app.on_event('startup')
def _startup() -> None:
    """Fail fast if the config is missing."""
    _load_rag_model()


@app.get('/health')
def health() -> dict[str, str]:
    """Simple readiness probe."""
    return {'status': 'ok'}


@app.post('/v1/chat/completions')
async def chat_completions(
    request: ChatCompletionRequest,
):
    """OpenAI-compatible chat completions endpoint."""
    rag_model = _load_rag_model()
    conversation_history, latest_user_message = _build_conversation(
        request.messages,
    )
    prompt_template = ConversationPromptTemplate(conversation_history)

    temperature = (
        request.temperature
        if request.temperature is not None
        else getattr(rag_model.generator, 'temperature', 0.0)
    )
    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else getattr(rag_model.generator, 'max_tokens', 1024)
    )

    retrieval_top_k = request.top_k or DEFAULT_TOP_K
    retrieval_score_threshold = request.score_threshold
    if retrieval_score_threshold is None:
        retrieval_score_threshold = DEFAULT_SCORE_THRESHOLD

    response_list = await run_in_threadpool(
        rag_model.generate,
        [latest_user_message],
        prompt_template,
        retrieval_top_k,
        retrieval_score_threshold,
        max_tokens,
        temperature,
        DEFAULT_DEBUG_RETRIEVAL,
    )
    assistant_response = response_list[0]

    payload = _build_response_payload(
        model=request.model
        or getattr(rag_model.generator, 'model', 'unknown'),
        assistant_message=assistant_response,
    )

    if request.stream:
        return StreamingResponse(
            _stream_response(payload),
            media_type='text/event-stream',
        )

    return JSONResponse(payload.model_dump())
