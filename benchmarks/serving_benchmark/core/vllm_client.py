"""vLLM client and model registry.

Provides:
- ``ModelConfig``: per-model connection parameters (loaded from models.yaml)
- ``ModelRegistry``: loads and indexes all models + workflow agent mapping
- ``VLLMClient``: thin async wrapper around vLLM's OpenAI-compatible API
  with streaming support for precise TTFT measurement
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import tiktoken
import yaml

logger = logging.getLogger("benchmark")

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
)


@dataclass
class ModelConfig:
    """Connection parameters for a single vLLM model server."""

    alias: str = ""
    model_name: str = ""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 4096
    context_length: int = 32768
    temperature: float = 0.7
    size_tier: str = "small"
    tensor_parallel: int = 1


class ModelRegistry:
    """Loads ``configs/models.yaml`` and provides lookup by alias."""

    def __init__(self, config_path: Optional[str] = None):
        self._path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        with open(self._path) as f:
            raw = yaml.safe_load(f)

        self._models: Dict[str, ModelConfig] = {}
        for alias, cfg in (raw.get("models") or {}).items():
            self._models[alias] = ModelConfig(
                alias=alias,
                model_name=cfg.get("model_name", ""),
                base_url=cfg.get("base_url", "http://localhost:8000/v1"),
                api_key=cfg.get("api_key", "EMPTY"),
                max_tokens=cfg.get("max_tokens", 4096),
                context_length=cfg.get("context_length", 32768),
                temperature=cfg.get("temperature", 0.7),
                size_tier=cfg.get("size_tier", "small"),
                tensor_parallel=cfg.get("tensor_parallel", 1),
            )

        self._workflow_cfg: Dict[str, Any] = raw.get("workflow") or {}

    # -- accessors ----------------------------------------------------------

    def get(self, alias: str) -> ModelConfig:
        if alias not in self._models:
            raise KeyError(
                f"Model '{alias}' not found. Available: {list(self._models)}"
            )
        return self._models[alias]

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    @property
    def workflow_config(self) -> Dict[str, Any]:
        return self._workflow_cfg

    @property
    def first_model(self) -> str:
        return next(iter(self._models))


class VLLMClient:
    """Async client for a single vLLM server (OpenAI-compatible API).

    Supports both streaming (for TTFT measurement) and non-streaming calls.
    """

    def __init__(self, model_config: ModelConfig):
        from openai import AsyncOpenAI

        self.cfg = model_config
        self._client = AsyncOpenAI(
            base_url=model_config.base_url,
            api_key=model_config.api_key,
        )

        try:
            self._tokenizer = tiktoken.encoding_for_model(model_config.model_name)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # -- prompt truncation --------------------------------------------------

    # tiktoken (cl100k_base) vs model-native tokenizer can differ by 10-20%;
    # this factor compensates so the actual token count stays within limits.
    TOKENIZER_SAFETY_FACTOR = 0.85

    @property
    def max_input_tokens(self) -> int:
        """Maximum input tokens (with safety margin for tokenizer mismatch)."""
        raw = self.cfg.context_length - self.cfg.max_tokens
        return int(raw * self.TOKENIZER_SAFETY_FACTOR)

    def truncate_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Truncate the longest user message so total tokens fit within context.

        Strategy: keep system messages intact, truncate the longest user/
        assistant content from the tail until the total fits.
        """
        limit = self.max_input_tokens
        total = self.count_messages_tokens(messages)
        if total <= limit:
            return messages

        overflow = total - limit
        logger.warning(
            "Prompt %d tokens exceeds limit %d (context=%d, gen=%d, safety=%.0f%%), "
            "truncating %d tokens",
            total, limit, self.cfg.context_length, self.cfg.max_tokens,
            self.TOKENIZER_SAFETY_FACTOR * 100, overflow,
        )

        messages = [dict(m) for m in messages]

        longest_idx = -1
        longest_len = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                continue
            n = self.count_tokens(msg.get("content", ""))
            if n > longest_len:
                longest_len = n
                longest_idx = i

        if longest_idx < 0:
            return messages

        content = messages[longest_idx]["content"]
        encoded = self._tokenizer.encode(content, disallowed_special=())
        keep = max(len(encoded) - overflow, 64)
        truncated = self._tokenizer.decode(encoded[:keep])
        messages[longest_idx]["content"] = truncated + "\n...[truncated]"
        return messages

    # -- core API -----------------------------------------------------------

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield content deltas from a streaming chat completion."""
        messages = self.truncate_messages(messages)
        params: Dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": (
                temperature if temperature is not None else self.cfg.temperature
            ),
        }
        response = await self._client.chat.completions.create(**params)
        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Non-streaming chat completion."""
        messages = self.truncate_messages(messages)
        params: Dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "temperature": (
                temperature if temperature is not None else self.cfg.temperature
            ),
        }
        response = await self._client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    async def chat_stream_with_timing(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, float, float]:
        """Stream and return (full_text, t_first_token, t_last_token)."""
        import time

        collected: list[str] = []
        t_first = 0.0
        t_last = 0.0

        async for delta in self.chat_stream(
            messages, max_tokens=max_tokens, temperature=temperature
        ):
            now = time.time()
            if not collected:
                t_first = now
            collected.append(delta)
            t_last = now

        return "".join(collected).strip(), t_first, t_last

    # -- token helpers ------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 4  # role overhead
            total += self.count_tokens(msg.get("content", ""))
        total += 2  # assistant priming
        return total
