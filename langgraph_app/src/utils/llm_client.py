"""Provider-agnostic LLM client (Nous Portal or OpenRouter) with heuristic
fallback when no API key is configured.

The fallback path keeps the whole debate graph runnable in dev mode:
Bull/Bear produce deterministic thesis outputs from the AnalystPack numbers.
"""
from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER

T = TypeVar("T", bound=BaseModel)

# Model specs (verified 2026-08-12):
#   deepseek/deepseek-v4-flash-0731 — Sparse MoE · 284B total · 13B active
#   1M context · $0.072/M input on OpenRouter · also served by Nous Portal


def _client():
    from openai import AsyncOpenAI
    import instructor

    kwargs = {
        "base_url": LLM_BASE_URL,
        "api_key": LLM_API_KEY,
    }
    if LLM_PROVIDER == "openrouter":  # attribution headers (optional)
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/agonistes-trading",
            "X-Title": "Project Agonistes",
        }
    raw = AsyncOpenAI(**kwargs)
    return instructor.from_openai(raw, mode=instructor.Mode.JSON)


async def call_openrouter_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> T:
    client = _client()
    return await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


async def call_openrouter_raw(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    from openai import AsyncOpenAI

    resp = await AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    ).chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    # Reasoning models may return empty `content` with the answer in the
    # `reasoning` field — fall back to it rather than returning "".
    content = msg.content or getattr(msg, "reasoning", None) or ""
    return content.strip()


def llm_available() -> bool:
    return bool(LLM_API_KEY)
