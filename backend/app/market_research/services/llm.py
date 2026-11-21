import json
import re
from functools import lru_cache
from typing import TypeVar

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from market_research.config import get_settings

T = TypeVar("T", bound=BaseModel)

_json_mode_supported = True


class LLMError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url, timeout=120)


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def _complete(messages: list[ChatCompletionMessageParam]) -> str:
    global _json_mode_supported
    s = get_settings()

    try:
        if _json_mode_supported:
            try:
                resp = _client().chat.completions.create(
                    model=s.llm_model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            except openai.BadRequestError:
                # Provider/model rejects JSON mode: remember and retry without it
                _json_mode_supported = False
                resp = _client().chat.completions.create(
                    model=s.llm_model, messages=messages, temperature=0.2
                )
        else:
            resp = _client().chat.completions.create(
                model=s.llm_model, messages=messages, temperature=0.2
            )
    except openai.APIStatusError as e:
        raise LLMError(
            f"LLM request failed ({e.status_code}) on provider "
            f"'{s.llm_provider}': {e.message}. If OpenRouter credits are "
            f"exhausted, set LLM_PROVIDER=ollama in .env."
        ) from e
    except openai.APIConnectionError as e:
        raise LLMError(
            f"Could not reach LLM provider '{s.llm_provider}' at "
            f"{s.llm_base_url}. Is it running / reachable?"
        ) from e

    content = resp.choices[0].message.content
    if not content:
        raise LLMError("LLM returned an empty response.")
    return content


def generate_structured(
    prompt: str,
    schema: type[T],
    system: str | None = None,
    max_retries: int = 1,
) -> T:
    """Ask the LLM for JSON matching `schema`, validate it, retry once on failure."""
    schema_json = json.dumps(schema.model_json_schema())
    system_msg = (system + "\n\n" if system else "") + (
        "Respond with a single JSON object that matches this JSON schema. "
        "No prose and no markdown fences.\n" + schema_json
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        raw = _complete(messages)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except (ValidationError, ValueError) as e:
            last_error = e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"That response was invalid: {e}\n"
                    "Return a corrected JSON object only.",
                }
            )

    raise LLMError(
        f"LLM output failed validation for {schema.__name__} "
        f"after {max_retries + 1} attempts: {last_error}"
    )