import json
import re
import time
from functools import lru_cache
from typing import TypeVar

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from market_research.config import get_settings

T = TypeVar("T", bound=BaseModel)
_json_mode_disabled = False
_consecutive_json_400 = 0


class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    pass


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=120,
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def _complete_with_openai_compatible(
    messages: list[ChatCompletionMessageParam],
) -> str:
    global _json_mode_disabled, _consecutive_json_400

    settings = get_settings()
    json_mode_setting = settings.llm_json_mode
    if json_mode_setting == "off":
        initial_json_mode = False
    elif json_mode_setting == "on":
        initial_json_mode = True
    else:
        initial_json_mode = not settings.llm_model.startswith("openai/gpt-oss")

    last_error: Exception = LLMError("LLM returned an empty response.")
    json_mode = initial_json_mode and not _json_mode_disabled

    def make_request(use_json_mode: bool):
        request = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": 4096,
        }
        if use_json_mode:
            request["response_format"] = {"type": "json_object"}
        if "gpt-oss" in settings.llm_model:
            request["extra_body"] = {
                "reasoning_format": "parsed",
                "reasoning_effort": "low",
            }
        return _openai_client().chat.completions.create(**request)

    for attempt in range(3):
        use_json_mode = json_mode
        try:
            response = make_request(use_json_mode)
        except openai.APIStatusError as error:
            status = getattr(error, "status_code", None)
            detail = str(error)
            retry_after = getattr(error, "retry_after", None)
            if retry_after is None:
                headers = getattr(error, "response", None)
                headers = getattr(headers, "headers", None)
                if headers:
                    retry_after = headers.get("retry-after")
            try:
                retry_wait = float(retry_after) if retry_after is not None else 0
            except (TypeError, ValueError):
                retry_wait = 0
            if (
                status == 429
                and ("daily limit" in detail.lower())
            ) or retry_wait > 30:
                raise LLMQuotaError(
                    "The language-model provider's quota is used up."
                ) from error
            json_error = (
                status == 400
                and "json" in detail.lower()
            )
            unsupported = (
                status == 400
                and "json mode" in detail.lower()
                and (
                    "not supported" in detail.lower()
                    or "unsupported" in detail.lower()
                )
            )
            if unsupported:
                _json_mode_disabled = True
                json_mode = False

            if json_error and use_json_mode:
                try:
                    response = make_request(False)
                    _consecutive_json_400 = 0
                    json_mode = True
                    content = response.choices[0].message.content
                    if content:
                        return content
                    raise LLMError("LLM returned an empty response.")
                except openai.APIStatusError as retry_error:
                    retry_status = getattr(retry_error, "status_code", None)
                    retry_detail = str(retry_error)
                    if (
                        retry_status == 400
                        and "json" in retry_detail.lower()
                    ):
                        _consecutive_json_400 += 1
                        if _consecutive_json_400 >= 2:
                            _json_mode_disabled = True
                            json_mode = False
                    raise LLMError(
                        f"LLM request failed ({retry_status}) on provider "
                        f"'{settings.llm_provider}': {retry_error.message}."
                    ) from retry_error
                except openai.APIConnectionError as retry_error:
                    raise LLMError(
                        f"Could not reach LLM provider '{settings.llm_provider}' "
                        f"at {settings.llm_base_url}. Is it running or reachable?"
                    ) from retry_error

            raise LLMError(
                f"LLM request failed ({status}) on provider "
                f"'{settings.llm_provider}': {error.message}."
            ) from error
        except openai.APIConnectionError as error:
            raise LLMError(
                f"Could not reach LLM provider '{settings.llm_provider}' "
                f"at {settings.llm_base_url}. Is it running or reachable?"
            ) from error

        content = response.choices[0].message.content
        if content:
            _consecutive_json_400 = 0
            return content
        last_error = LLMError("LLM returned an empty response.")
        if attempt < 2:
            time.sleep(1)

    raise last_error

def generate_structured(
    prompt: str,
    schema: type[T],
    system: str | None = None,
    max_retries: int = 1,
) -> T:
    """Ask the configured LLM for JSON matching a Pydantic schema."""
    schema_json = json.dumps(schema.model_json_schema())
    system_msg = (system + "\n\n" if system else "") + (
        "Respond with a single JSON object that matches this JSON schema. "
        "No prose and no markdown fences.\n" + schema_json
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception = LLMError("LLM returned an empty response.")
    for _ in range(max_retries + 1):
        raw = _complete_with_openai_compatible(messages)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except (ValidationError, ValueError) as error:
            last_error = error
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"That response was invalid: {error}. "
                    "Return a corrected JSON object only.",
                }
            )

    raise LLMError(
        f"LLM output failed validation for {schema.__name__} "
        f"after {max_retries + 1} attempts: {last_error}"
    )
