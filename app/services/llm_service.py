"""LLM calls.

openrouter (default, free models), openai_compatible (gemini/groq/openai/local) or anthropic.
All of them return LLMAnalysis. If a model fails with 429/5xx/timeout/bad json we move on to
the next model in the chain.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import httpx
from pydantic import ValidationError

from app.config import Settings
from app.errors import LLMError, LLMNotConfiguredError
from app.schemas import LLMAnalysis
from app.services.pricing import estimate_cost

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    analysis: LLMAnalysis
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    fallback_used: bool


class AnalysisLLM(Protocol):

    async def analyze(self, system_prompt: str, user_prompt: str) -> LLMResult: ...


class _RetryableFailure(Exception):
    """try the next model"""


class _BaseLLM:
    provider = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def models(self) -> list[str]:
        return self.settings.model_chain

    async def analyze(self, system_prompt: str, user_prompt: str) -> LLMResult:
        last_error: Exception | None = None
        models = self.models
        for attempt, model in enumerate(models):
            try:
                return await self._call(model, system_prompt, user_prompt, fallback_used=attempt > 0)
            except _RetryableFailure as exc:
                last_error = exc
                logger.warning("LLM %s/%s failed (%s); %s", self.provider, model, exc,
                               "trying fallback" if attempt + 1 < len(models) else "no fallback left")
        raise LLMError(
            "The AI analysis service is temporarily unavailable. Please retry shortly.",
            details={"reason": str(last_error)},
        )

    async def _call(self, model: str, system_prompt: str, user_prompt: str, *, fallback_used: bool) -> LLMResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


# claude
_ANTHROPIC_TRANSIENT: tuple[type[Exception], ...] = (
    anthropic.APIConnectionError,  # timeouts too
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class AnthropicLLM(_BaseLLM):
    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if settings.anthropic_api_key is None:
            raise LLMNotConfiguredError("ANTHROPIC_API_KEY is not set. Add it to your environment or .env file.")
        self.client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def aclose(self) -> None:
        await self.client.close()

    async def _call(self, model: str, system_prompt: str, user_prompt: str, *, fallback_used: bool) -> LLMResult:
        started = time.perf_counter()
        try:
            response = await self.client.messages.parse(
                model=model,
                max_tokens=self.settings.resolved_max_tokens,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_prompt}],
                output_config={"effort": self.settings.llm_effort},
                output_format=LLMAnalysis,
            )
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMNotConfiguredError("LLM provider rejected the API credentials.") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError("LLM provider rejected the request.", details={"reason": exc.message}) from exc
        except _ANTHROPIC_TRANSIENT as exc:
            raise _RetryableFailure(f"{type(exc).__name__}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code == 429:
                raise _RetryableFailure(f"HTTP {exc.status_code}") from exc
            raise LLMError("LLM provider returned an error.", details={"status": exc.status_code}) from exc
        except ValueError as exc:  # output didn't match schema
            raise _RetryableFailure(f"Invalid structured output: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.stop_reason == "refusal":
            raise _RetryableFailure("Model declined the request (refusal).")
        if response.stop_reason == "max_tokens":
            raise _RetryableFailure("Output hit max_tokens before the JSON was complete.")
        parsed = response.parsed_output
        if parsed is None:
            raise _RetryableFailure("Model returned no structured output.")

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        logger.info(
            "LLM %s ok in %dms: in=%d cache_write=%d cache_read=%d out=%d",
            model, latency_ms, usage.input_tokens, cache_write, cache_read, usage.output_tokens,
        )
        return LLMResult(
            analysis=parsed,
            provider=self.provider,
            model=model,
            input_tokens=usage.input_tokens + cache_write,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=cache_read,
            estimated_cost_usd=estimate_cost(model, usage.input_tokens, usage.output_tokens, cache_read, cache_write),
            latency_ms=latency_ms,
            fallback_used=fallback_used,
        )


# openrouter / openai style
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """some models wrap the json in ```json or <think> tags, dig it out"""
    text = _THINK_BLOCK.sub("", text or "").strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the model response.")
    return text[start : end + 1]


def analysis_json_schema() -> dict[str, Any]:
    return LLMAnalysis.model_json_schema()


class OpenAICompatibleLLM(_BaseLLM):

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(settings)
        self.provider = settings.llm_provider
        if settings.llm_api_key is None:
            raise LLMNotConfiguredError(
                "LLM_API_KEY is not set. Create a free key at https://openrouter.ai/keys and add it to .env."
            )
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["X-Title"] = settings.app_name
        self.client = client or httpx.AsyncClient(
            base_url=settings.resolved_base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
        )
        self._schema = analysis_json_schema()

    async def aclose(self) -> None:
        await self.client.aclose()

    def _body(self, model: str, messages: list[dict], *, strict_schema: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.settings.resolved_max_tokens,
            "temperature": 0.3,
        }
        if strict_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "link_analysis", "strict": True, "schema": self._schema},
            }
        else:
            body["response_format"] = {"type": "json_object"}
        if self.provider == "openrouter":
            # otherwise openrouter may send us to a host that ignores the schema
            body["provider"] = {"require_parameters": True}
        return body

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self.client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise _RetryableFailure("LLM request timed out.") from exc
        except httpx.HTTPError as exc:
            raise _RetryableFailure(f"Connection error: {type(exc).__name__}") from exc

        if resp.status_code in (401, 403):
            raise LLMNotConfiguredError("LLM provider rejected the API key (check LLM_API_KEY).")
        if resp.status_code == 402:
            raise LLMNotConfiguredError("LLM provider account has insufficient credits for this model.")
        if resp.status_code == 429:
            raise _RetryableFailure("Rate limited by the LLM provider (free-tier models have per-minute/day limits).")
        if resp.status_code >= 500:
            raise _RetryableFailure(f"LLM provider error HTTP {resp.status_code}.")
        if resp.status_code >= 400:
            raise _BadRequest(resp.text[:500])
        try:
            data = resp.json()
        except ValueError as exc:
            raise _RetryableFailure("LLM provider returned a non-JSON response.") from exc
        if isinstance(data, dict) and data.get("error"):  # openrouter sometimes sends 200 with an error inside
            raise _RetryableFailure(f"Provider error: {str(data['error'])[:200]}")
        return data

    async def _call(self, model: str, system_prompt: str, user_prompt: str, *, fallback_used: bool) -> LLMResult:
        started = time.perf_counter()
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            data = await self._post(self._body(model, messages, strict_schema=True))
            strict = True
        except _BadRequest as exc:
            # json_schema not supported here, fall back to json mode
            logger.info("Model %s rejected json_schema (%s); retrying in JSON mode", model, exc)
            strict = False
            messages[0]["content"] = system_prompt + self._schema_instruction()
            try:
                data = await self._post(self._body(model, messages, strict_schema=False))
            except _BadRequest as exc2:
                raise LLMError("LLM provider rejected the request.", details={"reason": str(exc2)}) from exc2

        input_tokens = output_tokens = 0
        for repair_round in range(2):  # one retry asking it to fix its json
            usage = data.get("usage") or {}
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            finish = choice.get("finish_reason")
            try:
                analysis = LLMAnalysis.model_validate_json(extract_json(content))
                break
            except (ValueError, ValidationError) as exc:
                if finish == "length":
                    raise _RetryableFailure("Output hit max_tokens before the JSON was complete.") from exc
                if repair_round == 1:
                    raise _RetryableFailure(f"Invalid structured output after repair: {str(exc)[:300]}") from exc
                logger.info("Model %s returned invalid JSON; asking it to repair once", model)
                messages += [
                    {"role": "assistant", "content": content[:20000]},
                    {"role": "user", "content": "That response did not match the required JSON schema:\n"
                     f"{str(exc)[:1500]}\nReturn the complete corrected JSON object only."},
                ]
                try:
                    data = await self._post(self._body(model, messages, strict_schema=strict))
                except _BadRequest as exc2:
                    raise _RetryableFailure(f"Repair request rejected: {exc2}") from exc2

        latency_ms = int((time.perf_counter() - started) * 1000)
        served_model = data.get("model") or model
        logger.info("LLM %s/%s ok in %dms: in=%d out=%d", self.provider, served_model, latency_ms,
                    input_tokens, output_tokens)
        return LLMResult(
            analysis=analysis,
            provider=self.provider,
            model=served_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            estimated_cost_usd=estimate_cost(served_model, input_tokens, output_tokens),
            latency_ms=latency_ms,
            fallback_used=fallback_used,
        )

    def _schema_instruction(self) -> str:
        return (
            "\n\n## Output format\nRespond with ONE JSON object (no markdown, no commentary) that validates "
            "against this JSON Schema. Every property is required; use null where the schema allows it.\n"
            + json.dumps(self._schema, separators=(",", ":"))
        )


class _BadRequest(Exception):
    pass


def create_llm(settings: Settings) -> _BaseLLM:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    return OpenAICompatibleLLM(settings)
