"""OpenAI-compatible provider (OpenRouter / Gemini / Groq ...) tested against a mock HTTP transport."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.errors import LLMError, LLMNotConfiguredError
from app.services.llm_service import OpenAICompatibleLLM, create_llm, extract_json
from tests.conftest import sample_llm_analysis

GOOD_JSON = sample_llm_analysis().model_dump_json()


def _completion(content: str, model: str = "primary/model:free", finish: str = "stop") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 9000, "completion_tokens": 4500},
    }


def _llm(responses: list, **overrides) -> tuple[OpenAICompatibleLLM, list[dict]]:
    """responses = list of (status, body), returned in order"""
    queue = list(responses)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        status, body = queue.pop(0)
        return httpx.Response(status, json=body)

    overrides.setdefault("llm_model", "primary/model:free")
    overrides.setdefault("llm_fallback_model", "openrouter/free")
    settings = Settings(_env_file=None, llm_provider="openrouter", llm_api_key="or-test", **overrides)
    client = httpx.AsyncClient(base_url=settings.resolved_base_url, transport=httpx.MockTransport(handler))
    return OpenAICompatibleLLM(settings, client=client), seen


def test_defaults_point_to_free_openrouter_models() -> None:
    s = Settings(_env_file=None)
    assert s.llm_provider == "openrouter"
    assert s.resolved_model.endswith(":free")
    assert len(s.model_chain) == 3 and s.model_chain[-1] == "openrouter/free"
    assert s.resolved_base_url == "https://openrouter.ai/api/v1"
    assert s.resolved_max_tokens == 32000
    assert Settings(_env_file=None, llm_provider="anthropic").resolved_max_tokens == 16000
    assert Settings(_env_file=None, llm_fallback_model="none").model_chain == [s.resolved_model]
    assert Settings(_env_file=None, llm_model="a", llm_fallback_model="b, a ,c").model_chain == ["a", "b", "c"]


def test_strict_json_schema_success() -> None:
    llm, seen = _llm([(200, _completion(GOOD_JSON))])
    result = asyncio.run(llm.analyze("sys", "user"))
    body = seen[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}
    assert result.analysis.brand.name == "Acme Web Studio"
    assert result.provider == "openrouter"
    assert result.estimated_cost_usd == 0.0  # free model
    assert (result.input_tokens, result.output_tokens) == (9000, 4500)


def test_falls_back_to_json_mode_when_schema_rejected() -> None:
    llm, seen = _llm([(400, {"error": {"message": "response_format json_schema not supported"}}),
                      (200, _completion(GOOD_JSON))])
    asyncio.run(llm.analyze("sys", "user"))
    assert seen[1]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in seen[1]["messages"][0]["content"]


def test_repairs_invalid_json_once() -> None:
    llm, seen = _llm([(200, _completion('{"brand": "incomplete"}')), (200, _completion(GOOD_JSON))])
    result = asyncio.run(llm.analyze("sys", "user"))
    assert len(seen) == 2
    assert seen[1]["messages"][-1]["role"] == "user"
    assert "did not match" in seen[1]["messages"][-1]["content"]
    assert result.input_tokens == 18000  # usage summed across both calls


def test_rate_limit_moves_to_fallback_model() -> None:
    llm, seen = _llm([(429, {"error": {"message": "rate limited"}}),
                      (200, _completion(GOOD_JSON, model="some/free-model:free"))])
    result = asyncio.run(llm.analyze("sys", "user"))
    assert [b["model"] for b in seen] == ["primary/model:free", "openrouter/free"]
    assert result.fallback_used is True
    assert result.model == "some/free-model:free"  # the model OpenRouter actually served


def test_error_body_with_200_status_is_retried() -> None:
    llm, _ = _llm([(200, {"error": {"message": "upstream overloaded"}}), (200, _completion(GOOD_JSON))])
    assert asyncio.run(llm.analyze("sys", "user")).fallback_used is True


def test_truncated_output_triggers_fallback() -> None:
    llm, _ = _llm([(200, _completion('{"brand": {', finish="length")), (200, _completion(GOOD_JSON))])
    assert asyncio.run(llm.analyze("sys", "user")).fallback_used is True


def test_walks_the_whole_fallback_chain() -> None:
    llm, seen = _llm([(503, {}), (429, {}), (200, _completion(GOOD_JSON))], llm_fallback_model="second:free,third:free")
    assert asyncio.run(llm.analyze("sys", "user")).fallback_used is True
    assert [b["model"] for b in seen] == ["primary/model:free", "second:free", "third:free"]
    assert seen[0]["max_tokens"] == 32000


def test_all_models_failing_raises_llm_error() -> None:
    llm, _ = _llm([(503, {}), (503, {})])
    with pytest.raises(LLMError):
        asyncio.run(llm.analyze("sys", "user"))


def test_bad_api_key() -> None:
    llm, seen = _llm([(401, {"error": {"message": "No auth"}})])
    with pytest.raises(LLMNotConfiguredError):
        asyncio.run(llm.analyze("sys", "user"))
    assert len(seen) == 1  # no pointless fallback on auth errors


def test_missing_key_and_factory() -> None:
    with pytest.raises(LLMNotConfiguredError):
        create_llm(Settings(_env_file=None, llm_provider="openrouter", llm_api_key=None))
    llm = create_llm(Settings(_env_file=None, llm_provider="openai_compatible", llm_api_key="k",
                              llm_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                              llm_model="gemini-3.5-flash-lite"))
    assert isinstance(llm, OpenAICompatibleLLM)
    assert str(llm.client.base_url).startswith("https://generativelanguage.googleapis.com/v1beta/openai")
    assert llm.models == ["gemini-3.5-flash-lite"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '<think>reasoning {not json}</think>\nHere you go: {"a": 1} hope it helps',
    ],
)
def test_extract_json(raw: str) -> None:
    assert json.loads(extract_json(raw)) == {"a": 1}
