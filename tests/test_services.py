"""Unit tests for extraction helpers, LLM fallback logic, pricing, evidence matching and caching."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from bs4 import BeautifulSoup

from app.config import Settings
from app.errors import LLMError, LLMNotConfiguredError
from app.services.analysis_service import _normalize_text, evidence_in_corpus
from app.services.llm_service import AnthropicLLM
from app.services.pricing import estimate_cost
from app.services.website_extractor import _candidate_subpages, _parse_json_ld, build_seo_audit
from app.utils.cache import TTLCache
from tests.conftest import HOME_HTML, sample_llm_analysis


# Lextraction
def test_subpage_candidates_prioritise_business_pages() -> None:
    soup = BeautifulSoup(HOME_HTML, "lxml")
    picked = _candidate_subpages(soup, "https://acme.example/", limit=4)
    assert picked[0] == "https://acme.example/services"
    assert "https://acme.example/about" in picked
    assert not any("login" in p or p.endswith(".pdf") for p in picked)


def test_json_ld_organization_parsed() -> None:
    data = _parse_json_ld(BeautifulSoup(HOME_HTML, "lxml"))
    assert "Organization" in data["types"]
    assert data["organization"]["name"] == "Acme Web Studio"


def test_seo_audit_flags_real_problems() -> None:
    html = "<html><head><title>Hi</title></head><body><h1>A</h1><h1>B</h1><img src=x></body></html>"
    audit = build_seo_audit(BeautifulSoup(html, "lxml"), "http://x.example/", len(html))
    messages = " ".join(i.message for i in audit.issues)
    for expected in ("Title is short", "Missing meta description", "2 <h1>", "no alt text",
                     "viewport", "HTTPS", "lang attribute", "structured data"):
        assert expected in messages, expected
    assert audit.uses_https is False


# Levidence matching
@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ("Custom website design and development", True),
        ("custom WEBSITE design & development!", True),
        ("We guarantee page one rankings", False),
        ("", False),
    ],
)
def test_evidence_matching(evidence: str, expected: bool) -> None:
    corpus = _normalize_text("Custom website design and development built on modern frameworks.")
    assert evidence_in_corpus(evidence, corpus) is expected


# Lpricing
def test_cost_estimate_haiku() -> None:
    # 8K in @ $0.10/M + 4K out @ $0.50/M = $0.0008 + $0.002
    assert estimate_cost("claude-haiku-5-5", 8000, 4000) == pytest.approx(0.0028)


def test_cost_estimate_long_prompt_rate_card() -> None:
    assert estimate_cost("claude-haiku-5-5", 200_000, 1000) == pytest.approx(0.1 + 0.0025)


def test_cost_unknown_model_is_zero() -> None:
    assert estimate_cost("some-other-model", 1000, 1000) == 0.0


# Lcache
def test_cache_dedupes_concurrent_requests() -> None:
    cache: TTLCache[str] = TTLCache(ttl_seconds=60, max_entries=5)
    calls = 0

    async def compute() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "result"

    async def run():
        return await asyncio.gather(*(cache.get_or_compute("k", compute) for _ in range(5)))

    results = asyncio.run(run())
    assert calls == 1
    assert [r[0] for r in results] == ["result"] * 5
    assert sum(1 for _, cached in results if not cached) == 1


def test_cache_lru_eviction() -> None:
    cache: TTLCache[int] = TTLCache(ttl_seconds=60, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1


# LLLM fallback
def _api_error(cls, status: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


def _ok_response():
    return SimpleNamespace(
        stop_reason="end_turn",
        parsed_output=sample_llm_analysis(),
        usage=SimpleNamespace(input_tokens=7000, output_tokens=3500,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class _FakeMessages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.models: list[str] = []

    async def parse(self, **kwargs):
        self.models.append(kwargs["model"])
        assert kwargs["output_config"] == {"effort": "low"}
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _llm(outcomes) -> tuple[AnthropicLLM, _FakeMessages]:
    llm = AnthropicLLM(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="test-key", llm_model="claude-haiku-5-5",
                                llm_fallback_model="claude-sonnet-5-5", llm_effort="low"))
    fake = _FakeMessages(outcomes)
    llm.client = SimpleNamespace(messages=fake)  # type: ignore[assignment]
    return llm, fake


def test_llm_success_reports_usage_and_cost() -> None:
    llm, fake = _llm([_ok_response()])
    result = asyncio.run(llm.analyze("sys", "user"))
    assert fake.models == ["claude-haiku-5-5"]
    assert result.fallback_used is False
    assert result.estimated_cost_usd == pytest.approx(7000 * 0.1e-6 + 3500 * 0.5e-6)


@pytest.mark.parametrize(
    "first",
    [
        _api_error(anthropic.InternalServerError, 500),
        _api_error(anthropic.RateLimitError, 429),
        SimpleNamespace(stop_reason="refusal", parsed_output=None, usage=None),
        SimpleNamespace(stop_reason="max_tokens", parsed_output=None, usage=None),
    ],
)
def test_llm_falls_back_to_secondary_model(first) -> None:
    llm, fake = _llm([first, _ok_response()])
    result = asyncio.run(llm.analyze("sys", "user"))
    assert fake.models == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert result.fallback_used is True
    assert result.model == "claude-sonnet-5-5"


def test_llm_all_models_fail() -> None:
    llm, _ = _llm([_api_error(anthropic.InternalServerError, 500)] * 2)
    with pytest.raises(LLMError):
        asyncio.run(llm.analyze("sys", "user"))


def test_llm_auth_error_does_not_fall_back() -> None:
    llm, fake = _llm([_api_error(anthropic.AuthenticationError, 401)])
    with pytest.raises(LLMNotConfiguredError):
        asyncio.run(llm.analyze("sys", "user"))
    assert fake.models == ["claude-haiku-5-5"]


def test_missing_api_key() -> None:
    with pytest.raises(LLMNotConfiguredError):
        AnthropicLLM(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key=None))
