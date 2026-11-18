"""Settings, read from env vars or .env"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# provider -> (main model, fallbacks)
DEFAULT_MODELS: dict[str, tuple[str, str | None]] = {
    # picked these after running scripts/benchmark_models.py (see docs/benchmarks)
    # ernie was fastest, nemotron never failed. openrouter/free is the last resort
    "openrouter": ("baidu/ernie-5.5-flash-preview:free", "nvidia/nemotron-3.5-super-vl:free,openrouter/free"),
    "anthropic": ("claude-haiku-5-5", "claude-sonnet-5-5"),
    "openai_compatible": ("gpt-6-luna", None),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # app
    app_name: str = "AI Link Analysis API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # llm
    # openrouter = free models, anthropic = claude, openai_compatible = gemini/groq/openai etc
    llm_provider: Literal["openrouter", "anthropic", "openai_compatible"] = "openrouter"
    anthropic_api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None  # openrouter or openai_compatible key
    llm_base_url: str | None = None
    llm_model: str | None = None  # empty = use DEFAULT_MODELS
    llm_fallback_model: str | None = None  # comma separated, "none" turns it off
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"  # only used for claude
    # free models burn 10-20k tokens just thinking, 16k was cutting the json off so 32k there
    llm_max_tokens: int | None = None
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2

    @property
    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider][0]

    @property
    def resolved_fallback_model(self) -> str | None:
        if self.llm_fallback_model is not None:
            value = self.llm_fallback_model.strip()
            return None if value.lower() in ("", "none") else value
        return DEFAULT_MODELS[self.llm_provider][1]

    @property
    def model_chain(self) -> list[str]:
        """main model first, then fallbacks (no dupes)"""
        chain = [self.resolved_model]
        for m in (self.resolved_fallback_model or "").split(","):
            m = m.strip()
            if m and m not in chain:
                chain.append(m)
        return chain

    @property
    def resolved_max_tokens(self) -> int:
        if self.llm_max_tokens:
            return self.llm_max_tokens
        return 16000 if self.llm_provider == "anthropic" else 32000

    @property
    def resolved_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        if self.llm_provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        return "https://api.openai.com/v1"

    # fetching
    fetch_timeout_seconds: float = 12.0
    fetch_max_bytes: int = 3_000_000
    fetch_max_redirects: int = 5
    fetch_max_subpages: int = 4
    fetch_user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36 LinkAnalysisBot/1.0"
    )
    # roughly 7.5k tokens of page text max
    content_char_budget: int = 30_000
    allowed_ports: list[int] = Field(default_factory=lambda: [80, 443, 8080, 8443])

    # competitors
    verify_competitor_websites: bool = True
    competitor_check_timeout_seconds: float = 5.0

    # cache / rate limit
    cache_ttl_seconds: int = 6 * 60 * 60
    cache_max_entries: int = 256
    rate_limit_per_minute: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
