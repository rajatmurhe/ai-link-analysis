"""Token prices in USD per 1M tokens, so each response can show what it cost.

from https://platform.claude.com/docs/en/about-claude/pricing (18 nov 2026). free models = $0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    # haiku costs more above 100k input tokens
    long_context_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None


PRICES: dict[str, ModelPrice] = {
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.01, 100_000, 0.50, 2.50),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00, 0.10),
    "claude-opus-5-5": ModelPrice(4.00, 20.00, 0.20),
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0
) -> float:
    """input_tokens doesn't include cache tokens (same as anthropic usage)"""
    if model.endswith(":free") or model == "openrouter/free":
        return 0.0
    price = PRICES.get(model)
    if price is None:
        return 0.0  # don't know the price
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    in_rate, out_rate = price.input, price.output
    if price.long_context_threshold and prompt_tokens > price.long_context_threshold:
        in_rate, out_rate = price.long_input or in_rate, price.long_output or out_rate
    cost = (
        input_tokens * in_rate
        + cache_write_tokens * in_rate * 1.25
        + cache_read_tokens * price.cache_read
        + output_tokens * out_rate
    ) / 1_000_000
    return round(cost, 6)
