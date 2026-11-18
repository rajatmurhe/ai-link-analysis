"""Benchmark candidate models on real websites with the production pipeline.

Usage (uses LLM_PROVIDER / keys from .env):
    python -m scripts.benchmark_models --models nvidia/nemotron-3.5-super-vl:free openrouter/free
    LLM_PROVIDER=anthropic python -m scripts.benchmark_models \
        --models claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5

Each site is extracted once; the identical prompt is then sent to every model. For each run we
record *measured* latency, tokens and cost plus objective grounding metrics:

* evidence_verified_rate - share of "website_content" services whose quote exists on the site
* competitors_reachable  - suggested competitor homepages that actually respond
* schema_valid           - structured output parsed and validated

Writes docs/benchmarks/results-<effort>.json and .md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.services.analysis_service import _normalize_text, evidence_in_corpus  # noqa: E402
from app.services.competitor_service import CompetitorService  # noqa: E402
from app.services.http_fetcher import SafeFetcher  # noqa: E402
from app.services.llm_service import create_llm  # noqa: E402
from app.services.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from app.services.website_extractor import WebsiteExtractor  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"


async def run(models: list[str], sites: list[str], effort: str) -> list[dict]:
    base = Settings()
    fetcher = SafeFetcher(base)
    extractor = WebsiteExtractor(base, fetcher)
    competitors = CompetitorService(fetcher, verify=True, timeout=6)
    rows: list[dict] = []
    try:
        for site_url in sites:
            try:
                site = await extractor.extract(site_url)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {site_url}: {exc}", flush=True)
                continue
            prompt = build_user_prompt(site.to_llm_context(), site.seo_audit)
            corpus = _normalize_text(site.corpus)
            async def bench(model: str) -> dict:
                settings = base.model_copy(update={"llm_model": model, "llm_fallback_model": "none", "llm_effort": effort})
                llm = create_llm(settings)
                started = time.perf_counter()
                row: dict = {"site": site.final_url, "model": model, "effort": effort}
                try:
                    res = await llm.analyze(SYSTEM_PROMPT, prompt)
                    a = res.analysis
                    on_site = [s for s in a.top_services if s.source == "website_content"]
                    verified = sum(1 for s in on_site if evidence_in_corpus(s.evidence, corpus))
                    cleaned = competitors.clean(a.competitor_analysis.competitors, a.brand.name, site.domain)
                    enriched = await competitors.enrich(cleaned)
                    row.update(
                        schema_valid=True,
                        served_model=res.model,
                        latency_s=round(res.latency_ms / 1000, 2),
                        input_tokens=res.input_tokens,
                        output_tokens=res.output_tokens,
                        cost_usd=res.estimated_cost_usd,
                        brand=a.brand.name,
                        category=a.brand.category,
                        services=[s.name for s in a.top_services],
                        services_on_site=len(on_site),
                        evidence_verified_rate=round(verified / len(on_site), 2) if on_site else None,
                        competitors=[c.name for c in enriched],
                        competitors_reachable=sum(1 for c in enriched if c.website_status == "reachable"),
                        comparison_rows_filled=sum(1 for r in a.competitor_analysis.comparison_table if r.competitors),
                        comparison_rows=len(a.competitor_analysis.comparison_table),
                        queries=len(a.target_search_queries),
                        platforms_covered=len({q.primary_platform for q in a.target_search_queries}),
                        title_len=len(a.seo_improvements.recommended_title),
                        meta_len=len(a.seo_improvements.meta_description),
                    )
                except Exception as exc:  # noqa: BLE001
                    row.update(schema_valid=False, error=str(getattr(exc, "details", "") or exc),
                               latency_s=round(time.perf_counter() - started, 2))
                finally:
                    await llm.aclose()
                print(json.dumps(row), flush=True)
                return row

            # run all models at once on the same prompt
            rows.extend(await asyncio.gather(*(bench(m) for m in models)))
    finally:
        await fetcher.aclose()
    return rows


def summarise(rows: list[dict]) -> str:
    lines = [
        f"# Model benchmark - measured {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "| Model | Runs | Valid JSON | Median latency (s) | Avg input tok | Avg output tok | Avg cost (USD) "
        "| Evidence verified | Competitors reachable (avg) | Platforms covered (of 5) | Title 50-60 chars | Meta 140-160 chars |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        ok = [r for r in rs if r.get("schema_valid")]
        if not ok:
            lines.append(f"| {model} | {len(rs)} | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        ev = [r["evidence_verified_rate"] for r in ok if r["evidence_verified_rate"] is not None]
        lines.append(
            f"| {model} | {len(rs)} | {len(ok)}/{len(rs)} "
            f"| {statistics.median(r['latency_s'] for r in ok):.1f} "
            f"| {statistics.mean(r['input_tokens'] for r in ok):,.0f} "
            f"| {statistics.mean(r['output_tokens'] for r in ok):,.0f} "
            f"| ${statistics.mean(r['cost_usd'] for r in ok):.4f} "
            f"| {statistics.mean(ev) * 100:.0f}% "
            f"| {statistics.mean(r['competitors_reachable'] for r in ok):.1f} / "
            f"{statistics.mean(len(r['competitors']) for r in ok):.1f} "
            f"| {statistics.mean(r['platforms_covered'] for r in ok):.1f} "
            f"| {sum(1 for r in ok if 50 <= r['title_len'] <= 60)}/{len(ok)} "
            f"| {sum(1 for r in ok if 140 <= r['meta_len'] <= 160)}/{len(ok)} |"
        )
    lines += ["", "## Per-run results", "", "| Site | Model | Latency (s) | Cost (USD) | Brand | Competitors |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("schema_valid"):
            lines.append(f"| {r['site']} | {r['model']} | {r['latency_s']} | ${r['cost_usd']:.4f} | {r['brand']} "
                         f"| {', '.join(r['competitors'])} |")
        else:
            lines.append(f"| {r['site']} | {r['model']} | {r['latency_s']} | - | ERROR: {r.get('error', '')[:80]} | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="defaults to the configured primary model")
    parser.add_argument("--sites", nargs="+", default=[
        "https://stripe.com", "https://basecamp.com", "https://www.python.org",
        "https://www.allbirds.com", "https://www.hubspot.com",
    ])
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    args = parser.parse_args()

    models = args.models or [Settings().resolved_model]
    rows = asyncio.run(run(models, args.sites, args.effort))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"results-{args.effort}.json").write_text(json.dumps(rows, indent=2))
    (OUT_DIR / f"results-{args.effort}.md").write_text(summarise(rows))
    print(summarise(rows))


if __name__ == "__main__":
    main()
