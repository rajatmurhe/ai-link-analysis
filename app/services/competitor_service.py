"""Cleans up the competitors the llm suggests.

The model sometimes lists the brand itself or the same company twice, and the websites
can be made up, so we remove those and ping each site to see if it's real.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from app.schemas import Competitor, LLMCompetitor
from app.services.http_fetcher import SafeFetcher

COMPETITOR_DISCLAIMER = (
    "Competitors are AI-generated suggestions based on the website content and the model's general "
    "market knowledge, not live market research. Each suggested website was checked for "
    "reachability (`website_status`); validate positioning claims before making business decisions."
)


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlsplit(url if "://" in url else f"https://{url}").hostname or ""
    except ValueError:
        return None
    host = host.lower()
    return host[4:] if host.startswith("www.") else host or None


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


class CompetitorService:
    def __init__(self, fetcher: SafeFetcher, *, verify: bool, timeout: float) -> None:
        self.fetcher = fetcher
        self.verify = verify
        self.timeout = timeout

    def clean(self, competitors: list[LLMCompetitor], brand_name: str, brand_domain: str) -> list[LLMCompetitor]:
        brand_key = _norm_name(brand_name)
        seen: set[str] = set()
        out: list[LLMCompetitor] = []
        for comp in competitors:
            key = _norm_name(comp.name)
            dom = _domain(comp.website)
            if not key or key == brand_key or (dom and (dom == brand_domain or dom.endswith("." + brand_domain))):
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(comp)
        return out

    async def enrich(self, competitors: list[LLMCompetitor]) -> list[Competitor]:
        async def check(comp: LLMCompetitor) -> Competitor:
            data = comp.model_dump()
            if not comp.website:
                return Competitor(**data, website_status="not_provided")
            if not self.verify:
                return Competitor(**data, website_status="not_checked")
            ok, final_url = await self.fetcher.probe(comp.website, timeout=self.timeout)
            if ok and final_url:
                data["website"] = final_url
            return Competitor(**data, website_status="reachable" if ok else "unreachable")

        return list(await asyncio.gather(*(check(c) for c in competitors)))
