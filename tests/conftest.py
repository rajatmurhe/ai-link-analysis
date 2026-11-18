from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import WebsiteHTTPError
from app.main import app
from app.schemas import LLMAnalysis
from app.services import llm_service as llm_module
from app.services.analysis_service import AnalysisService
from app.services.competitor_service import CompetitorService
from app.services.http_fetcher import FetchResult
from app.services.llm_service import LLMResult
from app.services.website_extractor import WebsiteExtractor
from app.utils import validators
from app.utils.cache import TTLCache

PUBLIC_IP = "93.184.216.34"
# Captured before the autouse fixture patches it, for tests that exercise real resolution logic.
REAL_RESOLVE_PUBLIC_IP = validators.resolve_public_ip

HOME_HTML = """<!doctype html>
<html lang="en"><head>
<title>Acme Web Studio</title>
<meta name="description" content="Websites for small businesses.">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Organization","name":"Acme Web Studio",
 "address":{"@type":"PostalAddress","addressLocality":"Pune","addressCountry":"IN"},
 "sameAs":["https://www.linkedin.com/company/acme-web"]}
</script>
</head><body>
<div id="cookie-banner">We use cookies to improve your experience. Accept all cookies</div>
<header><nav><a href="/">Home</a><a href="/services">Services</a><a href="/about">About us</a>
<a href="/pricing">Pricing</a><a href="/login">Login</a><a href="/brochure.pdf">Brochure</a></nav></header>
<main>
<h1>Websites that grow small businesses</h1>
<p>Acme Web Studio designs fast, mobile-friendly websites for small businesses and startups across India.</p>
<h2>What we do</h2>
<p>Custom website design and development built on modern frameworks.</p>
<p>E-commerce stores with Shopify and WooCommerce integration.</p>
<p>Search engine optimisation packages that bring local customers to your door.</p>
<p>Website maintenance plans with 24/7 monitoring and security updates.</p>
<p>Brand identity and logo design for new ventures.</p>
<img src="/a.png"><img src="/b.png" alt="Team photo">
</main>
<footer><p>Acme Web Studio, Baner Road, Pune, India</p>
<a href="https://www.instagram.com/acmeweb">Instagram</a>
<a href="https://www.facebook.com/sharer/sharer.php?u=x">Share</a></footer>
</body></html>"""

SERVICES_HTML = """<html><head><title>Services | Acme</title></head><body>
<header><nav><a href="/">Home</a><a href="/services">Services</a></nav></header>
<main><h1>Our services</h1>
<p>Custom website design and development built on modern frameworks.</p>
<p>Conversion rate optimisation audits for online stores, with a detailed action plan and A/B test roadmap.</p>
<p>Website maintenance plans with 24/7 monitoring and security updates.</p>
<p>We have helped over 120 local businesses launch their first website in under four weeks.</p>
</main>
<footer><p>Acme Web Studio, Baner Road, Pune, India</p></footer></body></html>"""

ABOUT_HTML = """<html><head><title>About | Acme</title></head><body><main>
<h1>About Acme</h1><p>Founded in 2018 by two engineers who wanted to make great websites affordable for local shops.</p>
<p>Our team of 12 designers and developers works with clients across Maharashtra.</p>
</main></body></html>"""


class FakeFetcher:
    """fake fetcher, 404 for unknown urls. keeps track of calls"""

    def __init__(self, pages: dict[str, str] | None = None, reachable: set[str] | None = None) -> None:
        self.pages = pages or {
            "https://acme.example/": HOME_HTML,
            "https://acme.example/services": SERVICES_HTML,
            "https://acme.example/about": ABOUT_HTML,
        }
        self.reachable = reachable or set()
        self.calls: list[str] = []

    async def fetch(self, url: str, **_: object) -> FetchResult:
        self.calls.append(url)
        html = self.pages.get(url)
        if html is None:
            raise WebsiteHTTPError("Website returned HTTP 404.", details={"status_code": 404})
        return FetchResult(url, url, 200, "text/html", html, 5)

    async def probe(self, url: str, timeout: float) -> tuple[bool, str | None]:
        ok = any(url.startswith(r) for r in self.reachable)
        return ok, (url if ok else None)

    async def aclose(self) -> None:  # pragma: no cover
        pass


def sample_llm_analysis() -> LLMAnalysis:
    competitor = {
        "website": None,
        "description": "Web agency.",
        "main_services": ["Web design"],
        "target_market": "SMBs",
        "strengths": ["Established brand"],
        "weaknesses_or_gaps": ["Higher prices"],
        "confidence": "medium",
    }
    service = {
        "description": "d",
        "target_industry": "Retail",
        "target_customer": "Small business owners",
        "main_benefit": "More customers",
    }
    return LLMAnalysis.model_validate(
        {
            "content_sufficiency": "sufficient",
            "brand": {
                "name": "Acme Web Studio",
                "category": "Web Design & Development Agency",
                "summary": "Pune-based studio building websites for small businesses.",
                "business_overview": "Acme designs and builds websites for SMBs.",
                "main_products_services": ["Website design", "E-commerce", "SEO"],
                "business_model": "Project-based agency services plus maintenance retainers",
                "geographic_focus": "Pune / Maharashtra, India",
            },
            "competitor_analysis": {
                "brand_overview": "Affordable SMB-focused web studio.",
                "market_positioning": "Affordable, fast delivery.",
                "competitors": [
                    {**competitor, "name": "Pixel Agency", "website": "https://pixel.example"},
                    {**competitor, "name": "Dead Domain Co", "website": "https://dead.example"},
                    {**competitor, "name": "No Site Ltd"},
                    {**competitor, "name": "Acme Web Studio", "website": "https://acme.example"},  # self -> dropped
                    {**competitor, "name": "Pixel  Agency"},  # duplicate -> dropped
                ],
                "comparison_table": [
                    {"dimension": "Brand Name", "your_brand": "Acme Web Studio",
                     "competitors": [{"competitor": "Pixel Agency", "value": "Pixel Agency"},
                                     {"competitor": "Acme Web Studio", "value": "self"}]},
                ],
                "competitive_advantages": ["4-week launches"],
                "opportunities": ["Publish case studies"],
            },
            "target_search_queries": [
                {"query": f"query {i}", "platforms": ["Google"], "primary_platform": "Google",
                 "intent": "Service Search", "intent_type": "commercial",
                 "business_relevance": "r", "target_audience": "SMB owners"}
                for i in range(6)  # 6 -> trimmed to 5
            ],
            "seo_improvements": {
                "primary_keyword": "small business website design",
                "secondary_keywords": ["web design pune"],
                "long_tail_keywords": ["affordable website design for small business pune"],
                "business_keywords": ["web studio"],
                "recommended_title": "Small Business Website Design in Pune | Acme Web Studio",
                "title_rationale": "Current title lacks keywords.",
                "meta_description": "Fast, mobile-friendly websites for small businesses in Pune.",
                "meta_description_keywords": ["small business website"],
                "supporting_content": [
                    {"content_type": "location_page", "title": "Web Design in Pune",
                     "target_keyword": "web design pune", "purpose": "Local SEO"}
                ],
                "technical_recommendations": [": [medium] Add alt text to 1 image.", "Add a canonical tag."],
            },
            "top_services": [
                {**service, "name": "Custom Website Design", "source": "website_content",
                 "evidence": "Custom website design and development built on modern frameworks."},
                {**service, "name": "E-commerce Stores", "source": "website_content",
                 "evidence": "E-commerce stores with Shopify and WooCommerce integration"},
                {**service, "name": "SEO", "source": "website_content",
                 "evidence": "We guarantee page one Google rankings in 7 days"},  # hallucinated quote
                {**service, "name": "Maintenance", "source": "website_content",
                 "evidence": "Website maintenance plans with 24/7 monitoring"},
                {**service, "name": "Social Media Management", "source": "inferred", "evidence": ""},
            ],
            "target_audience": {
                "primary_audience": ["Small business owners"],
                "secondary_audience": ["Startups"],
                "industry_segments": ["Retail"],
                "business_types": ["Local shops"],
                "customer_profiles": [
                    {"segment": "Audience 1", "business_type": "Small Businesses",
                     "requirement": "Website Development", "description": "d"}
                ],
                "geographic_focus": "Pune, India",
                "customer_needs_and_challenges": ["No online presence"],
                "customer_interests": ["Growth"],
                "recommended_marketing_segments": [
                    {"segment": "Local retailers", "rationale": "r",
                     "recommended_channels": ["Instagram"], "messaging_angle": "m"}
                ],
            },
            "analysis_notes": ["Competitors are AI suggestions."],
        }
    )


class FakeLLM:
    def __init__(self, analysis: LLMAnalysis | None = None, error: Exception | None = None) -> None:
        self.analysis = analysis or sample_llm_analysis()
        self.error = error
        self.calls = 0
        self.last_user_prompt = ""

    async def analyze(self, system_prompt: str, user_prompt: str) -> LLMResult:
        self.calls += 1
        self.last_user_prompt = user_prompt
        await asyncio.sleep(0.01)
        if self.error:
            raise self.error
        return LLMResult(self.analysis, "fake", "claude-haiku-5-5", 8000, 4000, 0, 0.0028, 10, False)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, llm_api_key=None, anthropic_api_key=None, rate_limit_per_minute=0,
                    content_char_budget=8000)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """no real dns in tests"""

    async def fake_resolve(hostname: str, port: int) -> str:
        return PUBLIC_IP

    monkeypatch.setattr(validators, "resolve_public_ip", fake_resolve)


@pytest.fixture
def fetcher() -> FakeFetcher:
    return FakeFetcher(reachable={"https://pixel.example"})


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(settings: Settings, fetcher: FakeFetcher, fake_llm: FakeLLM) -> AnalysisService:
    return AnalysisService(
        settings=settings,
        extractor=WebsiteExtractor(settings, fetcher),  # type: ignore[arg-type]
        llm=fake_llm,
        competitor_service=CompetitorService(fetcher, verify=True, timeout=1),  # type: ignore[arg-type]
        cache=TTLCache(ttl_seconds=60, max_entries=10),
    )


@pytest.fixture
def client(service: AnalysisService) -> Iterator[TestClient]:
    app.state.analysis_service = service
    try:
        with TestClient(app) as c:
            yield c
    finally:
        del app.state.analysis_service


__all__ = ["FakeFetcher", "FakeLLM", "sample_llm_analysis", "llm_module"]
