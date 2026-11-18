"""Request/response models.

LLM* classes = the json schema we force the model to follow (kept simple, no min/max since
structured output doesn't support them). The API classes extend them with stuff we work out
ourselves (evidence check, seo audit, meta).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]
Confidence = Literal["high", "medium", "low"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LinkAnalysisRequest(BaseModel):
    url: str = Field(
        ...,
        description="Public website URL to analyse. `https://` is assumed when the scheme is omitted.",
        examples=["https://www.anthropic.com"],
    )
    force_refresh: bool = Field(False, description="Bypass the result cache and re-run the analysis.")

    model_config = ConfigDict(json_schema_extra={"examples": [{"url": "https://stripe.com"}]})


# what the llm returns
class LLMBrand(_Strict):
    name: str = Field(description="Brand / company name as presented on the website.")
    category: str = Field(description="Main business category, e.g. 'B2B SaaS - Payments'.")
    summary: str = Field(description="One-sentence plain-language summary of what the business does.")
    business_overview: str = Field(description="3-5 sentence overview grounded in the website content.")
    main_products_services: list[str]
    business_model: str = Field(description="e.g. B2B SaaS subscription, D2C e-commerce, agency services.")
    geographic_focus: str | None = Field(description="Location / markets served if stated on the site, else null.")


class LLMCompetitor(_Strict):
    name: str
    website: str | None = Field(description="Competitor's official homepage URL if known with confidence, else null.")
    description: str
    main_services: list[str]
    target_market: str
    strengths: list[str]
    weaknesses_or_gaps: list[str]
    confidence: Confidence = Field(description="How confident you are this is a genuine direct competitor.")


class ComparisonValue(_Strict):
    competitor: str
    value: str


class ComparisonRow(_Strict):
    dimension: str = Field(description="e.g. Main Service, Target Market, Pricing Position, Key Strength, Opportunity.")
    your_brand: str
    competitors: list[ComparisonValue]


class LLMCompetitorAnalysis(_Strict):
    brand_overview: str
    market_positioning: str = Field(description="How the brand currently positions itself (price, niche, quality...).")
    competitors: list[LLMCompetitor]
    comparison_table: list[ComparisonRow]
    competitive_advantages: list[str] = Field(description="The submitted brand's advantages, evidenced by its site.")
    opportunities: list[str] = Field(description="Concrete actions to improve market positioning.")


class LLMSearchQuery(_Strict):
    query: str
    platforms: list[Platform] = Field(description="Every platform where this query is a natural fit.")
    primary_platform: Platform
    intent: str = Field(description="Short label, e.g. Service Search, Business Discovery, Inspiration, B2B Search.")
    intent_type: Literal["informational", "commercial", "transactional", "navigational", "inspirational"]
    business_relevance: str = Field(description="Why this query maps to the brand's offering.")
    target_audience: str


class SupportingContent(_Strict):
    content_type: Literal[
        "service_page", "industry_page", "location_page", "faq", "blog_post", "guide", "case_study",
        "comparison_page", "landing_page", "other",
    ]
    title: str
    target_keyword: str
    purpose: str


class LLMSEOImprovements(_Strict):
    primary_keyword: str
    secondary_keywords: list[str]
    long_tail_keywords: list[str]
    business_keywords: list[str]
    recommended_title: str = Field(description="50-60 characters, includes the primary keyword and the brand.")
    title_rationale: str = Field(description="What was wrong with the current title and how the new structure fixes it.")
    meta_description: str = Field(description="140-160 characters, human-friendly, includes primary keyword + CTA.")
    meta_description_keywords: list[str]
    supporting_content: list[SupportingContent]
    technical_recommendations: list[str] = Field(description="Prioritised fixes based on the provided SEO audit facts.")


class LLMService(_Strict):
    name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source: Literal["website_content", "inferred"] = Field(
        description="'website_content' only if the service is explicitly offered on the site; otherwise 'inferred'."
    )
    evidence: str = Field(
        description="A short VERBATIM quote (max ~15 words) copied exactly from the website text that supports "
        "this service. Empty string if inferred."
    )


class CustomerProfile(_Strict):
    segment: str
    business_type: str
    requirement: str = Field(description="What this segment needs from the brand.")
    description: str


class MarketingSegment(_Strict):
    segment: str
    rationale: str
    recommended_channels: list[str]
    messaging_angle: str


class LLMTargetAudience(_Strict):
    primary_audience: list[str]
    secondary_audience: list[str]
    industry_segments: list[str]
    business_types: list[str]
    customer_profiles: list[CustomerProfile]
    geographic_focus: str | None
    customer_needs_and_challenges: list[str]
    customer_interests: list[str]
    recommended_marketing_segments: list[MarketingSegment]


class LLMAnalysis(_Strict):

    content_sufficiency: Literal["sufficient", "limited", "insufficient"] = Field(
        description="Whether the extracted website content was enough for a confident analysis."
    )
    brand: LLMBrand
    competitor_analysis: LLMCompetitorAnalysis
    target_search_queries: list[LLMSearchQuery] = Field(
        description="Exactly 5 queries, one per primary_platform in this order: Google, Facebook, Instagram, LinkedIn, X."
    )
    seo_improvements: LLMSEOImprovements
    top_services: list[LLMService] = Field(description="Exactly 5 services, most important first.")
    target_audience: LLMTargetAudience
    analysis_notes: list[str] = Field(description="Caveats or assumptions the reader should know about.")


class SEOIssue(BaseModel):
    severity: Literal["high", "medium", "low"]
    message: str


class SEOAudit(BaseModel):
    """straight from the home page html, no llm"""

    title: str | None
    title_length: int
    meta_description: str | None
    meta_description_length: int
    h1: list[str]
    h2_count: int
    word_count: int
    image_count: int
    images_missing_alt: int
    canonical_url: str | None
    robots_meta: str | None
    lang: str | None
    has_viewport_meta: bool
    has_open_graph: bool
    has_structured_data: bool
    uses_https: bool
    html_size_kb: float
    issues: list[SEOIssue]


class SocialProfile(BaseModel):
    platform: str
    url: str


# api response
class Brand(LLMBrand):
    model_config = ConfigDict(extra="ignore")
    website: str
    social_profiles: list[SocialProfile] = []


class Competitor(LLMCompetitor):
    model_config = ConfigDict(extra="ignore")
    source: Literal["ai_suggestion"] = "ai_suggestion"
    website_status: Literal["reachable", "unreachable", "not_provided", "not_checked"] = "not_checked"


class CompetitorAnalysis(LLMCompetitorAnalysis):
    model_config = ConfigDict(extra="ignore")
    competitors: list[Competitor]  # type: ignore[assignment]
    disclaimer: str


class SearchQuery(LLMSearchQuery):
    model_config = ConfigDict(extra="ignore")
    search_volume: None = Field(
        None, description="Always null: no search-volume data source is connected, so none is reported."
    )


class SEOImprovements(LLMSEOImprovements):
    model_config = ConfigDict(extra="ignore")
    current_title: str | None
    current_meta_description: str | None
    recommended_title_length: int
    meta_description_length: int
    audit: SEOAudit


class Service(LLMService):
    model_config = ConfigDict(extra="ignore")
    evidence_verified: bool = Field(
        description="True when the evidence quote was found in the extracted website text."
    )


class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery]
    seo_improvements: SEOImprovements
    top_services: list[Service]
    target_audience: LLMTargetAudience


class PageInfo(BaseModel):
    url: str
    title: str | None
    chars_used: int


class LLMUsage(BaseModel):
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    estimated_cost_usd: float
    latency_ms: int
    fallback_used: bool = False


class Timings(BaseModel):
    extraction_ms: int
    llm_ms: int
    verification_ms: int
    total_ms: int


class Meta(BaseModel):
    request_id: str
    analyzed_at: datetime
    cached: bool
    content_sufficiency: str
    pages_analyzed: list[PageInfo]
    content_chars_sent: int
    llm: LLMUsage
    timings: Timings
    warnings: list[str]
    notes: list[str]


class LinkAnalysisResponse(BaseModel):
    success: Literal[True] = True
    website: str
    brand: Brand
    analysis: Analysis
    meta: Meta


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = {}


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorBody
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    llm_configured: bool
    provider: str
    model: str
    fallback_model: str | None
