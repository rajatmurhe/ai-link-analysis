"""API tests. real extractor, fake network + fake llm"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.errors import LLMError
from app.main import app
from app.utils.cache import RateLimiter
from tests.conftest import FakeFetcher, FakeLLM

URL = "https://acme.example/"


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True


def test_root_redirects_to_docs(client: TestClient) -> None:
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_openapi_documents_endpoint(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert "/link-analysis" in spec["paths"]


def test_successful_analysis_structure(client: TestClient, fetcher: FakeFetcher) -> None:
    resp = client.post("/link-analysis", json={"url": "acme.example"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["success"] is True
    assert body["website"] == URL
    assert body["brand"]["name"] == "Acme Web Studio"
    assert {"platform": "Instagram", "url": "https://www.instagram.com/acmeweb"} in body["brand"]["social_profiles"]
    assert set(body["analysis"]) == {
        "competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience",
    }
    assert resp.headers["X-Request-ID"] == body["meta"]["request_id"]
    # Subpages discovered from the nav were fetched; login/PDF links were not.
    assert "https://acme.example/services" in fetcher.calls
    assert not any("login" in c or c.endswith(".pdf") for c in fetcher.calls)


def test_exactly_five_queries_without_volumes(client: TestClient) -> None:
    queries = client.post("/link-analysis", json={"url": URL}).json()["analysis"]["target_search_queries"]
    assert len(queries) == 5
    assert all(q["search_volume"] is None for q in queries)


def test_missing_platform_coverage_is_flagged(client: TestClient) -> None:
    # The fixture's queries all target Google only.
    warnings = client.post("/link-analysis", json={"url": URL}).json()["meta"]["warnings"]
    assert "No search query targets: Facebook, Instagram, LinkedIn, X." in warnings


def test_service_evidence_is_verified_against_page_text(client: TestClient) -> None:
    body = client.post("/link-analysis", json={"url": URL}).json()
    services = {s["name"]: s for s in body["analysis"]["top_services"]}
    assert len(services) == 5
    assert services["Custom Website Design"]["evidence_verified"] is True
    assert services["E-commerce Stores"]["evidence_verified"] is True  # minor punctuation differences tolerated
    assert services["SEO"]["evidence_verified"] is False  # quote does not exist on the site
    assert services["Social Media Management"]["source"] == "inferred"
    assert any("SEO" in w for w in body["meta"]["warnings"])


def test_competitors_cleaned_verified_and_labelled(client: TestClient) -> None:
    comp = client.post("/link-analysis", json={"url": URL}).json()["analysis"]["competitor_analysis"]
    by_name = {c["name"]: c for c in comp["competitors"]}
    assert "Acme Web Studio" not in by_name  # the brand itself is removed
    assert len(by_name) == 3  # duplicate "Pixel  Agency" removed
    assert by_name["Pixel Agency"]["website_status"] == "reachable"
    assert by_name["Dead Domain Co"]["website_status"] == "unreachable"
    assert by_name["No Site Ltd"]["website_status"] == "not_provided"
    assert all(c["source"] == "ai_suggestion" for c in comp["competitors"])
    assert "AI-generated" in comp["disclaimer"]
    # The brand's own column is removed from the comparison table along with the self-competitor.
    assert comp["comparison_table"][0]["competitors"] == [{"competitor": "Pixel Agency", "value": "Pixel Agency"}]


def test_seo_section_contains_measured_audit(client: TestClient) -> None:
    seo = client.post("/link-analysis", json={"url": URL}).json()["analysis"]["seo_improvements"]
    assert seo["current_title"] == "Acme Web Studio"
    assert seo["technical_recommendations"] == ["Add alt text to 1 image.", "Add a canonical tag."]
    assert seo["recommended_title_length"] == len(seo["recommended_title"])
    audit = seo["audit"]
    assert audit["images_missing_alt"] == 1
    assert audit["has_structured_data"] is True
    assert any("Title is short" in i["message"] for i in audit["issues"])


def test_prompt_contains_real_site_content_and_audit(client: TestClient, fake_llm: FakeLLM) -> None:
    client.post("/link-analysis", json={"url": URL})
    prompt = fake_llm.last_user_prompt
    assert "Websites that grow small businesses" in prompt
    assert "Conversion rate optimisation audits" in prompt  # from the /services subpage
    assert "<seo_audit_facts>" in prompt
    assert "We use cookies" not in prompt  # cookie banner stripped
    assert prompt.count("Website maintenance plans with 24/7 monitoring") == 1  # cross-page de-duplication


def test_empty_comparison_rows_are_flagged(client: TestClient, fake_llm: FakeLLM) -> None:
    for row in fake_llm.analysis.competitor_analysis.comparison_table:
        row.competitors = []
    warnings = client.post("/link-analysis", json={"url": URL}).json()["meta"]["warnings"]
    assert any("Comparison table" in w for w in warnings)


def test_meta_reports_cost_tokens_and_timings(client: TestClient) -> None:
    meta = client.post("/link-analysis", json={"url": URL}).json()["meta"]
    assert meta["cached"] is False
    assert meta["llm"]["input_tokens"] == 8000
    assert meta["llm"]["estimated_cost_usd"] > 0
    assert set(meta["timings"]) == {"extraction_ms", "llm_ms", "verification_ms", "total_ms"}
    assert len(meta["pages_analyzed"]) >= 2


def test_second_request_is_served_from_cache(client: TestClient, fake_llm: FakeLLM) -> None:
    first = client.post("/link-analysis", json={"url": URL}).json()
    second = client.post("/link-analysis", json={"url": "https://ACME.example"}).json()
    assert fake_llm.calls == 1
    assert second["meta"]["cached"] is True
    assert second["meta"]["request_id"] != first["meta"]["request_id"]

    client.post("/link-analysis", json={"url": URL, "force_refresh": True})
    assert fake_llm.calls == 2


def test_invalid_url_returns_structured_error(client: TestClient) -> None:
    resp = client.post("/link-analysis", json={"url": "ftp://acme.example"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_URL"
    assert body["request_id"]


def test_private_address_blocked(client: TestClient) -> None:
    resp = client.post("/link-analysis", json={"url": "http://169.254.169.254/latest/meta-data"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "BLOCKED_URL"


def test_missing_body_field(client: TestClient) -> None:
    resp = client.post("/link-analysis", json={})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unreachable_website(client: TestClient) -> None:
    resp = client.post("/link-analysis", json={"url": "https://missing.example"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "WEBSITE_HTTP_ERROR"


def test_insufficient_content(client: TestClient, fetcher: FakeFetcher) -> None:
    fetcher.pages["https://empty.example/"] = "<html><body><div id='root'></div><script>app()</script></body></html>"
    resp = client.post("/link-analysis", json={"url": "https://empty.example"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INSUFFICIENT_CONTENT"


def test_llm_failure_is_reported_and_not_cached(client: TestClient, fake_llm: FakeLLM) -> None:
    fake_llm.error = LLMError("The AI analysis service is temporarily unavailable.")
    resp = client.post("/link-analysis", json={"url": URL})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "LLM_ERROR"

    fake_llm.error = None
    assert client.post("/link-analysis", json={"url": URL}).status_code == 200


def test_llm_not_configured() -> None:
    app.state.analysis_service = None
    try:
        with TestClient(app) as c:
            resp = c.post("/link-analysis", json={"url": URL})
            assert resp.status_code == 503
            assert resp.json()["error"]["code"] == "LLM_NOT_CONFIGURED"
            assert c.get("/health").json()["llm_configured"] is False
    finally:
        del app.state.analysis_service


def test_rate_limit(client: TestClient) -> None:
    app.state.rate_limiter = RateLimiter(per_minute=2)
    codes = [client.post("/link-analysis", json={"url": URL}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert client.post("/link-analysis", json={"url": URL}).headers["Retry-After"]
