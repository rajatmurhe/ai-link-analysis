"""uvicorn app.main:app --host 0.0.0.0 --port 4000  (docs at /docs)"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import Settings, get_settings
from app.errors import AppError, LLMNotConfiguredError, RateLimitedError
from app.schemas import ErrorResponse, HealthResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.competitor_service import CompetitorService
from app.services.http_fetcher import SafeFetcher
from app.services.llm_service import create_llm
from app.services.website_extractor import WebsiteExtractor
from app.utils.cache import RateLimiter, TTLCache

logger = logging.getLogger("app")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_analysis_service(settings: Settings, fetcher: SafeFetcher) -> AnalysisService | None:
    try:
        llm = create_llm(settings)
    except LLMNotConfiguredError as exc:
        logger.warning("%s /link-analysis will return 503 until it is configured.", exc.message)
        return None
    return AnalysisService(
        settings=settings,
        extractor=WebsiteExtractor(settings, fetcher),
        llm=llm,
        competitor_service=CompetitorService(
            fetcher,
            verify=settings.verify_competitor_websites,
            timeout=settings.competitor_check_timeout_seconds,
        ),
        cache=TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    fetcher = SafeFetcher(settings)
    app.state.settings = settings
    app.state.fetcher = fetcher
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    if not hasattr(app.state, "analysis_service"):  # tests set their own
        app.state.analysis_service = build_analysis_service(settings, fetcher)
    logger.info("Started %s v%s (provider=%s, model=%s, fallback=%s)", settings.app_name, settings.app_version,
                settings.llm_provider, settings.resolved_model, settings.resolved_fallback_model)
    yield
    await fetcher.aclose()
    llm = getattr(app.state.analysis_service, "llm", None)
    if llm is not None and hasattr(llm, "aclose"):
        await llm.aclose()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Analyses a public website with an LLM and returns five structured sections: "
        "**competitor vs brand**, **top 5 search queries** (Google, Facebook, Instagram, LinkedIn, X), "
        "**SEO improvements** (with a measured HTML audit), **top 5 services** (with on-site evidence) "
        "and **target audience**.\n\n"
        "Includes SSRF protection, DNS pinning, result caching, and per-request cost/latency reporting."
    ),
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid URL or unusable website content"},
        500: {"model": ErrorResponse},
    },
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = str(elapsed)
    if request.url.path not in ("/health", "/favicon.ico"):
        logger.info("%s %s -> %d (%dms) [%s]", request.method, request.url.path, response.status_code, elapsed, request_id)
    return response


def _error(request: Request, status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    body = ErrorResponse(
        error={"code": code, "message": message, "details": details or {}},
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    log = logger.warning if exc.status_code < 500 else logger.error
    log("%s: %s %s", exc.code, exc.message, exc.details)
    headers_resp = _error(request, exc.status_code, exc.code, exc.message, exc.details)
    if isinstance(exc, RateLimitedError):
        headers_resp.headers["Retry-After"] = str(exc.details.get("retry_after_seconds", 60))
    return headers_resp


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
    return _error(request, 422, "VALIDATION_ERROR", "Request body is invalid.", {"errors": errors})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %r", exc)
    return _error(request, 500, "INTERNAL_ERROR", "An unexpected error occurred.")


def rate_limit(request: Request) -> None:
    # not reading X-Forwarded-For directly, anyone could fake it. uvicorn --proxy-headers handles it
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = request.app.state.rate_limiter.allow(client_ip)
    if not allowed:
        raise RateLimitedError(
            "Too many analysis requests. Please wait before retrying.",
            details={"retry_after_seconds": retry_after},
        )


def get_analysis_service(request: Request) -> AnalysisService:
    service = request.app.state.analysis_service
    if service is None:
        raise LLMNotConfiguredError("The LLM provider is not configured on this server (missing or invalid API key).")
    return service


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    s: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=s.app_version,
        llm_configured=request.app.state.analysis_service is not None,
        provider=s.llm_provider,
        model=s.resolved_model,
        fallback_model=s.resolved_fallback_model,
    )


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    tags=["analysis"],
    summary="Analyse a website URL",
    responses={
        400: {"model": ErrorResponse, "description": "URL blocked (private/internal address, SSRF protection)"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        502: {"model": ErrorResponse, "description": "Website unreachable / HTTP error, or LLM failure"},
        503: {"model": ErrorResponse, "description": "LLM provider not configured"},
        504: {"model": ErrorResponse, "description": "Website timed out"},
    },
    dependencies=[Depends(rate_limit)],
)
async def link_analysis(
    payload: LinkAnalysisRequest,
    request: Request,
    service: AnalysisService = Depends(get_analysis_service),
) -> LinkAnalysisResponse:
    """Fetch the website, understand it with an LLM and return the five analysis sections."""
    return await service.analyze(payload.url, request_id=request.state.request_id, force_refresh=payload.force_refresh)
