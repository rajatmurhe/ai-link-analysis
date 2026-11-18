"""App errors. each one has an error code + http status"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    code = "INTERNAL_ERROR"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidURLError(AppError):
    code = "INVALID_URL"
    status_code = 422


class BlockedURLError(AppError):
    """private / localhost / metadata ip etc"""

    code = "BLOCKED_URL"
    status_code = 400


class WebsiteUnreachableError(AppError):
    code = "WEBSITE_UNREACHABLE"
    status_code = 502


class WebsiteTimeoutError(AppError):
    code = "WEBSITE_TIMEOUT"
    status_code = 504


class WebsiteHTTPError(AppError):
    code = "WEBSITE_HTTP_ERROR"
    status_code = 502


class UnsupportedContentError(AppError):
    code = "UNSUPPORTED_CONTENT"
    status_code = 422


class InsufficientContentError(AppError):
    code = "INSUFFICIENT_CONTENT"
    status_code = 422


class LLMNotConfiguredError(AppError):
    code = "LLM_NOT_CONFIGURED"
    status_code = 503


class LLMError(AppError):
    code = "LLM_ERROR"
    status_code = 502


class RateLimitedError(AppError):
    code = "RATE_LIMITED"
    status_code = 429
