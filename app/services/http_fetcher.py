"""Fetcher that won't hit internal addresses.

Every url and redirect is validated, and we connect to the ip we already checked (Host header
+ SNI so https still works). Body is streamed with a size limit.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from app.config import Settings
from app.errors import (
    UnsupportedContentError,
    WebsiteHTTPError,
    WebsiteTimeoutError,
    WebsiteUnreachableError,
)
from app.utils.validators import ValidatedURL, validate_public_url

_HTML_TYPES = ("text/html", "application/xhtml+xml")
_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-]+)""", re.IGNORECASE)


@dataclass
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    text: str
    elapsed_ms: int
    redirects: list[str] = field(default_factory=list)
    truncated: bool = False


def _pinned_url(target: ValidatedURL) -> str:
    """swap the hostname for the checked ip"""
    parts = urlsplit(target.url)
    host = f"[{target.ip}]" if ":" in target.ip else target.ip
    default_port = 443 if target.scheme == "https" else 80
    netloc = host if target.port == default_port else f"{host}:{target.port}"
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))


def _host_header(target: ValidatedURL) -> str:
    default_port = 443 if target.scheme == "https" else 80
    return target.hostname if target.port == default_port else f"{target.hostname}:{target.port}"


def _decode(body: bytes, response: httpx.Response) -> str:
    encoding = response.charset_encoding
    if not encoding:
        match = _META_CHARSET.search(body[:4096])
        encoding = match.group(1).decode("ascii", "ignore") if match else "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class SafeFetcher:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(settings.fetch_timeout_seconds, connect=6.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            http2=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch(
        self,
        url: str,
        *,
        require_html: bool = True,
        max_bytes: int | None = None,
        timeout: float | None = None,
    ) -> FetchResult:
        started = time.perf_counter()
        max_bytes = max_bytes or self.settings.fetch_max_bytes
        redirects: list[str] = []
        current = url

        for _ in range(self.settings.fetch_max_redirects + 1):
            target = await validate_public_url(current, self.settings.allowed_ports)
            headers = {
                "Host": _host_header(target),
                "User-Agent": self.settings.fetch_user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.9",
            }
            extensions = {"sni_hostname": target.hostname} if target.scheme == "https" else {}
            request = self._client.build_request(
                "GET",
                _pinned_url(target),
                headers=headers,
                extensions=extensions,
                timeout=timeout or self.settings.fetch_timeout_seconds,
            )
            try:
                response = await self._client.send(request, stream=True)
            except httpx.TimeoutException as exc:
                raise WebsiteTimeoutError(
                    f"The website took too long to respond ({target.hostname}).",
                    details={"url": target.url},
                ) from exc
            except httpx.HTTPError as exc:
                raise WebsiteUnreachableError(
                    f"Could not connect to {target.hostname}: {type(exc).__name__}.",
                    details={"url": target.url},
                ) from exc

            try:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise WebsiteHTTPError("Redirect response without a Location header.")
                    current = urljoin(target.url, location)
                    redirects.append(current)
                    continue

                if response.status_code >= 400:
                    hint = ""
                    if response.status_code in (401, 403, 429, 503):
                        hint = " The site may block automated access (bot protection / WAF)."
                    raise WebsiteHTTPError(
                        f"Website returned HTTP {response.status_code}.{hint}",
                        details={"url": target.url, "status_code": response.status_code},
                    )

                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if require_html and content_type and not content_type.startswith(_HTML_TYPES):
                    raise UnsupportedContentError(
                        f"URL returned '{content_type}', not an HTML web page.",
                        details={"url": target.url, "content_type": content_type},
                    )

                chunks: list[bytes] = []
                size = 0
                truncated = False
                try:
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size >= max_bytes:
                            truncated = True
                            break
                except httpx.TimeoutException as exc:
                    raise WebsiteTimeoutError("Timed out while downloading the page.") from exc
                except httpx.HTTPError as exc:
                    raise WebsiteUnreachableError(f"Connection error while downloading: {exc}") from exc

                body = b"".join(chunks)[:max_bytes]
                return FetchResult(
                    requested_url=url,
                    final_url=target.url,
                    status_code=response.status_code,
                    content_type=content_type or "unknown",
                    text=_decode(body, response),
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                    redirects=redirects,
                    truncated=truncated,
                )
            finally:
                await response.aclose()

        raise WebsiteHTTPError(
            f"Too many redirects (more than {self.settings.fetch_max_redirects}).",
            details={"redirect_chain": redirects},
        )

    async def probe(self, url: str, timeout: float) -> tuple[bool, str | None]:
        """quick check that a competitor site actually exists"""
        try:
            result = await self.fetch(url, require_html=False, max_bytes=65_536, timeout=timeout)
            return True, result.final_url
        except Exception:  # noqa: BLE001
            return False, None
