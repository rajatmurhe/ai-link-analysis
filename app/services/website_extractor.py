"""Scrapes the home page + a few useful pages (about, services, pricing...).

Removes menus/cookie banners/repeated footer text, builds the text we send to the llm and
also does a basic seo audit of the home page (no llm there).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from app.config import Settings
from app.errors import AppError, InsufficientContentError
from app.schemas import SEOAudit, SEOIssue, SocialProfile
from app.services.http_fetcher import FetchResult, SafeFetcher

logger = logging.getLogger(__name__)

_NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "template", "canvas", "object", "embed", "picture"]
_NOISE_PATTERN = re.compile(r"cookie|consent|gdpr|newsletter-popup|modal|popup|banner-ad|skip-link", re.I)
_WS = re.compile(r"\s+")

# higher = more useful page
_SUBPAGE_KEYWORDS: dict[str, int] = {
    "services": 14, "service": 13, "solutions": 13, "solution": 12, "products": 13, "product": 12,
    "what-we-do": 13, "offerings": 12, "capabilities": 11, "platform": 10, "features": 9,
    "about": 12, "about-us": 12, "company": 9, "who-we-are": 11, "our-story": 8,
    "pricing": 9, "plans": 8, "industries": 9, "customers": 7, "clients": 7, "case-studies": 7,
    "portfolio": 7, "work": 4, "contact": 5, "locations": 5, "shop": 6, "menu": 5, "treatments": 8,
}
_SKIP_PATH = re.compile(
    r"login|signin|sign-in|signup|register|cart|checkout|account|privacy|terms|cookie|legal|"
    r"careers|jobs|wp-admin|wp-login|feed|rss|/tag/|/category/|/author/|/page/\d|\?|#",
    re.I,
)
_SKIP_EXT = re.compile(r"\.(pdf|jpe?g|png|gif|webp|svg|zip|mp4|mp3|docx?|xlsx?|pptx?|xml|json|css|js)$", re.I)

_SOCIAL_DOMAINS = {
    "facebook.com": "Facebook", "instagram.com": "Instagram", "linkedin.com": "LinkedIn",
    "twitter.com": "X", "x.com": "X", "youtube.com": "YouTube", "tiktok.com": "TikTok",
    "pinterest.com": "Pinterest", "github.com": "GitHub",
}


@dataclass
class PageContent:
    url: str
    title: str | None
    meta_description: str | None
    h1: list[str]
    h2: list[str]
    text: str
    nav_items: list[str] = field(default_factory=list)


@dataclass
class ExtractedWebsite:
    input_url: str
    final_url: str
    domain: str
    pages: list[PageContent]
    seo_audit: SEOAudit
    social_profiles: list[SocialProfile]
    structured_data: dict
    language: str | None
    fetch_ms: int
    warnings: list[str] = field(default_factory=list)

    @property
    def homepage(self) -> PageContent:
        return self.pages[0]

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def corpus(self) -> str:
        """all the text, used to check the llm's evidence quotes"""
        parts: list[str] = []
        for p in self.pages:
            parts.extend([p.title or "", p.meta_description or "", *p.h1, *p.h2, *p.nav_items, p.text])
        return "\n".join(parts)

    def to_llm_context(self) -> str:
        home = self.homepage
        lines = [
            f"WEBSITE: {self.final_url}",
            f"DOMAIN: {self.domain}",
            f"LANGUAGE: {self.language or 'unknown'}",
            f"HOMEPAGE TITLE: {home.title or '(missing)'}",
            f"HOMEPAGE META DESCRIPTION: {home.meta_description or '(missing)'}",
        ]
        if self.structured_data:
            lines.append(f"STRUCTURED DATA (JSON-LD): {json.dumps(self.structured_data, ensure_ascii=False)[:1500]}")
        if self.social_profiles:
            lines.append("SOCIAL PROFILES: " + ", ".join(f"{s.platform}: {s.url}" for s in self.social_profiles))
        if home.nav_items:
            lines.append("NAVIGATION MENU: " + " | ".join(home.nav_items[:40]))
        for idx, page in enumerate(self.pages):
            label = "HOMEPAGE" if idx == 0 else f"SUBPAGE {idx}"
            lines.append(f"\n===== {label}: {page.url} =====")
            if idx > 0 and page.title:
                lines.append(f"TITLE: {page.title}")
            if page.h1:
                lines.append("H1: " + " | ".join(page.h1[:5]))
            if page.h2:
                lines.append("H2: " + " | ".join(page.h2[:15]))
            lines.append(page.text)
        return "\n".join(lines)


def _clean(text: str | None) -> str:
    return _WS.sub(" ", text or "").strip()


def _registrable(host: str) -> str:
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _meta(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str | None:
    tag = soup.find("meta", attrs={"name": name}) if name else soup.find("meta", attrs={"property": prop})
    if isinstance(tag, Tag) and tag.get("content"):
        return _clean(str(tag["content"])) or None
    return None


def _headings(soup: BeautifulSoup, level: str, limit: int = 25) -> list[str]:
    out: list[str] = []
    for h in soup.find_all(level):
        t = _clean(h.get_text(" "))
        if t and t not in out:
            out.append(t[:200])
        if len(out) >= limit:
            break
    return out


def _parse_json_ld(soup: BeautifulSoup) -> dict:
    """pull org info out of json-ld if the site has it"""
    types: list[str] = []
    org: dict = {}
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
                continue
            if not isinstance(node, dict):
                continue
            if "@graph" in node:
                stack.extend(node["@graph"] if isinstance(node["@graph"], list) else [node["@graph"]])
            node_type = node.get("@type")
            for t in node_type if isinstance(node_type, list) else [node_type]:
                if isinstance(t, str) and t not in types:
                    types.append(t)
            if any(isinstance(t, str) and ("Organization" in t or "Business" in t or t in {"Corporation", "Store"})
                   for t in (node_type if isinstance(node_type, list) else [node_type])):
                for key in ("name", "legalName", "description", "slogan", "foundingDate", "areaServed", "sameAs", "address"):
                    if key in node and key not in org:
                        org[key] = node[key]
    result: dict = {}
    if types:
        result["types"] = types[:20]
    if org:
        result["organization"] = org
    return result


def _social_profiles(soup: BeautifulSoup) -> list[SocialProfile]:
    found: dict[str, SocialProfile] = {}
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        try:
            host = _registrable(urlsplit(href).hostname or "")
        except ValueError:
            continue
        for domain, platform in _SOCIAL_DOMAINS.items():
            if (host == domain or host.endswith("." + domain)) and platform not in found:
                path = urlsplit(href).path.strip("/")
                if path and not path.startswith(("share", "sharer", "intent")):
                    found[platform] = SocialProfile(platform=platform, url=href[:300])
    return list(found.values())


def _nav_items(soup: BeautifulSoup) -> list[str]:
    items: list[str] = []
    for nav in soup.find_all(["nav", "header"]):
        for a in nav.find_all("a"):
            t = _clean(a.get_text(" "))
            if t and 1 < len(t) <= 40 and t not in items:
                items.append(t)
    return items[:60]


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        if not isinstance(tag, Tag) or tag.attrs is None:
            continue
        ident = " ".join([str(tag.get("id", "")), " ".join(tag.get("class", []) or [])])
        if ident.strip() and _NOISE_PATTERN.search(ident) and tag.name not in ("body", "html", "main"):
            tag.decompose()
        elif tag.get("aria-hidden") == "true" or tag.get("role") == "dialog":
            tag.decompose()


def _text_lines(soup: BeautifulSoup) -> list[str]:
    root: Tag = soup.body or soup  # type: ignore[assignment]
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    # use <main> if it has most of the text, footer too (address etc lives there)
    if isinstance(main, Tag) and len(main.get_text(" ", strip=True)) > 400:
        sources: list[Tag] = [main]
        footer = soup.find("footer")
        if isinstance(footer, Tag):
            sources.append(footer)
    else:
        for nav in root.find_all("nav"):
            nav.decompose()
        sources = [root]
    lines: list[str] = []
    for src in sources:
        for raw in src.get_text("\n").split("\n"):
            line = _clean(raw)
            if len(line) >= 3 and any(c.isalpha() for c in line):
                lines.append(line)
    return lines


def build_seo_audit(soup: BeautifulSoup, final_url: str, html_size: int) -> SEOAudit:
    title = _clean(soup.title.get_text()) if soup.title else None
    meta_description = _meta(soup, name="description")
    h1 = _headings(soup, "h1", limit=10)
    h2_count = len(soup.find_all("h2"))
    images = soup.find_all("img")
    missing_alt = sum(1 for img in images if not _clean(str(img.get("alt", ""))))
    canonical_tag = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
    canonical = str(canonical_tag.get("href")) if isinstance(canonical_tag, Tag) and canonical_tag.get("href") else None
    robots = _meta(soup, name="robots")
    html_tag = soup.find("html")
    lang = str(html_tag.get("lang")) if isinstance(html_tag, Tag) and html_tag.get("lang") else None
    viewport = _meta(soup, name="viewport") is not None
    og = {k: _meta(soup, prop=f"og:{k}") for k in ("title", "description", "image")}
    has_jsonld = bool(soup.find("script", attrs={"type": "application/ld+json"}))
    word_count = len((soup.body or soup).get_text(" ", strip=True).split())

    issues: list[SEOIssue] = []

    def add(sev: str, msg: str) -> None:
        issues.append(SEOIssue(severity=sev, message=msg))  # type: ignore[arg-type]

    if not title:
        add("high", "Page has no <title> tag.")
    elif len(title) < 30:
        add("medium", f"Title is short ({len(title)} chars); aim for 50-60 characters with the primary keyword.")
    elif len(title) > 60:
        add("medium", f"Title is long ({len(title)} chars) and may be truncated in Google results (aim for 50-60).")
    if not meta_description:
        add("high", "Missing meta description; Google will auto-generate a snippet.")
    elif len(meta_description) < 70:
        add("medium", f"Meta description is short ({len(meta_description)} chars); aim for 140-160.")
    elif len(meta_description) > 160:
        add("low", f"Meta description is long ({len(meta_description)} chars) and may be truncated (aim for 140-160).")
    if not h1:
        add("high", "No <h1> heading found on the home page.")
    elif len(h1) > 1:
        add("low", f"{len(h1)} <h1> headings found; a single descriptive H1 is recommended.")
    if images and missing_alt:
        add("medium", f"{missing_alt} of {len(images)} images have no alt text (accessibility + image search).")
    if not canonical:
        add("low", "No canonical URL declared (<link rel=\"canonical\">).")
    if robots and "noindex" in robots.lower():
        add("high", "Page is marked 'noindex' and will not appear in search results.")
    if not lang:
        add("low", "<html> element has no lang attribute.")
    if not viewport:
        add("high", "No responsive viewport meta tag (mobile-friendliness).")
    if not all(og.values()):
        missing = [k for k, v in og.items() if not v]
        add("low", f"Missing Open Graph tags for social sharing: {', '.join('og:' + m for m in missing)}.")
    if not has_jsonld:
        add("medium", "No schema.org structured data (JSON-LD); add Organization / LocalBusiness markup.")
    if not final_url.startswith("https://"):
        add("high", "Site is not served over HTTPS.")
    if word_count < 300:
        add("medium", f"Thin home page content ({word_count} words); search engines favour substantive copy.")

    return SEOAudit(
        title=title,
        title_length=len(title) if title else 0,
        meta_description=meta_description,
        meta_description_length=len(meta_description) if meta_description else 0,
        h1=h1,
        h2_count=h2_count,
        word_count=word_count,
        image_count=len(images),
        images_missing_alt=missing_alt,
        canonical_url=canonical,
        robots_meta=robots,
        lang=lang,
        has_viewport_meta=viewport,
        has_open_graph=all(og.values()),
        has_structured_data=has_jsonld,
        uses_https=final_url.startswith("https://"),
        html_size_kb=round(html_size / 1024, 1),
        issues=issues,
    )


def _candidate_subpages(soup: BeautifulSoup, base_url: str, limit: int) -> list[str]:
    base = urlsplit(base_url)
    base_host = _registrable(base.hostname or "")
    scored: dict[str, tuple[int, str]] = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, str(a["href"]).strip())
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https") or _registrable(parts.hostname or "") != base_host:
            continue
        path = parts.path.rstrip("/").lower()
        if not path or path == base.path.rstrip("/").lower() or _SKIP_EXT.search(path) or _SKIP_PATH.search(href):
            continue
        segments = [s for s in path.split("/") if s]
        if len(segments) > 3:
            continue
        anchor = _clean(a.get_text(" ")).lower().replace(" ", "-")
        score, matched = 0, ""
        for kw, weight in _SUBPAGE_KEYWORDS.items():
            if weight > score and (segments[-1] == kw or anchor == kw or anchor.startswith(kw + "-")
                                   or (kw in segments and segments[-1] == kw)):
                score, matched = weight, kw
        if score:
            score -= len(segments) - 1  # shallow pages first
            clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            if score > scored.get(clean_url, (0, ""))[0]:
                scored[clean_url] = (score, matched)
    ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
    picked: list[str] = []
    seen_topics: set[str] = set()
    for url, (_, topic) in ranked:
        # max one page per topic, don't want /services and /service both
        topic_key = topic.rstrip("s")
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)
        picked.append(url)
        if len(picked) >= limit:
            break
    return picked


def _parse_page(html: str, url: str) -> tuple[BeautifulSoup, PageContent, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    title = _clean(soup.title.get_text()) if soup.title else None
    page = PageContent(
        url=url,
        title=title,
        meta_description=_meta(soup, name="description") or _meta(soup, prop="og:description"),
        h1=_headings(soup, "h1", 5),
        h2=_headings(soup, "h2", 20),
        text="",
        nav_items=_nav_items(soup),
    )
    work = BeautifulSoup(html, "lxml")
    _strip_noise(work)
    return soup, page, _text_lines(work)


def _fit_budget(lines: list[str], seen: set[str], budget: int) -> str:
    out: list[str] = []
    used = 0
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        if used + len(line) + 1 > budget:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


class WebsiteExtractor:
    def __init__(self, settings: Settings, fetcher: SafeFetcher) -> None:
        self.settings = settings
        self.fetcher = fetcher

    async def extract(self, url: str) -> ExtractedWebsite:
        home: FetchResult = await self.fetcher.fetch(url)
        soup, home_page, home_lines = _parse_page(home.text, home.final_url)

        warnings: list[str] = []
        if home.truncated:
            warnings.append("Home page HTML exceeded the size limit and was truncated.")

        subpage_urls = _candidate_subpages(soup, home.final_url, self.settings.fetch_max_subpages)
        results = await asyncio.gather(*(self.fetcher.fetch(u) for u in subpage_urls), return_exceptions=True)

        sub_parsed: list[tuple[PageContent, list[str]]] = []
        for sub_url, res in zip(subpage_urls, results):
            if isinstance(res, AppError):
                logger.info("Skipping subpage %s: %s", sub_url, res.message)
                continue
            if isinstance(res, BaseException):
                logger.warning("Unexpected error fetching subpage %s: %r", sub_url, res)
                continue
            if res.final_url.rstrip("/") == home.final_url.rstrip("/"):
                continue
            _, page, lines = _parse_page(res.text, res.final_url)
            sub_parsed.append((page, lines))

        # home page gets up to 45% of the budget, rest is split.
        # `seen` skips lines we already took from another page (menus, footers)
        total = self.settings.content_char_budget
        seen: set[str] = set()
        home_page.text = _fit_budget(home_lines, seen, int(total * 0.45) if sub_parsed else total)
        remaining = total - len(home_page.text)
        pages = [home_page]
        for i, (page, lines) in enumerate(sub_parsed):
            share = remaining // (len(sub_parsed) - i)
            page.text = _fit_budget(lines, seen, share)
            remaining -= len(page.text)
            if page.text:
                pages.append(page)

        audit = build_seo_audit(soup, home.final_url, len(home.text))
        extracted = ExtractedWebsite(
            input_url=url,
            final_url=home.final_url,
            domain=_registrable(urlsplit(home.final_url).hostname or ""),
            pages=pages,
            seo_audit=audit,
            social_profiles=_social_profiles(soup),
            structured_data=_parse_json_ld(soup),
            language=audit.lang,
            fetch_ms=home.elapsed_ms,
            warnings=warnings,
        )

        if extracted.total_chars < 150 and not home_page.meta_description:
            raise InsufficientContentError(
                "The page has almost no readable text. It may be rendered entirely with JavaScript, "
                "be a parked domain, or block automated access.",
                details={"url": home.final_url, "extracted_chars": extracted.total_chars},
            )
        if extracted.total_chars < 600:
            warnings.append(
                "Very little text could be extracted (the site may rely on client-side JavaScript); "
                "analysis confidence is reduced."
            )
        return extracted
