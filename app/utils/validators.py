"""URL checks + SSRF protection.

- only http/https, no user:pass@, only allowed ports
- block localhost / .internal / metadata hostnames
- every DNS answer has to be a public IP
- we return the IP so the fetcher connects to that exact address (stops dns rebinding)
- redirects get checked again in the fetcher
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from app.errors import BlockedURLError, InvalidURLError, WebsiteUnreachableError

_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata",
    "metadata.google.internal",
    "instance-data",
}
_BLOCKED_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".intranet",
    ".lan",
    ".home",
    ".corp",
    ".private",
    ".home.arpa",
)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class ValidatedURL:
    url: str  # cleaned url, no #fragment
    scheme: str
    hostname: str
    port: int
    ip: str  # connect to this ip, not a fresh lookup


def normalize_url(raw: str) -> str:
    """add https:// if missing, drop #fragment, lowercase host"""
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidURLError("URL must be a non-empty string.")
    candidate = raw.strip()
    if any(ch.isspace() for ch in candidate):
        raise InvalidURLError("URL must not contain whitespace.")
    if len(candidate) > 2048:
        raise InvalidURLError("URL is too long (max 2048 characters).")
    if "://" not in candidate:
        candidate = "https://" + candidate.lstrip("/")

    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise InvalidURLError(f"Unsupported URL scheme '{parts.scheme}'. Only http and https are allowed.")
    if parts.username or parts.password:
        raise BlockedURLError("URLs with embedded credentials are not allowed.")
    if not parts.hostname:
        raise InvalidURLError("URL must include a hostname.")
    try:
        port = parts.port
    except ValueError as exc:
        raise InvalidURLError("URL contains an invalid port.") from exc

    host = parts.hostname.lower().rstrip(".")
    try:
        host_ascii = host.encode("idna").decode("ascii") if not _is_ip_literal(host) else host
    except UnicodeError as exc:
        raise InvalidURLError("URL hostname is not a valid domain name.") from exc

    netloc = f"[{host_ascii}]" if ":" in host_ascii else host_ascii
    if port is not None:
        netloc = f"{netloc}:{port}"
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def is_public_ip(ip: IPAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_public_ip(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            return is_public_ip(ip.sixtofour)
        if ip.teredo is not None:
            return False
    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
    )


def check_hostname_syntax(hostname: str) -> None:
    host = hostname.lower().rstrip(".")
    if host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_SUFFIXES):
        raise BlockedURLError(f"Host '{hostname}' is an internal hostname and cannot be analysed.")
    if _is_ip_literal(host):
        if not is_public_ip(ipaddress.ip_address(host)):
            raise BlockedURLError("URL points to a private or reserved IP address.")
        return
    labels = host.split(".")
    if labels[-1].isdigit() or host.replace(".", "").isdigit() or host.startswith("0x"):
        # stuff like 2130706433 or 127.1 - still resolves to localhost on some systems
        raise BlockedURLError("Numeric / obfuscated IP hostnames are not allowed.")
    if len(labels) < 2:
        raise InvalidURLError(f"'{hostname}' is not a fully-qualified domain name.")
    for label in labels:
        if not label or len(label) > 63 or label.startswith("-") or label.endswith("-"):
            raise InvalidURLError(f"'{hostname}' is not a valid domain name.")
        if not all(c.isalnum() or c == "-" for c in label):
            raise InvalidURLError(f"'{hostname}' is not a valid domain name.")


async def resolve_public_ip(hostname: str, port: int) -> str:
    """resolve and make sure ALL returned ips are public"""
    host = hostname.strip("[]")
    if _is_ip_literal(host):
        return host
    loop = asyncio.get_running_loop()
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP), timeout=5
        )
    except (socket.gaierror, asyncio.TimeoutError, UnicodeError) as exc:
        raise WebsiteUnreachableError(
            f"Could not resolve hostname '{hostname}'. Check the domain is correct and online."
        ) from exc
    addresses: list[IPAddress] = []
    for info in infos:
        addr = ipaddress.ip_address(info[4][0].split("%")[0])
        if addr not in addresses:
            addresses.append(addr)
    if not addresses:
        raise WebsiteUnreachableError(f"Hostname '{hostname}' did not resolve to any address.")
    for addr in addresses:
        if not is_public_ip(addr):
            raise BlockedURLError(
                "URL resolves to a private, loopback or reserved network address and cannot be analysed."
            )
    # ipv4 first, some hosts have no ipv6
    addresses.sort(key=lambda a: a.version)
    return str(addresses[0])


async def validate_public_url(raw: str, allowed_ports: list[int] | None = None) -> ValidatedURL:
    url = normalize_url(raw)
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if allowed_ports is not None and port not in allowed_ports:
        raise BlockedURLError(f"Port {port} is not allowed. Allowed ports: {sorted(allowed_ports)}.")
    check_hostname_syntax(hostname)
    ip = await resolve_public_ip(hostname, port)
    return ValidatedURL(url=url, scheme=parts.scheme, hostname=hostname, port=port, ip=ip)
