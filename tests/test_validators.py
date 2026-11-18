import asyncio
import socket

import pytest

from app.errors import BlockedURLError, InvalidURLError, WebsiteUnreachableError
from app.utils import validators
from app.utils.validators import normalize_url, validate_public_url
from tests.conftest import REAL_RESOLVE_PUBLIC_IP

PORTS = [80, 443, 8080, 8443]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "https://example.com/"),
        ("  https://Example.COM/path?q=1#frag ", "https://example.com/path?q=1"),
        ("http://example.com:8080", "http://example.com:8080/"),
        ("https://bücher.de", "https://xn--bcher-kva.de/"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "ftp://example.com", "javascript:alert(1)", "https://exa mple.com", "https://example.com:99999",
     "https://", "file:///etc/passwd"],
)
def test_invalid_urls_rejected(raw: str) -> None:
    with pytest.raises((InvalidURLError, BlockedURLError)):
        asyncio.run(validate_public_url(raw, PORTS))


@pytest.mark.parametrize(
    "raw",
    [
        "http://127.0.0.1",
        "http://10.0.0.5",
        "http://192.168.1.1",
        "http://172.16.0.1",
        "http://169.254.169.254/latest/meta-data",  # cloud metadata
        "http://100.64.0.1",  # carrier-grade NAT
        "http://0.0.0.0",
        "http://[::1]",
        "http://[::ffff:127.0.0.1]",  # IPv4-mapped IPv6
        "http://[fd00::1]",  # unique local IPv6
        "http://localhost",
        "http://LOCALHOST.",
        "http://app.localhost",
        "http://metadata.google.internal",
        "http://printer.local",
        "http://2130706433",  # decimal-encoded 127.0.0.1
        "http://127.1",
        "http://user:pass@example.com",
        "http://example.com:22",  # disallowed port
    ],
)
def test_ssrf_targets_blocked(raw: str) -> None:
    with pytest.raises(BlockedURLError):
        asyncio.run(validate_public_url(raw, PORTS))


def test_public_ip_literal_allowed() -> None:
    result = asyncio.run(validate_public_url("http://93.184.216.34", PORTS))
    assert result.ip == "93.184.216.34"


def _fake_getaddrinfo(addresses: list[str]):
    async def getaddrinfo(host, port, **_):
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
                for a in addresses]
    return getaddrinfo


@pytest.mark.parametrize(
    "addresses",
    [["10.1.2.3"], ["93.184.216.34", "127.0.0.1"], ["::1"]],  # any private address -> blocked (rebinding)
)
def test_dns_resolving_to_private_address_blocked(monkeypatch, addresses) -> None:
    monkeypatch.setattr(validators, "resolve_public_ip", REAL_RESOLVE_PUBLIC_IP)

    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", _fake_getaddrinfo(addresses))
        await validate_public_url("https://evil-rebind.example", PORTS)

    with pytest.raises(BlockedURLError):
        asyncio.run(run())


def test_dns_resolving_to_public_address_pins_ip(monkeypatch) -> None:
    monkeypatch.setattr(validators, "resolve_public_ip", REAL_RESOLVE_PUBLIC_IP)

    async def run():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", _fake_getaddrinfo(["2606:2800::1", "93.184.216.34"]))
        return await validate_public_url("https://example.com", PORTS)

    result = asyncio.run(run())
    assert result.ip == "93.184.216.34"  # IPv4 preferred
    assert result.hostname == "example.com"


def test_unresolvable_domain(monkeypatch) -> None:
    monkeypatch.setattr(validators, "resolve_public_ip", REAL_RESOLVE_PUBLIC_IP)

    async def run():
        loop = asyncio.get_running_loop()

        async def fail(*_, **__):
            raise socket.gaierror("nodename nor servname provided")

        monkeypatch.setattr(loop, "getaddrinfo", fail)
        await validate_public_url("https://does-not-exist.example", PORTS)

    with pytest.raises(WebsiteUnreachableError):
        asyncio.run(run())
