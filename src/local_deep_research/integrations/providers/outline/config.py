"""Outline provider configuration and egress policy.

Known limitation: origin classification resolves the configured host once,
at configuration time. A name that resolves to a public address here and to
a private address by the time the request is made (DNS rebinding) is not
caught; pinning the resolved address for the lifetime of a request is out of
scope for this module.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse
from dataclasses import dataclass
from typing import Protocol

from .errors import OutlineProviderError


class SettingsReader(Protocol):
    def get_setting(
        self, key: str, default: str | int | bool | None = None
    ) -> str | int | bool | None: ...


_ALLOWED_ORIGINS_ENV = "LDR_INTEGRATIONS_ALLOWED_ORIGINS"
_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True, slots=True, repr=False)
class OutlineProviderConfig:
    """Configuration for syncing Outline documents via REST API.

    Targets the Outline HTTP API (``/api/v1``) which exposes documents
    and collections with stable UUIDs, titles, Markdown content, and
    update timestamps.
    """

    base_url: str
    api_token: str
    collection_id: str = ""
    max_documents: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise OutlineProviderError("base_url_missing")
        if not isinstance(self.api_token, str) or not self.api_token.strip():
            raise OutlineProviderError("api_token_missing")
        if not isinstance(self.collection_id, str) or (
            self.collection_id and not _is_uuid(self.collection_id)
        ):
            raise OutlineProviderError("collection_id_invalid")
        if (
            isinstance(self.max_documents, bool)
            or not isinstance(self.max_documents, int)
            or self.max_documents < 0
        ):
            raise OutlineProviderError("max_documents_invalid")
        # Egress policy: enforce origin safety at construction time so the
        # HTTP client never sees an unsafe target.
        canonical_origin(self.base_url)

    @property
    def api_url(self) -> str:
        base = self.base_url.rstrip("/")
        return f"{base}/api"

    def __repr__(self) -> str:
        return (
            f"OutlineProviderConfig("
            f"base_url={self.base_url!r}, "
            f"collection_id={self.collection_id!r}, "
            f"max_documents={self.max_documents})"
        )


def load_outline_config(settings: SettingsReader) -> OutlineProviderConfig:
    base_url = settings.get_setting("integration.outline.base_url", "")
    api_token = settings.get_setting("integration.outline.api_token", "")
    collection_id = settings.get_setting(
        "integration.outline.collection_id", ""
    )
    max_documents_raw = settings.get_setting(
        "integration.outline.max_documents", 0
    )

    if not isinstance(base_url, str) or not base_url.strip():
        raise OutlineProviderError("base_url_missing")
    if not isinstance(api_token, str) or not api_token.strip():
        raise OutlineProviderError("api_token_missing")
    if not isinstance(collection_id, str) or (
        collection_id and not _is_uuid(collection_id)
    ):
        raise OutlineProviderError("collection_id_invalid")
    max_documents = _coerce_int(max_documents_raw, "max_documents")
    if max_documents < 0:
        raise OutlineProviderError("max_documents_invalid")
    canonical_origin(base_url)

    return OutlineProviderConfig(
        base_url=base_url.strip(),
        # Strip before use: a token pasted with a trailing newline or
        # surrounding spaces is otherwise rejected by ``http.client`` with a
        # ``ValueError`` that quotes the whole header value, or silently 401s.
        api_token=api_token.strip(),
        collection_id=collection_id.strip(),
        max_documents=max_documents,
    )


def _coerce_int(value: str | int | bool | None, field_name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise OutlineProviderError(f"{field_name}_invalid")
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise OutlineProviderError(f"{field_name}_invalid") from error


def _is_uuid(value: str) -> bool:
    """Lenient UUID check (any hex dashes, 8-4-4-4-12 or 32 contiguous)."""
    stripped = value.replace("-", "")
    if len(stripped) != 32:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in stripped)


def canonical_origin(raw: str) -> str:
    """Return the canonical origin ``"scheme://host:port"`` or raise.

    Enforces the integration egress policy at configuration time:

    * Schemes must be ``http`` or ``https``.
    * Userinfo, query, and fragment components are forbidden.
    * Public (non-private) origins must use HTTPS.
    * Private / loopback origins must be listed verbatim in the
      ``LDR_INTEGRATIONS_ALLOWED_ORIGINS`` environment variable
      (comma-separated canonical origins).
    """
    parts = urllib.parse.urlsplit(raw.strip())
    if parts.scheme not in ("http", "https"):
        raise OutlineProviderError("base_url_scheme_invalid")
    if not parts.hostname:
        raise OutlineProviderError("base_url_host_missing")
    if parts.username is not None or parts.password is not None:
        raise OutlineProviderError("base_url_userinfo_forbidden")
    if parts.query or parts.fragment:
        raise OutlineProviderError("base_url_query_or_fragment_forbidden")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port < 1 or port > 65535:
        raise OutlineProviderError("base_url_port_invalid")

    origin = f"{parts.scheme.lower()}://{parts.hostname.lower()}:{port}"
    if not _is_private_loopback(parts.hostname) and parts.scheme == "http":
        raise OutlineProviderError("public_http_forbidden")
    if _is_private_loopback(parts.hostname) and not _origin_allowed(origin):
        raise OutlineProviderError("private_origin_not_allowlisted")
    return origin


def _is_private_loopback(host: str) -> bool:
    """Return ``True`` when ``host`` designates a private/loopback target.

    Classification runs on *resolved* addresses, never on the literal
    string. ``127.1``, ``2130706433``, ``0x7f.0.0.1`` and ``017700000001``
    are all spellings of ``127.0.0.1`` that :func:`ipaddress.ip_address`
    rejects but the resolver accepts, and a public DNS name may carry a
    private ``A`` record. Every address the host resolves to is checked, so
    one private answer makes the origin private.
    """
    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
    ):
        return True
    return any(
        address.is_loopback or address.is_private or address.is_link_local
        for address in _resolve_host(host)
    )


def _resolve_host(host: str) -> tuple[_IPAddress, ...]:
    """Resolve ``host`` to every address it currently maps to.

    An IP literal is returned without a lookup. A host that cannot be
    resolved yields no addresses and is therefore treated as public: it is
    unreachable anyway, and the HTTPS requirement still applies to it.
    """
    literal = _parse_ip_address(host)
    if literal is not None:
        return (literal,)
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError, ValueError):
        return ()
    resolved: list[_IPAddress] = []
    for info in infos:
        # Strip any IPv6 zone index before parsing (``fe80::1%eth0``).
        address = _parse_ip_address(str(info[4][0]).partition("%")[0])
        if address is not None:
            resolved.append(address)
    return tuple(resolved)


def _parse_ip_address(value: str) -> _IPAddress | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _origin_allowed(origin: str) -> bool:
    raw = os.environ.get(_ALLOWED_ORIGINS_ENV, "")
    if not raw.strip():
        return False
    return origin in {
        entry.strip() for entry in raw.split(",") if entry.strip()
    }
