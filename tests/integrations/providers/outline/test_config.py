# allow: no-sut-import - test_config validates config dataclass, not production SUT.
from __future__ import annotations

import socket

import pytest

from local_deep_research.integrations.providers.outline import (
    config as config_mod,
)
from local_deep_research.integrations.providers.outline.config import (
    OutlineProviderConfig,
    load_outline_config,
)
from local_deep_research.integrations.providers.outline.errors import (
    OutlineProviderError,
)


class _FakeSettings:
    def __init__(self, **kwargs: object) -> None:
        self._data = {
            "integration.outline.base_url": "https://wiki.example.com",
            "integration.outline.api_token": "ol-secret-token",
            "integration.outline.collection_id": "",
            "integration.outline.max_documents": 0,
        }
        self._data.update(kwargs)

    def get_setting(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


def test_valid_config() -> None:
    cfg = OutlineProviderConfig(
        base_url="https://wiki.example.com",
        api_token="token",
    )
    assert cfg.base_url == "https://wiki.example.com"
    assert cfg.collection_id == ""
    assert cfg.max_documents == 0


def test_api_url_strips_trailing_slash() -> None:
    cfg = OutlineProviderConfig(
        base_url="https://wiki.example.com/",
        api_token="t",
    )
    assert cfg.api_url == "https://wiki.example.com/api"


def test_missing_base_url_raises() -> None:
    with pytest.raises(OutlineProviderError, match="base_url_missing"):
        OutlineProviderConfig(base_url="", api_token="t")


def test_missing_api_token_raises() -> None:
    with pytest.raises(OutlineProviderError, match="api_token_missing"):
        OutlineProviderConfig(base_url="https://wiki.example.com", api_token="")


def test_invalid_collection_id_raises() -> None:
    with pytest.raises(OutlineProviderError, match="collection_id_invalid"):
        OutlineProviderConfig(
            base_url="https://wiki.example.com",
            api_token="t",
            collection_id=7,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad", [-1, True, "abc", None])
def test_invalid_max_documents_raises(bad: object) -> None:
    with pytest.raises(OutlineProviderError, match="max_documents_invalid"):
        OutlineProviderConfig(
            base_url="https://wiki.example.com",
            api_token="t",
            max_documents=bad,  # type: ignore[arg-type]
        )


def test_repr_hides_api_token() -> None:
    cfg = OutlineProviderConfig(
        base_url="https://wiki.example.com",
        api_token="ol-secret-token",
    )
    assert "ol-secret-token" not in repr(cfg)


def test_load_outline_config_reads_settings() -> None:
    cfg = load_outline_config(_FakeSettings())
    assert cfg.base_url == "https://wiki.example.com"
    assert cfg.api_token == "ol-secret-token"
    assert cfg.collection_id == ""
    assert cfg.max_documents == 0


def test_load_outline_config_coerces_string_max_documents() -> None:
    cfg = load_outline_config(
        _FakeSettings(**{"integration.outline.max_documents": "25"})
    )
    assert cfg.max_documents == 25


def test_load_outline_config_missing_token_raises() -> None:
    with pytest.raises(OutlineProviderError, match="api_token_missing"):
        load_outline_config(
            _FakeSettings(**{"integration.outline.api_token": ""})
        )


def test_load_outline_config_invalid_max_documents_raises() -> None:
    with pytest.raises(OutlineProviderError, match="max_documents_invalid"):
        load_outline_config(
            _FakeSettings(**{"integration.outline.max_documents": "abc"})
        )


# ---- Origin policy (SSRF / egress) ----


@pytest.mark.parametrize(
    "url",
    [
        "ftp://wiki.example.com",
        "gopher://wiki.example.com",
        "javascript:alert(1)",
    ],
)
def test_non_http_scheme_rejected(url: str) -> None:
    with pytest.raises(OutlineProviderError, match="base_url_scheme_invalid"):
        OutlineProviderConfig(base_url=url, api_token="t")


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@wiki.example.com",
        "https://wiki.example.com/path?q=1",
        "https://wiki.example.com/path#frag",
    ],
)
def test_userinfo_query_fragment_rejected(url: str) -> None:
    with pytest.raises(
        OutlineProviderError,
        match=(
            "base_url_userinfo_forbidden|base_url_query_or_fragment_forbidden"
        ),
    ):
        OutlineProviderConfig(base_url=url, api_token="t")


def test_public_http_rejected() -> None:
    with pytest.raises(OutlineProviderError, match="public_http_forbidden"):
        OutlineProviderConfig(base_url="http://wiki.example.com", api_token="t")


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost",
        "https://127.0.0.1",
        "https://192.168.1.10",
        "https://10.0.0.5",
    ],
)
def test_private_origin_rejected_without_allowlist(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)
    with pytest.raises(
        OutlineProviderError,
        match="private_origin_not_allowlisted",
    ):
        OutlineProviderConfig(base_url=url, api_token="t")


def test_private_origin_allowed_when_in_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LDR_INTEGRATIONS_ALLOWED_ORIGINS",
        "https://localhost:443,https://other.example.com",
    )
    cfg = OutlineProviderConfig(base_url="https://localhost", api_token="t")
    assert cfg.api_url == "https://localhost/api"


def test_private_origin_rejected_when_missing_from_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", "https://other:443")
    with pytest.raises(
        OutlineProviderError,
        match="private_origin_not_allowlisted",
    ):
        OutlineProviderConfig(base_url="https://localhost", api_token="t")


def test_collection_id_validates_uuid_shape() -> None:
    with pytest.raises(OutlineProviderError, match="collection_id_invalid"):
        OutlineProviderConfig(
            base_url="https://wiki.example.com",
            api_token="t",
            collection_id="not-a-uuid",
        )


def test_collection_id_accepts_valid_uuid() -> None:
    cfg = OutlineProviderConfig(
        base_url="https://wiki.example.com",
        api_token="t",
        collection_id="c1f9b8e2-1234-5678-9abc-def012345678",
    )
    assert cfg.collection_id.startswith("c1f9b8e2")


def test_load_outline_config_strips_api_token() -> None:
    """A token pasted with surrounding whitespace must not reach the header.

    ``http.client.putheader`` rejects ``Bearer tok\n`` with a ``ValueError``
    that quotes the whole header value, and ``Bearer  tok `` silently 401s.
    """
    cfg = load_outline_config(
        _FakeSettings(**{"integration.outline.api_token": " outline-token\n"})
    )
    assert cfg.api_token == "outline-token"


@pytest.mark.parametrize(
    "url",
    [
        "https://127.1",
        "https://2130706433",
        "https://0x7f.0.0.1",
        "https://017700000001",
        "https://[::1]",
    ],
)
def test_alternate_loopback_notations_rejected(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every spelling of loopback the resolver accepts is a private origin.

    ``ipaddress.ip_address`` parses only dotted-quad IPv4, so classifying the
    literal let these through as public and skipped the allowlist entirely.
    """
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)
    with pytest.raises(
        OutlineProviderError, match="private_origin_not_allowlisted"
    ):
        OutlineProviderConfig(base_url=url, api_token="t")


def test_public_name_with_private_address_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public DNS name pointing at RFC1918 space is still a private origin."""
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(
        config_mod.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("10.1.2.3", 443))],
    )
    with pytest.raises(
        OutlineProviderError, match="private_origin_not_allowlisted"
    ):
        OutlineProviderConfig(
            base_url="https://wiki.example.com", api_token="t"
        )


def test_unresolvable_host_is_classified_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolvable host needs no allowlist but still must use HTTPS."""
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)

    def _nxdomain(*args: object, **kwargs: object) -> list:
        raise socket.gaierror("nxdomain")

    monkeypatch.setattr(config_mod.socket, "getaddrinfo", _nxdomain)
    cfg = OutlineProviderConfig(
        base_url="https://wiki.example.com", api_token="t"
    )
    assert cfg.api_url == "https://wiki.example.com/api"
    with pytest.raises(OutlineProviderError, match="public_http_forbidden"):
        OutlineProviderConfig(base_url="http://wiki.example.com", api_token="t")


def test_resolved_private_origin_allowed_when_allowlisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allowlist still opens the door for a deliberately private target."""
    monkeypatch.setenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", "https://127.1:443")
    cfg = OutlineProviderConfig(base_url="https://127.1", api_token="t")
    assert cfg.api_url == "https://127.1/api"
