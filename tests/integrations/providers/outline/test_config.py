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


@pytest.mark.parametrize("bad", ["not-a-uuid", "abc", "-1", 7, None])
def test_collection_id_validates_uuid_shape(bad: object) -> None:
    with pytest.raises(OutlineProviderError, match="collection_id_invalid"):
        OutlineProviderConfig(
            base_url="https://wiki.example.com",
            api_token="t",
            collection_id=bad,  # type: ignore[arg-type]
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
        # Spellings that need no resolver at all.
        "https://localhost.",
        "https://LOCALHOST",
        "https://[::ffff:127.0.0.1]",
        "https://[2002:7f00:1::]",
        "https://[64:ff9b::7f00:1]",
    ],
)
def test_alternate_loopback_notations_rejected(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every spelling of loopback is a private origin.

    ``ipaddress.ip_address`` parses only dotted-quad IPv4, so classifying
    the literal let the numeric forms through as public and skipped the
    allowlist entirely. Deferring them to the resolver is not a fix either:
    only glibc's non-standard ``getaddrinfo`` parses them, so on musl
    (Alpine) they would fall through to DNS, NXDOMAIN, and a "public"
    verdict. ``localhost.`` defeats a bare ``==`` comparison,
    ``::ffff:127.0.0.1`` only classifies private through ``ipv4_mapped``
    from CPython 3.12.4 onwards, and the well-known NAT64 prefix
    ``64:ff9b::/96`` is not in :mod:`ipaddress`' private list at all.
    """
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)
    with pytest.raises(
        OutlineProviderError, match="private_origin_not_allowlisted"
    ):
        OutlineProviderConfig(base_url=url, api_token="t")


def test_numeric_loopback_is_classified_without_the_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The numeric IPv4 forms must not depend on libc parsing them."""
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)

    def _never(*args: object, **kwargs: object) -> list:
        raise AssertionError("the resolver must not be consulted")

    monkeypatch.setattr(config_mod.socket, "getaddrinfo", _never)
    for url in ("https://127.1", "https://2130706433", "https://0x7f.0.0.1"):
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


def test_unresolvable_host_is_lenient_at_config_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config time tolerates a host that does not resolve, but demands HTTPS.

    The leniency exists so a configuration can be saved offline; it is not
    a security decision, and :func:`assert_egress_allowed` closes it before
    any request goes out.
    """
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


def test_assert_egress_allowed_rejects_an_unresolvable_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request-time check fails closed where config time fails open.

    Without it, an attacker whose nameserver SERVFAILs during the
    configuration lookup and answers ``127.0.0.1`` afterwards reaches
    loopback with no allowlist entry - no rebinding TTL games needed, since
    nothing pins the address and every request re-resolves.
    """
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)

    def _nxdomain(*args: object, **kwargs: object) -> list:
        raise socket.gaierror("nxdomain")

    monkeypatch.setattr(config_mod.socket, "getaddrinfo", _nxdomain)
    with pytest.raises(OutlineProviderError, match="base_url_unresolvable"):
        config_mod.assert_egress_allowed("https://wiki.example.com")


def test_assert_egress_allowed_accepts_a_resolvable_public_host() -> None:
    assert config_mod.assert_egress_allowed("https://wiki.example.com") == (
        "https://wiki.example.com:443"
    )


# ---- URL parsing failures stay inside the taxonomy ----


@pytest.mark.parametrize(
    "url",
    [
        "https://wiki.example.com:99999",
        "https://wiki.example.com:abc",
        # NFKC-normalising netloc: ``urlsplit`` itself raises here.
        "https://loc℀alhost",
        "https://[::1",
    ],
    ids=["port-out-of-range", "port-not-numeric", "nfkc-netloc", "bad-ipv6"],
)
def test_unparseable_base_url_stays_in_the_taxonomy(url: str) -> None:
    """A bare ``ValueError`` from ``urlsplit`` must not escape.

    ``urlsplit(...).port`` raises before any range check can run, so the
    old ``if port < 1 or port > 65535`` guard was unreachable and callers
    saw a raw ``ValueError`` instead of a provider error.
    """
    with pytest.raises(
        OutlineProviderError,
        match="base_url_invalid|base_url_port_invalid",
    ):
        OutlineProviderConfig(base_url=url, api_token="t")


def test_zero_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """``:0`` must not be rewritten to the scheme default.

    ``urlsplit`` reports port ``0``, which the ``or 443`` fallback turned
    into ``https://localhost:443`` - an origin that could match an
    allowlist entry for a port the client can never connect to.
    """
    monkeypatch.setenv(
        "LDR_INTEGRATIONS_ALLOWED_ORIGINS", "https://localhost:443"
    )
    with pytest.raises(OutlineProviderError, match="base_url_port_invalid"):
        OutlineProviderConfig(base_url="https://localhost:0", api_token="t")


# ---- API token charset ----


@pytest.mark.parametrize(
    "token",
    ["tok\nen", "tok\ren", "tok\x00en", "tok\ten", "tokően", "tok en"],
    ids=["lf", "cr", "nul", "tab", "non-latin-1", "space"],
)
def test_api_token_with_header_hostile_characters_rejected(
    token: str,
) -> None:
    """Stripping is not enough; validate the whole token.

    ``.strip()`` only removes leading and trailing whitespace, so an
    *internal* newline still trips ``putheader``'s illegal-value check and
    a non-Latin-1 character still makes its ``encode("latin-1")`` raise a
    ``UnicodeEncodeError`` carrying the plaintext token.
    """
    with pytest.raises(OutlineProviderError, match="api_token_invalid"):
        OutlineProviderConfig(
            base_url="https://wiki.example.com", api_token=token
        )


def test_api_token_with_url_safe_punctuation_accepted() -> None:
    """Real bearer tokens must still load."""
    cfg = OutlineProviderConfig(
        base_url="https://wiki.example.com",
        api_token="eyJhbGciOi.J9-_~+/=.abc",
    )
    assert cfg.api_token == "eyJhbGciOi.J9-_~+/=.abc"


def test_load_outline_config_rejects_a_hostile_token() -> None:
    with pytest.raises(OutlineProviderError, match="api_token_invalid"):
        load_outline_config(
            _FakeSettings(**{"integration.outline.api_token": "tok\nen"})
        )


def test_repr_labels_the_config_fields() -> None:
    """The repr must stay diagnosable, not just token-free."""
    cfg = OutlineProviderConfig(
        base_url="https://wiki.example.com",
        api_token="outline-secret-token",
        collection_id="c1f9b8e2-1234-5678-9abc-def012345678",
        max_documents=5,
    )
    text = repr(cfg)
    assert "base_url='https://wiki.example.com'" in text
    assert "collection_id='c1f9b8e2-1234-5678-9abc-def012345678'" in text
    assert "max_documents=5" in text
    assert "api_token" not in text
