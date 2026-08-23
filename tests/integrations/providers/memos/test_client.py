# allow: no-sut-import - test_client exercises the HTTP client against a fake urlopen.
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from local_deep_research.integrations.providers.memos.client import (
    MemosClient,
)
from local_deep_research.integrations.providers.memos.config import (
    MemosProviderConfig,
)
from local_deep_research.integrations.providers.memos.errors import (
    MemosConnectionError,
    MemosProtocolError,
    MemosProviderError,
)


def _config(**overrides: object) -> MemosProviderConfig:
    values: dict[str, object] = {
        "base_url": "https://memos.example.com",
        "api_token": "secret-token",
    }
    values.update(overrides)
    return MemosProviderConfig(**values)  # type: ignore[arg-type]


def _install_urlopen(
    monkeypatch: pytest.MonkeyPatch, body: bytes, captured: list
) -> None:
    def _fake_urlopen(req: object, timeout: int | None = None) -> io.BytesIO:
        captured.append(req)
        return io.BytesIO(body)

    monkeypatch.setattr(
        "local_deep_research.integrations.providers.memos.client._OPENER.open",
        _fake_urlopen,
    )


def _install_urlopen_error(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    def _fake_urlopen(req: object, timeout: int | None = None) -> io.BytesIO:
        raise error

    monkeypatch.setattr(
        "local_deep_research.integrations.providers.memos.client._OPENER.open",
        _fake_urlopen,
    )


def _request_of(captured: list) -> object:
    assert len(captured) == 1
    return captured[0]


def _query_of(req: object) -> dict[str, list[str]]:
    return parse_qs(urlsplit(req.full_url).query)


class TestRequestShape:
    def test_probe_lists_one_memo_with_bearer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps({"memos": [{"name": "memos/x"}]}).encode(),
            captured,
        )
        MemosClient(_config()).probe()
        req = _request_of(captured)
        assert urlsplit(req.full_url)._replace(query="").geturl() == (
            "https://memos.example.com/api/v1/memos"
        )
        assert req.get_method() == "GET"
        assert req.get_header("Authorization") == "Bearer secret-token"
        assert _query_of(req) == {"pageSize": ["1"]}

    def test_list_memos_sends_page_size_and_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps(
                {"memos": [{"name": "memos/a"}], "nextPageToken": "tok-2"}
            ).encode(),
            captured,
        )
        batch, next_token = MemosClient(_config()).list_memos(
            page_token="tok-1", page_size=50
        )
        assert batch == [{"name": "memos/a"}]
        assert next_token == "tok-2"
        query = _query_of(_request_of(captured))
        assert query == {"pageSize": ["50"], "pageToken": ["tok-1"]}

    def test_list_memos_omits_empty_page_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps({"memos": []}).encode(),
            captured,
        )
        MemosClient(_config()).list_memos()
        assert _query_of(_request_of(captured)) == {"pageSize": ["100"]}

    def test_get_memo_fetches_by_uid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps({"name": "memos/n7", "content": "hi"}).encode(),
            captured,
        )
        memo = MemosClient(_config()).get_memo("n7")
        assert memo["name"] == "memos/n7"
        req = _request_of(captured)
        assert urlsplit(req.full_url)._replace(query="").geturl() == (
            "https://memos.example.com/api/v1/memos/n7"
        )


class TestEnvelopeParsing:
    def test_probe_non_list_memos_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(
            monkeypatch, json.dumps({"memos": "nope"}).encode(), []
        )
        with pytest.raises(MemosProtocolError, match="memos_not_list"):
            MemosClient(_config()).probe()

    def test_list_memos_missing_field_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(monkeypatch, json.dumps({"items": []}).encode(), [])
        with pytest.raises(MemosProtocolError, match="memos_not_list"):
            MemosClient(_config()).list_memos()

    def test_next_page_token_non_string_coerces_to_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(
            monkeypatch,
            json.dumps({"memos": [], "nextPageToken": 17}).encode(),
            [],
        )
        _, next_token = MemosClient(_config()).list_memos()
        assert next_token == ""

    def test_get_memo_not_object_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(monkeypatch, json.dumps([1, 2, 3]).encode(), [])
        with pytest.raises(MemosProtocolError, match="memo_not_object"):
            MemosClient(_config()).get_memo("n7")

    def test_malformed_json_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(monkeypatch, b"<<<not json", [])
        with pytest.raises(MemosProtocolError, match="json_decode_error"):
            MemosClient(_config()).probe()

    def test_empty_body_returns_none_and_probe_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(monkeypatch, b"", [])
        with pytest.raises(MemosProtocolError, match="memos_not_list"):
            MemosClient(_config()).probe()


class TestErrorMapping:
    def test_http_error_maps_to_protocol_error_with_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen_error(
            monkeypatch,
            urllib.error.HTTPError(
                "https://memos.example.com/api/v1/memos",
                401,
                "Unauthorized",
                None,
                io.BytesIO(b""),
            ),
        )
        with pytest.raises(MemosProtocolError, match="http_401"):
            MemosClient(_config()).probe()

    def test_url_error_maps_to_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen_error(
            monkeypatch, urllib.error.URLError("conn refused")
        )
        with pytest.raises(MemosConnectionError, match="url_error"):
            MemosClient(_config()).probe()

    @pytest.mark.parametrize(
        "error", [TimeoutError("t"), socket.gaierror("dns")]
    )
    def test_socket_errors_map_to_connection_error(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        _install_urlopen_error(monkeypatch, error)
        with pytest.raises(MemosConnectionError, match="connect_failed"):
            MemosClient(_config()).probe()


def test_client_is_a_context_manager() -> None:
    with MemosClient(_config()) as client:
        assert client.config.api_url == "https://memos.example.com/api/v1"


def test_no_redirect_handler_blocks_3xx() -> None:
    """The HTTP transport must not follow 3xx responses; the default
    Python handler copies the ``Authorization`` header onto redirects.
    """
    from local_deep_research.integrations.providers.memos import (
        client as client_mod,
    )

    handler = client_mod._NoRedirectHandler()
    redirected = handler.redirect_request(
        None,
        None,
        302,
        "Found",
        {"Location": "https://attacker.example/"},
        "https://attacker.example/",
    )
    assert redirected is None


def test_response_too_large_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body over the ceiling raises ``response_too_large``.

    Driven through the real client: an in-test re-implementation of the
    ceiling check passes even with the ceiling deleted from ``client.py``.
    The ceiling is shrunk rather than allocating 32 MiB per run.
    """
    from local_deep_research.integrations.providers.memos import (
        client as client_mod,
    )

    monkeypatch.setattr(client_mod, "_MAX_JSON_BYTES", 8)
    _install_urlopen(monkeypatch, b"x" * 9, [])
    with pytest.raises(MemosProtocolError, match="response_too_large"):
        MemosClient(_config()).probe()


def test_response_at_the_ceiling_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body exactly at the ceiling is still parsed."""
    from local_deep_research.integrations.providers.memos import (
        client as client_mod,
    )

    body = json.dumps({"memos": [], "nextPageToken": ""}).encode()
    monkeypatch.setattr(client_mod, "_MAX_JSON_BYTES", len(body))
    _install_urlopen(monkeypatch, body, [])
    MemosClient(_config()).probe()


def test_get_memo_url_encodes_uid_unsafe_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UID path segments must be percent-encoded with no safe characters.

    A literal slash in the user-defined UID must not request a different
    path segment; quote(safe="") encodes it.
    """
    captured: list = []
    body = json.dumps({"name": "memos/a/b", "content": "x"}).encode()
    _install_urlopen(monkeypatch, body, captured)
    MemosClient(_config()).get_memo("a/b")
    req = _request_of(captured)
    # The slash in "a/b" is percent-encoded to %2F.
    assert "%2F" in req.full_url
    assert "/memos/a/b" not in req.full_url.split("?")[0]


def _chain_texts(error: BaseException) -> list[str]:
    """Every string an error reporter could pull out of an exception chain."""
    texts: list[str] = []
    seen: set[int] = set()
    pending: list[BaseException | None] = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        texts += [repr(current), str(current), repr(current.args)]
        # ``UnicodeEncodeError.object`` holds the whole offending string.
        payload = getattr(current, "object", None)
        if payload is not None:
            texts.append(repr(payload))
        pending += [current.__context__, current.__cause__]
    return texts


@pytest.mark.parametrize(
    "transport_error",
    [
        ValueError("Invalid header value b'Bearer secret-token\\n'"),
        # ``putheader`` encodes the value as latin-1; a non-Latin-1
        # character raises this ``ValueError`` subclass, whose ``object``
        # attribute is the whole ``Bearer <token>`` string.
        UnicodeEncodeError(
            "latin-1",
            "Bearer secret-token\u0151",
            19,
            20,
            "ordinal not in range(256)",
        ),
    ],
    ids=["invalid-header-value", "non-latin-1-token"],
)
def test_transport_value_error_never_leaks_the_token(
    monkeypatch: pytest.MonkeyPatch, transport_error: BaseException
) -> None:
    """The token must not survive anywhere in the raised exception chain.

    ``raise ... from None`` is not enough: it sets ``__suppress_context__``,
    which only stops the traceback module from *printing* the chain, while
    ``error.__context__`` still holds the original exception with the token
    in its ``args`` (or, for ``UnicodeEncodeError``, in ``.object``). A
    structured error reporter or any ``while error.__context__`` walker
    reads it straight back out.
    """
    _install_urlopen_error(monkeypatch, transport_error)
    with pytest.raises(MemosProtocolError) as excinfo:
        MemosClient(_config()).probe()
    assert str(excinfo.value) == "memos_protocol:invalid_request"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    for text in _chain_texts(excinfo.value):
        assert "secret-token" not in text, text


def test_request_to_an_unresolvable_host_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config time tolerates an unresolvable host; request time must not.

    Otherwise an attacker whose nameserver SERVFAILs while the config is
    saved, and answers ``127.0.0.1`` afterwards, reaches loopback with no
    allowlist entry - there is no address pinning, so ``urllib``
    re-resolves on every call.
    """
    monkeypatch.delenv("LDR_INTEGRATIONS_ALLOWED_ORIGINS", raising=False)
    # Accepted at construction time (the conftest stub NXDOMAINs .invalid).
    config = MemosProviderConfig(
        base_url="https://memos.invalid", api_token="t"
    )
    _install_urlopen(monkeypatch, json.dumps({"memos": []}).encode(), [])
    with pytest.raises(MemosProviderError, match="base_url_unresolvable"):
        MemosClient(config).probe()


def test_repr_labels_the_base_url_field() -> None:
    """``__repr__`` must not label the derived API URL as ``base_url``."""
    assert repr(MemosClient(_config())) == (
        "MemosClient(base_url='https://memos.example.com')"
    )


def test_opener_ignores_environment_proxies() -> None:
    """``http_proxy``/``https_proxy``/``ALL_PROXY`` must not capture traffic.

    The opener is built at import time, so the check runs in a subprocess
    with the proxy variables set: urllib's default ``ProxyHandler`` would
    register itself from the environment and route token-bearing requests
    through a third party.
    """
    code = (
        "import urllib.request\n"
        "from local_deep_research.integrations.providers.memos "
        "import client as c\n"
        "print(any(isinstance(h, urllib.request.ProxyHandler) "
        "for h in c._OPENER.handlers))\n"
    )
    env = {
        **os.environ,
        "http_proxy": "http://proxy.invalid:3128",
        "https_proxy": "http://proxy.invalid:3128",
        "ALL_PROXY": "http://proxy.invalid:3128",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "False", result.stderr
