# allow: no-sut-import - test_client exercises the HTTP client against a fake urlopen.
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import urllib.error

import pytest

from local_deep_research.integrations.providers.outline.client import (
    OutlineClient,
)
from local_deep_research.integrations.providers.outline.config import (
    OutlineProviderConfig,
)
from local_deep_research.integrations.providers.outline.errors import (
    OutlineConnectionError,
    OutlineProtocolError,
    OutlineProviderError,
)


def _config(**overrides: object) -> OutlineProviderConfig:
    values: dict[str, object] = {
        "base_url": "https://wiki.example.com",
        "api_token": "secret-token",
    }
    values.update(overrides)
    return OutlineProviderConfig(**values)  # type: ignore[arg-type]


def _install_urlopen(
    monkeypatch: pytest.MonkeyPatch, body: bytes, captured: list
) -> None:
    def _fake_urlopen(req: object, timeout: int | None = None) -> io.BytesIO:
        captured.append(req)
        return io.BytesIO(body)

    monkeypatch.setattr(
        "local_deep_research.integrations.providers.outline.client._OPENER"
        ".open",
        _fake_urlopen,
    )


def _install_urlopen_error(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    def _fake_urlopen(req: object, timeout: int | None = None) -> io.BytesIO:
        raise error

    monkeypatch.setattr(
        "local_deep_research.integrations.providers.outline.client._OPENER"
        ".open",
        _fake_urlopen,
    )


def _request_of(captured: list) -> object:
    assert len(captured) == 1
    return captured[0]


class TestRequestShape:
    def test_probe_lists_one_document_with_bearer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe must exercise the endpoint the sync actually calls.

        ``auth.info`` succeeds for any valid token, so it reported
        "connection successful" while ``documents.list`` returned 403.
        """
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps({"data": []}).encode(),
            captured,
        )
        OutlineClient(_config()).probe()
        req = _request_of(captured)
        assert req.full_url == "https://wiki.example.com/api/documents.list"
        assert req.get_method() == "POST"
        assert req.get_header("Authorization") == "Bearer secret-token"
        assert json.loads(req.data.decode("utf-8")) == {
            "offset": 0,
            "limit": 1,
        }

    def test_list_documents_sends_pagination_and_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps({"data": [{"id": "d1"}]}).encode(),
            captured,
        )
        client = OutlineClient(_config())
        batch = client.list_documents(
            offset=100, limit=100, collection_id="col-9"
        )
        assert batch == [{"id": "d1"}]
        req = _request_of(captured)
        assert req.full_url == "https://wiki.example.com/api/documents.list"
        body = json.loads(req.data.decode("utf-8"))
        assert body == {"offset": 100, "limit": 100, "collectionId": "col-9"}

    def test_list_documents_omits_empty_collection_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps({"data": []}).encode(),
            captured,
        )
        OutlineClient(_config()).list_documents()
        body = json.loads(_request_of(captured).data.decode("utf-8"))
        assert body == {"offset": 0, "limit": 100}

    def test_get_document_requests_info_with_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps({"data": {"id": "d1", "title": "T"}}).encode(),
            captured,
        )
        OutlineClient(_config()).get_document("d1")
        req = _request_of(captured)
        assert req.full_url == "https://wiki.example.com/api/documents.info"
        assert json.loads(req.data.decode("utf-8")) == {"id": "d1"}

    def test_get_collection_name_caches_collections(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list = []
        _install_urlopen(
            monkeypatch,
            json.dumps(
                {"data": [{"id": "c1", "name": "Engineering"}]}
            ).encode(),
            captured,
        )
        client = OutlineClient(_config())
        assert client.get_collection_name("c1") == "Engineering"
        assert client.get_collection_name("c1") == "Engineering"
        assert client.get_collection_name("missing") == ""
        # One collections.list call serves every lookup after the first.
        assert len(captured) == 1


class TestEnvelopeParsing:
    def test_probe_data_not_a_list_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(
            monkeypatch, json.dumps({"data": {"teams": []}}).encode(), []
        )
        with pytest.raises(OutlineProtocolError, match="documents_not_list"):
            OutlineClient(_config()).probe()

    def test_envelope_without_data_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(
            monkeypatch, json.dumps({"ok": False, "error": "boom"}).encode(), []
        )
        with pytest.raises(OutlineProtocolError, match="envelope_missing_data"):
            OutlineClient(_config()).probe()

    def test_documents_data_not_list_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(
            monkeypatch, json.dumps({"data": {"unexpected": True}}).encode(), []
        )
        with pytest.raises(OutlineProtocolError, match="documents_not_list"):
            OutlineClient(_config()).list_documents()

    def test_document_not_object_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(monkeypatch, json.dumps({"data": [1, 2]}).encode(), [])
        with pytest.raises(OutlineProtocolError, match="document_not_object"):
            OutlineClient(_config()).get_document("d1")

    def test_malformed_json_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(monkeypatch, b"not-json{", [])
        with pytest.raises(OutlineProtocolError, match="json_decode_error"):
            OutlineClient(_config()).probe()

    def test_empty_body_returns_none_and_probe_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen(monkeypatch, b"", [])
        with pytest.raises(OutlineProtocolError, match="documents_not_list"):
            OutlineClient(_config()).probe()


class TestErrorMapping:
    def test_http_error_maps_to_protocol_error_with_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen_error(
            monkeypatch,
            urllib.error.HTTPError(
                "https://wiki.example.com/api/documents.list",
                401,
                "Unauthorized",
                None,
                io.BytesIO(b""),
            ),
        )
        with pytest.raises(OutlineProtocolError, match="http_401"):
            OutlineClient(_config()).probe()

    def test_url_error_maps_to_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_urlopen_error(
            monkeypatch, urllib.error.URLError("no route to host")
        )
        with pytest.raises(OutlineConnectionError, match="url_error"):
            OutlineClient(_config()).probe()

    @pytest.mark.parametrize(
        "error", [TimeoutError("t"), socket.gaierror("dns")]
    )
    def test_socket_errors_map_to_connection_error(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        _install_urlopen_error(monkeypatch, error)
        with pytest.raises(OutlineConnectionError, match="connect_failed"):
            OutlineClient(_config()).probe()


def test_client_is_a_context_manager() -> None:
    with OutlineClient(_config()) as client:
        assert client.config.base_url == "https://wiki.example.com"


def test_no_redirect_handler_blocks_3xx() -> None:
    """The HTTP transport must not follow 3xx responses; the default
    Python handler copies the ``Authorization`` header onto redirects.
    """
    from local_deep_research.integrations.providers.outline import (
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
    from local_deep_research.integrations.providers.outline import (
        client as client_mod,
    )

    monkeypatch.setattr(client_mod, "_MAX_JSON_BYTES", 8)
    _install_urlopen(monkeypatch, b"x" * 9, [])
    with pytest.raises(OutlineProtocolError, match="response_too_large"):
        OutlineClient(_config()).probe()


def test_response_at_the_ceiling_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body exactly at the ceiling is still parsed."""
    from local_deep_research.integrations.providers.outline import (
        client as client_mod,
    )

    body = json.dumps({"data": []}).encode()
    monkeypatch.setattr(client_mod, "_MAX_JSON_BYTES", len(body))
    _install_urlopen(monkeypatch, body, [])
    OutlineClient(_config()).probe()


def _install_paged_urlopen(
    monkeypatch: pytest.MonkeyPatch, bodies: list[bytes], captured: list
) -> None:
    def _fake_urlopen(req: object, timeout: int | None = None) -> io.BytesIO:
        captured.append(req)
        index = min(len(captured) - 1, len(bodies) - 1)
        return io.BytesIO(bodies[index])

    monkeypatch.setattr(
        "local_deep_research.integrations.providers.outline.client._OPENER"
        ".open",
        _fake_urlopen,
    )


def test_collections_pagination_terminates_on_empty_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty page ends the collections loop even at a page boundary."""
    from local_deep_research.integrations.providers.outline import (
        client as client_mod,
    )

    full_page = json.dumps(
        {
            "data": [
                {"id": f"c{i}", "name": f"N{i}"}
                for i in range(client_mod._COLLECTIONS_PAGE_SIZE)
            ]
        }
    ).encode()
    captured: list = []
    _install_paged_urlopen(
        monkeypatch, [full_page, json.dumps({"data": []}).encode()], captured
    )
    assert OutlineClient(_config()).get_collection_name("c0") == "N0"
    assert len(captured) == 2


def test_collections_pagination_valve_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server replaying a full page forever must trip the safety valve.

    The valve counts rows fetched, not rows cached: replaying the *same*
    page never grows the cache, so a cache-size check would still spin.
    """
    from local_deep_research.integrations.providers.outline import (
        client as client_mod,
    )

    monkeypatch.setattr(client_mod, "_MAX_PAGINATED_COLLECTIONS", 150)
    full_page = json.dumps(
        {
            "data": [
                {"id": f"c{i}", "name": f"N{i}"}
                for i in range(client_mod._COLLECTIONS_PAGE_SIZE)
            ]
        }
    ).encode()
    _install_paged_urlopen(monkeypatch, [full_page], [])
    with pytest.raises(
        OutlineProtocolError, match="pagination_not_terminating"
    ):
        OutlineClient(_config()).get_collection_name("c0")


def test_collections_non_dict_entry_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One malformed collection row must not deny the whole sync.

    ``get_collection_name`` is a display-name lookup on the per-document
    path, so raising here failed every remaining document - and left
    ``_collections_cache`` unset, so each one re-fetched the entire
    listing. Rows whose id or name is not a string were already skipped;
    a non-dict row is now treated the same way.
    """
    body = json.dumps(
        {"data": ["nope", {"id": "c1", "name": "Ops"}, None]}
    ).encode()
    _install_urlopen(monkeypatch, body, [])
    client = OutlineClient(_config())
    assert client.get_collection_name("c1") == "Ops"
    assert client.get_collection_name("unknown") == ""


def test_collections_listing_is_fetched_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache is populated even when a row was malformed."""
    captured: list = []
    body = json.dumps({"data": ["nope"]}).encode()
    _install_urlopen(monkeypatch, body, captured)
    client = OutlineClient(_config())
    assert client.get_collection_name("c1") == ""
    assert client.get_collection_name("c2") == ""
    assert len(captured) == 1


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
    with pytest.raises(OutlineProtocolError) as excinfo:
        OutlineClient(_config()).probe()
    assert str(excinfo.value) == "outline_protocol:invalid_request"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    for text in _chain_texts(excinfo.value):
        assert "secret-token" not in text, text


def test_repr_labels_the_base_url_field() -> None:
    """``__repr__`` must not label the derived API URL as ``base_url``."""
    assert repr(OutlineClient(_config())) == (
        "OutlineClient(base_url='https://wiki.example.com')"
    )


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
    config = OutlineProviderConfig(
        base_url="https://wiki.invalid", api_token="t"
    )
    _install_urlopen(monkeypatch, json.dumps({"data": []}).encode(), [])
    with pytest.raises(OutlineProviderError, match="base_url_unresolvable"):
        OutlineClient(config).probe()


def test_opener_ignores_environment_proxies() -> None:
    """``http_proxy``/``https_proxy``/``ALL_PROXY`` must not capture traffic.

    The opener is built at import time, so the check runs in a subprocess
    with the proxy variables set: urllib's default ``ProxyHandler`` would
    register itself from the environment and route token-bearing requests
    through a third party.
    """
    code = (
        "import urllib.request\n"
        "from local_deep_research.integrations.providers.outline "
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
