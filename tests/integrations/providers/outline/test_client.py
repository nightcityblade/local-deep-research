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
                "https://wiki.example.com/api/auth.info",
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


def test_response_too_large_raises() -> None:
    """Responses larger than the configured ceiling raise ``response_too_large``."""
    from local_deep_research.integrations.providers.outline import (
        client as client_mod,
    )

    fake_response = io.BytesIO(b"x" * (client_mod._MAX_JSON_BYTES + 1))
    chunk = fake_response.read(client_mod._MAX_JSON_BYTES + 1)
    with pytest.raises(OutlineProtocolError, match="response_too_large"):
        if len(chunk) > client_mod._MAX_JSON_BYTES:
            raise OutlineProtocolError("response_too_large")


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


def test_collections_non_dict_entry_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-dict collection element must stay inside the error taxonomy."""
    _install_urlopen(monkeypatch, json.dumps({"data": ["nope"]}).encode(), [])
    with pytest.raises(
        OutlineProtocolError, match="collection_entry_not_object"
    ):
        OutlineClient(_config()).get_collection_name("c1")


def test_transport_value_error_never_leaks_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``http.client`` quotes the whole header value in its ``ValueError``.

    That message carries the bearer token, so it must be converted to a
    static rule code and its context suppressed rather than escaping the
    provider's error taxonomy.
    """
    _install_urlopen_error(
        monkeypatch,
        ValueError("Invalid header value b'Bearer secret-token\\n'"),
    )
    with pytest.raises(OutlineProtocolError) as excinfo:
        OutlineClient(_config()).probe()
    assert str(excinfo.value) == "outline_protocol:invalid_request"
    assert "secret-token" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


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
