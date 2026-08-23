from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from types import TracebackType
from typing import Any

from .config import MemosProviderConfig, assert_egress_allowed
from .errors import MemosConnectionError, MemosProtocolError

_MAX_JSON_BYTES = 32 * 1024 * 1024  # 32 MiB response-body ceiling.


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Block 3xx redirects so the ``Authorization`` header cannot leak
    to a different origin via Python's default redirect handling."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None


# ``ProxyHandler({})`` replaces urllib's default handler, which reads
# ``http_proxy``/``https_proxy``/``ALL_PROXY`` from the environment. Integration
# traffic carries a bearer token, so it must never be routed through a third
# party because of an unrelated environment variable (the project uses
# ``trust_env = False`` for the same reason elsewhere).
_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _NoRedirectHandler()
)


class MemosClient:
    """HTTP client for the Memos REST API.

    Uses Bearer-token authentication (a personal access token created in
    Memos under Settings → Access Tokens). The API is served at
    ``/api/v1``; list endpoints paginate with ``pageSize``/``pageToken``.
    """

    def __init__(self, config: MemosProviderConfig) -> None:
        self._config = config
        self._egress_checked = False

    def __enter__(self) -> MemosClient:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"MemosClient(base_url={self._config.base_url!r})"

    @property
    def config(self) -> MemosProviderConfig:
        return self._config

    def probe(self) -> None:
        """Verify connectivity and credentials with a minimal list call."""
        data = self._get_json("memos", {"pageSize": "1"})
        if not isinstance(data, dict) or not isinstance(
            data.get("memos"), list
        ):
            raise MemosProtocolError("memos_not_list")

    def list_memos(
        self,
        *,
        page_token: str = "",
        page_size: int = 100,
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch one batch of memos and the next page token ("" = end)."""
        params: dict[str, str] = {"pageSize": str(page_size)}
        if page_token:
            params["pageToken"] = page_token
        data = self._get_json("memos", params)
        if not isinstance(data, dict) or not isinstance(
            data.get("memos"), list
        ):
            raise MemosProtocolError("memos_not_list")
        next_token = data.get("nextPageToken", "")
        if not isinstance(next_token, str):
            next_token = ""
        return data["memos"], next_token

    def get_memo(self, memo_uid: str) -> dict[str, Any]:
        """Fetch a single memo with full content by its UID.

        The UID is percent-encoded with ``safe=""`` so a literal slash in a
        user-defined UID cannot request a different path segment.
        """
        data = self._get_json(
            f"memos/{urllib.parse.quote(memo_uid, safe='')}", None
        )
        if not isinstance(data, dict) or "name" not in data:
            raise MemosProtocolError("memo_not_object")
        return data

    def close(self) -> None:
        """No persistent resources to clean up."""

    def _get_json(self, path: str, params: dict[str, str] | None) -> Any:
        self._assert_egress_allowed()
        url = f"{self._config.api_url}/{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        # ``http.client`` rejects a malformed header value with a
        # ``ValueError`` - or, for a non-Latin-1 character, a
        # ``UnicodeEncodeError`` - that carries the whole header, including
        # the bearer token. The replacement is built here and raised *after*
        # the handler has exited: ``raise ... from None`` only sets
        # ``__suppress_context__``, which stops the traceback module from
        # printing the chain while leaving the original exception (and the
        # token in its ``args``/``object``) reachable through
        # ``error.__context__``.
        header_error: MemosProtocolError | None = None
        try:
            req = urllib.request.Request(  # noqa: S310
                url,
                method="GET",
                headers={
                    "Authorization": f"Bearer {self._config.api_token}",
                    "Accept": "application/json",
                },
            )
            with _OPENER.open(  # noqa: S310
                req, timeout=30
            ) as response:
                raw = response.read(_MAX_JSON_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise MemosProtocolError(f"http_{error.code}") from error
        except urllib.error.URLError as error:
            raise MemosConnectionError("url_error") from error
        except (OSError, socket.gaierror, TimeoutError) as error:
            raise MemosConnectionError("connect_failed") from error
        except ValueError:
            header_error = MemosProtocolError("invalid_request")
        if header_error is not None:
            raise header_error

        if len(raw) > _MAX_JSON_BYTES:
            raise MemosProtocolError("response_too_large")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise MemosProtocolError("json_decode_error") from error

    def _assert_egress_allowed(self) -> None:
        """Re-check the egress policy once per client, fail-closed.

        Configuration time tolerates a host that does not resolve; that
        leniency is a bypass on its own (SERVFAIL while the config is
        saved, ``127.0.0.1`` afterwards). A client is constructed for one
        sync, so checking on its first request keeps the guarantee without
        paying a resolver round trip per page.
        """
        if self._egress_checked:
            return
        assert_egress_allowed(self._config.base_url)
        self._egress_checked = True
