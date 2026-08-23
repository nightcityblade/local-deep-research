"""Outline snapshot reconciliation.

Known limitation: ``provider_revision`` is derived from the server's own
change markers (``updatedAt`` and the owning collection id) and never hashes
the item's content. The listing endpoint does not return content, so hashing
it would cost one extra request per item on every sync. A server that omits
those markers therefore reports a constant revision for every item and edits
are not re-synced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ...models import RemoteSnapshot, RemoteSnapshotItem
from .client import OutlineClient
from .errors import OutlineProtocolError

_PAGE_SIZE = 100

# Safety valve against a server that keeps returning a full page forever;
# exposed as a module constant for tests.
_MAX_PAGINATED_DOCUMENTS = 500_000


@dataclass(frozen=True, slots=True)
class OutlineDocument:
    """One Outline document with content and metadata."""

    document_id: str
    title: str
    text: str
    url: str
    collection_id: str
    collection_name: str
    created_at: str
    updated_at: str
    tags: tuple[str, ...]


def fetch_outline_snapshot(client: OutlineClient) -> RemoteSnapshot:
    """Build a stable :class:`RemoteSnapshot` from all Outline documents."""
    config = client.config

    all_documents: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = client.list_documents(
            offset=offset,
            limit=_PAGE_SIZE,
            collection_id=config.collection_id,
        )
        if not batch:
            break
        all_documents.extend(batch)
        if len(batch) < _PAGE_SIZE:
            break
        offset += len(batch)
        if len(all_documents) > _MAX_PAGINATED_DOCUMENTS:
            raise OutlineProtocolError("pagination_not_terminating")

    if not all_documents:
        raise OutlineProtocolError("no_documents")

    items: list[RemoteSnapshotItem] = []
    seen: set[str] = set()
    for document in all_documents:
        if not isinstance(document, dict):
            raise OutlineProtocolError("document_entry_not_object")
        document_id = str(document.get("id", ""))
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        updated_at = str(document.get("updatedAt", ""))
        collection_id = str(document.get("collectionId", ""))
        provider_revision = f"{updated_at}_{collection_id}"
        revision = sha256(
            b"outline-effective-v1\0" + provider_revision.encode("utf-8")
        ).hexdigest()
        items.append(
            RemoteSnapshotItem(
                external_id=document_id,
                provider_revision=provider_revision,
                revision=revision,
            )
        )
    # ``RemoteSnapshot`` validates ``item_ids == sorted(item_ids)``, so the
    # lexicographic order is both the emitted order and the selection order.
    # Outline document ids are UUIDs, so unlike Linkwarden there is no
    # numeric order to prefer.
    items.sort(key=lambda si: si.external_id)
    if not items:
        raise OutlineProtocolError("no_valid_documents")
    if config.max_documents > 0:
        # Select only after fetching the whole listing, and never bound the
        # fetch by the cap. Items missing from a snapshot are marked
        # ``pending_removal``, so any selection that depends on the
        # server's listing order - including an early break once the cap is
        # reached - makes the retained set flap between syncs whenever that
        # order changes. ``_MAX_PAGINATED_DOCUMENTS``, not the cap, is what
        # bounds a non-terminating server.
        items = items[: config.max_documents]

    triples = [
        [si.external_id, si.provider_revision, si.revision] for si in items
    ]
    signature = sha256(
        json.dumps(
            triples,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return RemoteSnapshot(
        items=tuple(items),
        expected_count=len(items),
        signature=signature,
    )


def fetch_outline_document(
    client: OutlineClient, item: RemoteSnapshotItem
) -> OutlineDocument:
    """Fetch a single document's full content and resolved metadata."""
    data = client.get_document(item.external_id)

    title = str(data.get("title") or "")
    text = str(data.get("text") or "")
    document_id = str(data.get("id", item.external_id))
    url = str(data.get("url") or "")
    collection_id = str(data.get("collectionId") or "")
    collection_name = (
        client.get_collection_name(collection_id) if collection_id else ""
    )

    tags: list[str] = []
    raw_tags = data.get("tags")
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            name = tag.get("name") if isinstance(tag, dict) else None
            if isinstance(name, str) and name.strip():
                tags.append(name.strip())

    return OutlineDocument(
        document_id=document_id,
        title=title,
        text=text,
        url=_absolute_url(client.config.base_url, url, document_id),
        collection_id=collection_id,
        collection_name=collection_name,
        created_at=str(data.get("createdAt", "")),
        updated_at=str(data.get("updatedAt", "")),
        tags=tuple(tags),
    )


def _absolute_url(base_url: str, url: str, document_id: str) -> str:
    """Resolve an Outline document URL against the instance base URL.

    Unlike the sibling providers, Outline returns ``url`` as an
    instance-relative path (``/doc/title-slug``), so a strict
    ``http(s)://`` allowlist would discard every real value. A path
    beginning ``//`` or ``/\\`` is not one of those: a browser reads it as
    protocol-relative - an authority, not a path - so it is rejected rather
    than concatenated onto the base. Everything else that is neither
    ``http(s)://`` nor a plain absolute path falls back to an inert
    provider URI.
    """
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/") and url[1:2] not in ("/", "\\"):
        return f"{base_url.rstrip('/')}{url}"
    return f"outline://document/{document_id}"
