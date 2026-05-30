"""ChromaDB-backed RAG store for per-story research chunks (MCP process-local)."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from template_mcp_server.src.settings import settings
from template_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()

_COLLECTION = "journalism_research_rag"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def _client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    path = settings.RAG_CHROMA_PERSIST_DIR
    return chromadb.PersistentClient(
        path=path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    """Return Chroma collection (creates client + collection on first use)."""
    client = _client()
    return client.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def clear_story_chunks(story_id: str) -> int:
    """Remove all chunks for story_id. Returns number of chunks deleted."""
    col = get_collection()
    existing = col.get(where={"story_id": story_id}, include=[])
    ids = existing.get("ids") or []
    if ids:
        col.delete(ids=ids)
    return len(ids)


def ingest_search_web_payload(
    story_id: str,
    payload: dict[str, Any],
    *,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """Parse search_web-style JSON, chunk snippets, embed into Chroma.

    If replace_existing is True (default), removes prior chunks for this story_id
    before adding — use when ingesting one merged payload of all searches.

    If False, **appends** without clearing — use for follow-up search rounds so
    earlier results are not dropped.
    """
    if not story_id or not str(story_id).strip():
        return {"status": "error", "message": "story_id is required"}

    results = payload.get("results")
    if not isinstance(results, list):
        return {"status": "error", "message": "payload must contain 'results' list"}

    chunk_size = settings.RAG_CHUNK_CHARS
    overlap = settings.RAG_CHUNK_OVERLAP_CHARS

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    # Unique per ingest batch so append mode never collides with existing ids
    batch_token = uuid.uuid4().hex[:16]

    for i, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")[:500]
        url = str(item.get("url") or "")
        snippet = str(item.get("snippet") or item.get("content") or "")
        body = f"{title}\n{url}\n{snippet}".strip()
        if not body:
            continue
        for j, chunk in enumerate(_chunk_text(body, chunk_size, overlap)):
            uid = hashlib.sha256(
                f"{story_id}|{url}|{i}|{j}|{chunk[:80]}|{batch_token}".encode()
            ).hexdigest()[:32]
            documents.append(chunk)
            metadatas.append(
                {
                    "story_id": story_id,
                    "url": url[:2000],
                    "title": title[:500],
                    "result_index": str(i),
                    "chunk_index": str(j),
                }
            )
            ids.append(uid)

    if not documents:
        return {"status": "error", "message": "No text extracted from results"}

    removed = clear_story_chunks(story_id) if replace_existing else 0
    col = get_collection()
    col.add(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(
        "rag_ingest",
        extra={
            "story_id": story_id,
            "chunks_added": len(documents),
            "chunks_removed_prior": removed,
            "replace_existing": replace_existing,
        },
    )

    return {
        "status": "success",
        "story_id": story_id,
        "chunks_indexed": len(documents),
        "previous_chunks_cleared": removed,
        "replace_existing": replace_existing,
    }


def query_story(story_id: str, query: str, top_k: int) -> dict[str, Any]:
    """Semantic search over chunks for one story."""
    if not story_id or not str(story_id).strip():
        return {"status": "error", "message": "story_id is required"}
    if not query or not str(query).strip():
        return {"status": "error", "message": "query is required"}

    top_k = max(1, min(int(top_k), settings.RAG_MAX_TOP_K))
    col = get_collection()

    res = col.query(
        query_texts=[query.strip()],
        n_results=top_k,
        where={"story_id": story_id},
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for d, m, dist in zip(docs, metas, dists, strict=False):
        hits.append(
            {
                "text": d,
                "title": (m or {}).get("title", ""),
                "url": (m or {}).get("url", ""),
                "distance": float(dist) if dist is not None else None,
            }
        )

    return {
        "status": "success",
        "story_id": story_id,
        "query": query.strip(),
        "top_k": top_k,
        "hits": hits,
    }


def format_query_context(result: dict[str, Any]) -> str:
    """Flatten query hits into a single context string for LLM consumption."""
    if result.get("status") != "success":
        return json.dumps(result, indent=2)
    lines: list[str] = []
    for i, h in enumerate(result.get("hits") or [], start=1):
        title = h.get("title") or ""
        url = h.get("url") or ""
        text = h.get("text") or ""
        lines.append(f"[{i}] {title}\nURL: {url}\n{text}\n")
    if not lines:
        return "No matching chunks in RAG for this story_id and query."
    return "\n---\n".join(lines)
