"""MCP tools: ingest web search results into Chroma and query per-story RAG."""

from __future__ import annotations

import json
from typing import Any

from template_mcp_server.src.rag.chroma_store import (
    format_query_context,
    ingest_search_web_payload,
    query_story,
)
from template_mcp_server.src.settings import settings
from template_mcp_server.utils.pylogger import get_python_logger

logger = get_python_logger()


def research_rag_ingest_search_results(
    story_id: str,
    search_web_result_json: str,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """Index Tavily/search_web JSON into the vector store for later RAG queries.

    Pass the **full tool return value** from ``search_web`` as a JSON string.
    Typical flow: call ``search_web``, then call this with the same ``story_id``
    the orchestrator uses for the whole pipeline (e.g. conversation thread id).

    Args:
        story_id: Isolates chunks for one story (use a stable id across agents).
        search_web_result_json: JSON string with ``results`` list (title, url, snippet).
        replace_existing: If True (default), clears prior chunks for this story then adds
            (use with **one merged** payload of all searches). If False, **appends** without
            deleting — use for extra ``search_web`` rounds so nothing already saved is lost.

    Returns:
        Status dict with chunk counts or error details.
    """
    if not settings.RAG_ENABLED:
        return {"status": "error", "message": "RAG is disabled (RAG_ENABLED=false)"}

    try:
        payload = json.loads(search_web_result_json)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON: {e}"}

    if not isinstance(payload, dict):
        return {"status": "error", "message": "Root JSON must be an object"}

    try:
        return ingest_search_web_payload(
            story_id.strip(),
            payload,
            replace_existing=replace_existing,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("research_rag_ingest_search_results failed")
        return {"status": "error", "message": str(e)}


def research_rag_query(
    story_id: str,
    query: str,
    top_k: int = 5,
) -> str:
    """Retrieve top-k research chunks for a story (vector similarity).

    Use for gap analysis, fact-check cross-reference, or editor context.
    Returns a plain-text block suitable to paste into reasoning (titles, URLs, excerpts).

    Args:
        story_id: Same id used when ingesting.
        query: Natural-language question or keywords.
        top_k: Number of chunks (clamped by server config).

    Returns:
        Formatted context string or JSON error message.
    """
    if not settings.RAG_ENABLED:
        return json.dumps({"status": "error", "message": "RAG is disabled (RAG_ENABLED=false)"})

    try:
        raw = query_story(story_id.strip(), query, top_k)
        return format_query_context(raw)
    except Exception as e:  # noqa: BLE001
        logger.exception("research_rag_query failed")
        return json.dumps({"status": "error", "message": str(e)})


def research_rag_clear_story(story_id: str) -> dict[str, Any]:
    """Remove all indexed chunks for a story (e.g. before re-ingest). Usually automatic on ingest."""
    if not settings.RAG_ENABLED:
        return {"status": "error", "message": "RAG is disabled (RAG_ENABLED=false)"}
    try:
        from template_mcp_server.src.rag.chroma_store import clear_story_chunks

        n = clear_story_chunks(story_id.strip())
        return {"status": "success", "story_id": story_id, "chunks_deleted": n}
    except Exception as e:  # noqa: BLE001
        logger.exception("research_rag_clear_story failed")
        return {"status": "error", "message": str(e)}
