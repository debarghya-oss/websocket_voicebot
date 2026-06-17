import logging
from typing import List, Optional, Dict, Any

from .embedder import get_embedder, embed_query_text
from .vectorstore import get_collection

logger = logging.getLogger(__name__)

SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"ef": 200},
}

OUTPUT_FIELDS = ["text", "document", "source", "topic"]


def _get_field(entity, field: str) -> str:
    """Compatible getter for Milvus 2.4 and 2.5 SDK hit.entity."""
    try:
        val = entity[field]
        return val if val is not None else ""
    except (TypeError, KeyError):
        pass
    try:
        return getattr(entity, field, "") or ""
    except Exception:
        return ""


def retrieve_chunks(
    question: str,
    top_k: int = 5,
    topic_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Embed the question and perform Top-K semantic search in Milvus.
    Optionally pre-filter by topic metadata.
    """
    query_embedding = embed_query_text(question)

    collection = get_collection()

    expr = None
    if topic_filter:
        expr = f'topic == "{topic_filter}"'
        logger.info(f"Applying metadata filter: {expr}")

    # Fetch 2x top_k to account for duplicates after dedup
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=SEARCH_PARAMS,
        limit=top_k * 2,
        expr=expr,
        output_fields=OUTPUT_FIELDS,
    )

    chunks = []
    seen_texts = set()
    for hit in results[0]:
        text = _get_field(hit.entity, "text")
        # Deduplicate by text content
        if text in seen_texts:
            continue
        seen_texts.add(text)
        chunks.append({
            "text": text,
            "document": _get_field(hit.entity, "document"),
            "source": _get_field(hit.entity, "source"),
            "topic": _get_field(hit.entity, "topic"),
            "score": hit.score,
        })
        if len(chunks) >= top_k:
            break

    logger.info(f"Retrieved {len(chunks)} unique chunks for query: '{question[:60]}...'")
    return chunks