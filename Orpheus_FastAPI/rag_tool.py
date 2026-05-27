# rag_tool.py
"""
RAG Tool — optional retrieval layer for the conversational pipeline.

Design:
  1. run_intent_check(prompt)  →  True if the query needs external knowledge
  2. fetch_rag_context(prompt) →  formatted context string from the RAG service

Intent is decided by a cheap single-call to the LLM that returns JSON {needs_rag: bool}.
If the RAG service is unreachable, the pipeline degrades gracefully (no retrieval, no crash).
"""

import json
import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8080")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "6"))          # seconds; keep low for voice latency

# The LLM endpoint used for the intent check (same server as the main LLM)
LLM_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")
INTENT_MODEL = os.getenv("LMSTUDIO_MODEL", "gemma-3n-e4b-it")
INTENT_TIMEOUT = int(os.getenv("INTENT_CHECK_TIMEOUT", "5"))  # seconds

# Minimum similarity score below which chunks are discarded
# COSINE similarity — 0.75+ is a strong topical match; below that is noise
MIN_CHUNK_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.65"))

# ── Intent classification ─────────────────────────────────────────────────────

_INTENT_SYSTEM = """\
You are a binary routing classifier. Your only job is to output JSON.

Output {"needs_rag": true} ONLY if the message is asking for specific private information that cannot be answered from general knowledge — for example: a named person's medical record, a specific client's account details, a company's internal policy document, or a product's internal specification sheet.

Output {"needs_rag": false} for EVERYTHING else — including:
- greetings and small talk ("good morning", "how are you", "just chatting")
- general knowledge questions ("what is diabetes", "how does Python work")
- opinions, advice, or conversation
- follow-up messages that don't request new private facts

Examples:
User: "good morning" → {"needs_rag": false}
User: "just here to chat" → {"needs_rag": false}
User: "what is the weather like" → {"needs_rag": false}
User: "what medication is patient John Doe on" → {"needs_rag": true}
User: "look up policy number 4821 for me" → {"needs_rag": true}
User: "tell me about yourself" → {"needs_rag": false}

Reply with ONLY the JSON object. No explanation. No markdown. No other text.\
"""


def needs_rag(prompt: str) -> bool:
    """
    Ask the LLM whether this prompt requires a RAG lookup.
    Returns False on any error so the pipeline never blocks.
    """
    payload = {
        "model": INTENT_MODEL,
        "messages": [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 20,
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=INTENT_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if the model wraps JSON anyway
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        decision = json.loads(raw)
        result = bool(decision.get("needs_rag", False))
        logger.info(f"RAG intent check → needs_rag={result}  (prompt: '{prompt[:80]}')")
        return result
    except Exception as e:
        logger.warning(f"RAG intent check failed ({e}); defaulting to no-RAG.")
        return False


# ── RAG retrieval ─────────────────────────────────────────────────────────────

def fetch_rag_context(prompt: str, topic_filter: Optional[str] = None):
    """
    Call /retrieve on the RAG service and return (context_block, sources_list).
    context_block is a formatted string for the system prompt.
    sources_list is a list of dicts with doc/topic/score for display.
    Returns (None, []) if retrieval fails or no chunks pass the threshold.
    """
    payload: dict = {"question": prompt, "top_k": RAG_TOP_K}
    if topic_filter:
        payload["topic_filter"] = topic_filter

    try:
        resp = requests.post(
            f"{RAG_BASE_URL}/retrieve",
            json=payload,
            timeout=RAG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning("RAG service unreachable at %s — skipping retrieval.", RAG_BASE_URL)
        return None, []
    except requests.exceptions.Timeout:
        logger.warning("RAG service timed out after %ds — skipping retrieval.", RAG_TIMEOUT)
        return None, []
    except Exception as e:
        logger.warning("RAG retrieval error: %s — skipping.", e)
        return None, []

    chunks = data.get("chunks", [])
    # Filter low-confidence chunks
    chunks = [c for c in chunks if c.get("score", 0.0) >= MIN_CHUNK_SCORE]

    if not chunks:
        logger.info("RAG returned no chunks above score threshold %.2f.", MIN_CHUNK_SCORE)
        return None, []

    lines = ["--- RETRIEVED KNOWLEDGE BASE CONTEXT (use this to answer if relevant) ---"]
    for i, c in enumerate(chunks, 1):
        doc   = c.get("document", "unknown")
        topic = c.get("topic",    "general")
        score = round(c.get("score", 0.0), 3)
        text  = c.get("text", "").strip()
        lines.append(f"[{i}] doc={doc} | topic={topic} | score={score}\n{text}")
    lines.append("--- END OF RETRIEVED CONTEXT ---")

    context_block = "\n\n".join(lines)
    sources = [
        {
            "document": c.get("document", "unknown"),
            "topic":    c.get("topic", "general"),
            "score":    round(c.get("score", 0.0), 4),
        }
        for c in chunks
    ]
    logger.info(
        "RAG context built: %d chunk(s), %d chars total.",
        len(chunks), len(context_block),
    )
    return context_block, sources


# ── Combined helper ───────────────────────────────────────────────────────────

def maybe_get_rag_context(
    prompt: str,
    topic_filter: Optional[str] = None,
    force: bool = False,
) -> Optional[str]:
    """
    Full pipeline: classify intent → retrieve if needed → return context or None.

    force=True  — skip intent check, always retrieve. Score threshold still applies,
                  so greetings with no matching chunks return None cleanly.
    force=False — run intent classifier first (default, for when no docs are uploaded).
    """
    if not force and not needs_rag(prompt):
        logger.info("RAG intent check → needs_rag=False, skipping retrieval.")
        return None, []
    if force:
        logger.info("RAG force-retrieve mode (docs are uploaded) — skipping intent check.")
    return fetch_rag_context(prompt, topic_filter=topic_filter)