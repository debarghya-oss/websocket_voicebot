import os
import logging
import requests
from typing import List, Dict, Any, Tuple

from .modelfile_store import get_system_prompt

logger = logging.getLogger(__name__)

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://172.17.0.1:1234")
LM_STUDIO_MODEL    = os.getenv("LM_STUDIO_MODEL",    "gemma-3n-e4b-it")
MAX_CONTEXT_CHARS  = int(os.getenv("MAX_CONTEXT_CHARS", 12_000))


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict]]:
    """
    Build the user prompt and return it alongside a deduplicated source list.
    Improvement 1: truncates assembled context to MAX_CONTEXT_CHARS.
    Improvement 2: returns source metadata separately so the caller can attach
                   citations to the response without re-parsing the prompt.
    """
    parts = []
    sources = []

    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | doc: {chunk['document']} | topic: {chunk['topic']}]\n"
            f"{chunk['text']}"
        )
        sources.append({
            "index":    i,
            "document": chunk["document"],
            "topic":    chunk["topic"],
            "score":    round(chunk.get("score", 0.0), 4),
        })

    context = "\n\n---\n\n".join(parts)

    # Improvement 1: hard truncation with a visible marker so the LLM knows
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"
        logger.warning(
            "Context truncated to %d chars (%d chunks, original ~%d chars)",
            MAX_CONTEXT_CHARS, len(chunks), len("\n\n---\n\n".join(parts)),
        )

    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return prompt, sources


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call LM Studio's OpenAI-compatible /v1/chat/completions endpoint.
    System prompt is pulled live from modelfile_store so it reflects
    whatever Modelfile the user has uploaded at this moment.
    """
    prompt, sources = build_prompt(question, chunks)

    # Live system prompt — changes instantly when user uploads a new Modelfile
    system_prompt = get_system_prompt()
    logger.info("Generating answer via LM Studio (%s)...", LM_STUDIO_MODEL)

    payload = {
        "model":       LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  512,
        "stream":      False,
    }

    try:
        resp = requests.post(
            f"{LM_STUDIO_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data   = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        logger.info("LM Studio response received (%d chars)", len(answer))
        return {"answer": answer, "sources": sources}

    # Improvement 3: split timeout from generic exceptions for clearer recovery
    except requests.exceptions.Timeout:
        logger.error("LM Studio timed out after 120s — model may be overloaded")
        return {
            "answer":  "[LM Studio timed out. The model is busy — please retry in a moment.]",
            "sources": sources,
        }
    except requests.exceptions.ConnectionError:
        logger.error("Cannot reach LM Studio at %s — is it running?", LM_STUDIO_BASE_URL)
        return {
            "answer":  "[LM Studio unreachable — is it running on the host at port 1234?]",
            "sources": sources,
        }
    except requests.exceptions.HTTPError as e:
        logger.error("LM Studio HTTP error: %s", e)
        return {
            "answer":  f"[LM Studio returned an error: {e}]",
            "sources": sources,
        }
    except Exception as e:
        logger.error("LM Studio generation error: %s", e)
        return {
            "answer":  f"[Unexpected LLM error: {e}]",
            "sources": sources,
        }