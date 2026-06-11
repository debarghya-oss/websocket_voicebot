# llm_router.py
import logging
import os
import json
import uuid
import time
from typing import Any, Dict, Generator, List, Optional

import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── LLM connection ────────────────────────────────────────────
# Change SERVER_BASE_URL in .env to switch backends — no code changes needed.
#   LM Studio (local):  SERVER_BASE_URL=http://127.0.0.1:1234
#   vLLM (local):       SERVER_BASE_URL=http://127.0.0.1:8001
#   Remote vLLM:        SERVER_BASE_URL=http://192.168.x.x:8001
SERVER_BASE_URL       = os.getenv("SERVER_BASE_URL",    "http://127.0.0.1:1234")
LMSTUDIO_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/chat/completions"

# Model identifier — must match what the backend has loaded.
# LM Studio:  use the model string shown in the UI  (e.g. "gemma-3n-e4b-it")
# vLLM:       use the HuggingFace repo id           (e.g. "google/gemma-3-4b-it")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "gemma-3n-e4b-it")

LMSTUDIO_SYSTEM_PROMPT = os.getenv(
    "LMSTUDIO_SYSTEM_PROMPT",
    "You are a concise Bengali-speaking AI assistant. "
    "Do not generate any text in english or any translation",
)

# ── RAG threshold-gated routing ───────────────────────────────
# RAG_ENABLED=true  activates RAG routing.
# RAG_SCORE_THRESHOLD is the gate:
#   - Retrieve top-K chunks and score them via HNSW cosine search.
#   - Find the BEST (highest) score among all retrieved chunks.
#   - If best score >= RAG_SCORE_THRESHOLD → confident match →
#     inject all chunks that also meet the threshold into the prompt.
#   - If best score <  RAG_SCORE_THRESHOLD → no confident match →
#     skip RAG entirely, LLM answers from its own knowledge.
#
# 0.6 is the right gate for BGE-small on a voice assistant:
#   < 0.6  weak/unrelated match  → answer directly, no hallucination risk
#   ≥ 0.6  genuine semantic hit  → ground the answer in your documents
#
# Tune after testing with your actual knowledge base:
#   false negatives (relevant questions skipped) → lower to 0.55
#   noisy context injected                       → raise to 0.65
RAG_ENABLED         = os.getenv("RAG_ENABLED",         "false").lower() == "true"
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.6"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K",           "3"))

DEFAULT_LMSTUDIO_MAX_TOKENS  = 256
DEFAULT_LMSTUDIO_TEMP        = 0.3
DEFAULT_LMSTUDIO_TOP_P       = 0.85
DEFAULT_LMSTUDIO_TOP_K       = int(os.getenv("DEFAULT_LMSTUDIO_TOP_K", "20"))
DEFAULT_LMSTUDIO_REP_PENALTY = 1.2
LLM_CONTEXT_TURN_LIMIT       = 3

STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS         = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX        = "data:"
SSE_DONE_MARKER        = "[DONE]"
LLM_FAILED_PREFIX      = "[Error"


# ── RAG threshold-gated retrieval ────────────────────────────

def _retrieve_context(prompt: str) -> str:
    """
    Threshold-gated RAG routing.

    1. Retrieve top-K chunks from Milvus.
    2. Find the best (highest) cosine score among them.
    3. If best score < RAG_SCORE_THRESHOLD → return "" (skip RAG, no context).
    4. If best score >= RAG_SCORE_THRESHOLD → inject all chunks that also
       meet the threshold. Chunks below the threshold but in the top-K
       are discarded — only the confident matches go into the prompt.

    This means RAG is never called unless there is at least one genuinely
    relevant document. Weak retrievals never pollute the LLM context.
    """
    if not RAG_ENABLED:
        return ""

    try:
        from rag.retriever import retrieve_chunks
        chunks = retrieve_chunks(question=prompt, top_k=RAG_TOP_K)

        if not chunks:
            logger.info("RAG: no chunks retrieved")
            return ""

        # Gate: check best score first
        best_score = max(c.get("score", 0.0) for c in chunks)

        if best_score < RAG_SCORE_THRESHOLD:
            logger.info(
                "RAG: best score %.3f < threshold %.2f — skipping RAG, LLM answers directly",
                best_score, RAG_SCORE_THRESHOLD,
            )
            return ""

        # Best score qualifies — keep only chunks that also meet the threshold
        qualified = [c for c in chunks if c.get("score", 0.0) >= RAG_SCORE_THRESHOLD]

        parts = []
        for i, c in enumerate(qualified, 1):
            parts.append(
                f"[Source {i} | {c['document']} | {c['topic']}]\n{c['text']}"
            )
        context = "\n\n---\n\n".join(parts)

        logger.info(
            "RAG: best score %.3f >= threshold %.2f — injecting %d/%d chunks",
            best_score, RAG_SCORE_THRESHOLD, len(qualified), len(chunks),
        )
        return f"Use the following context to answer the question:\n\n{context}\n\n---\n\n"

    except Exception as e:
        logger.warning("RAG retrieval failed (continuing without context): %s", e)
        return ""


# ── Pydantic model ────────────────────────────────────────────

class LLMChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    model: Optional[str] = None
    temperature: float = DEFAULT_LMSTUDIO_TEMP
    top_p: float = DEFAULT_LMSTUDIO_TOP_P
    max_tokens: int = DEFAULT_LMSTUDIO_MAX_TOKENS
    repetition_penalty: float = DEFAULT_LMSTUDIO_REP_PENALTY
    top_k: Optional[int] = DEFAULT_LMSTUDIO_TOP_K


# ── Stream generator ──────────────────────────────────────────

def generate_llm_text_stream(
    prompt: str,
    history: List[Dict[str, str]],
    llm_temperature: float,
    llm_top_p: float,
    llm_max_tokens: int,
    llm_repetition_penalty: float,
    llm_top_k: Optional[int] = None,
    model: Optional[str] = None,
) -> Generator[str, None, None]:
    request_id     = str(uuid.uuid4())
    selected_model = model if model else LMSTUDIO_MODEL
    logger.info("[%s] LLM stream | model=%s | rag=%s", request_id, selected_model, RAG_ENABLED)

    # Threshold-gated RAG — returns "" if no chunk scores >= 0.6
    context_prefix = _retrieve_context(prompt)

    # System prompt from modelfile_store (reflects uploaded Modelfile live)
    try:
        from rag.modelfile_store import get_system_prompt
        system_content = get_system_prompt()
    except Exception:
        system_content = LMSTUDIO_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_content}]

    if history:
        messages.extend(history[-(LLM_CONTEXT_TURN_LIMIT * 2):])

    # context_prefix is either a grounded context block or "" (no RAG)
    messages.append({"role": "user", "content": context_prefix + prompt})

    payload = {
        "model":          selected_model,
        "messages":       messages,
        "temperature":    llm_temperature,
        "top_p":          llm_top_p,
        "max_tokens":     llm_max_tokens if llm_max_tokens != -1 else None,
        "repeat_penalty": llm_repetition_penalty,
        "stream":         True,
    }
    if llm_top_k is not None and llm_top_k > 0:
        payload["top_k"] = llm_top_k

    payload = {k: v for k, v in payload.items() if v is not None}

    response_obj      = None
    stream_start_time = time.time()

    try:
        response_obj = requests.post(
            LMSTUDIO_API_ENDPOINT,
            json=payload,
            headers=STREAM_HEADERS,
            stream=True,
            timeout=STREAM_TIMEOUT_SECONDS,
        )
        response_obj.raise_for_status()
        logger.info("[%s] LLM stream connected (%.3fs)", request_id, time.time() - stream_start_time)
        error_in_stream = False

        for line in response_obj.iter_lines():
            if error_in_stream:
                break
            if not line:
                continue

            try:
                decoded = line.decode("utf-8")
            except UnicodeDecodeError:
                continue

            if decoded.startswith(SSE_DATA_PREFIX):
                json_str = decoded[len(SSE_DATA_PREFIX):].strip()
                if json_str == SSE_DONE_MARKER:
                    break
                if not json_str:
                    continue
                try:
                    data          = json.loads(json_str)
                    delta_content = None

                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "delta" in choice and choice["delta"] is not None:
                            delta_content = choice["delta"].get("content")

                    if delta_content is not None:
                        yield delta_content
                    elif "error" in data:
                        msg = data.get("error", {}).get("message", "Unknown stream error")
                        yield f"{LLM_FAILED_PREFIX} (Stream Error: {msg})"
                        error_in_stream = True
                        break

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    yield f"{LLM_FAILED_PREFIX} (Processing Error: {e})"
                    error_in_stream = True
                    break

        logger.info("[%s] LLM stream done (%.3fs)", request_id, time.time() - stream_start_time)

    except requests.exceptions.Timeout:
        yield f"{LLM_FAILED_PREFIX} (LLM stream timed out after {STREAM_TIMEOUT_SECONDS}s)"
    except requests.exceptions.RequestException as e:
        err = f"{LLM_FAILED_PREFIX} (Error connecting to LLM server)"
        if hasattr(e, "response") and e.response is not None:
            try:
                detail = e.response.json().get("error", {}).get("message", e.response.text)
                err = f"{LLM_FAILED_PREFIX} (LLM Server Error: {str(detail)[:200]})"
            except Exception:
                err = f"{LLM_FAILED_PREFIX} (LLM Server Error: {e.response.status_code})"
        yield err
    except Exception as e:
        yield f"{LLM_FAILED_PREFIX} (Unexpected Error: {e})"
    finally:
        if response_obj:
            response_obj.close()


# ── Router ────────────────────────────────────────────────────

router = APIRouter()


@router.post("/chat/stream", summary="Stream LLM Chat Completions", tags=["LLM"])
async def llm_chat_stream_endpoint_router(request_data: LLMChatRequest):
    logger.info("LLM Router: POST /chat/stream")
    generator = generate_llm_text_stream(
        prompt=request_data.prompt,
        history=request_data.history,
        llm_temperature=request_data.temperature,
        llm_top_p=request_data.top_p,
        llm_max_tokens=request_data.max_tokens,
        llm_repetition_penalty=request_data.repetition_penalty,
        llm_top_k=request_data.top_k,
        model=request_data.model,
    )
    return StreamingResponse(generator, media_type="text/event-stream")