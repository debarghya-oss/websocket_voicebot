"""
LLM Chat Service.

Provides the streaming text generator that proxies chat completions from
an LM Studio / Ollama / OpenAI-compatible backend.
"""
import json
import logging
import time
import uuid
from typing import Dict, Generator, List, Optional

import requests

from voicebot.config import (
    LLM_CONTEXT_TURN_LIMIT,
    LLM_FAILED_PREFIX,
    LMSTUDIO_API_ENDPOINT,
    LMSTUDIO_MODEL,
    LMSTUDIO_SYSTEM_PROMPT,
    SSE_DATA_PREFIX,
    SSE_DONE_MARKER,
    STREAM_HEADERS,
    STREAM_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


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
    """Stream LLM chat completions from the configured backend."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] LLM: Initiating stream request.")

    selected_model = model if model else LMSTUDIO_MODEL
    logger.info(f"[{request_id}] LLM: Using model: {selected_model}")

    messages = [{"role": "system", "content": LMSTUDIO_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(LLM_CONTEXT_TURN_LIMIT * 2):])
    messages.append({"role": "user", "content": prompt})

    payload: dict = {
        "model": selected_model,
        "messages": messages,
        "temperature": llm_temperature,
        "top_p": llm_top_p,
        "max_tokens": llm_max_tokens if llm_max_tokens != -1 else None,
        "repeat_penalty": llm_repetition_penalty,
        "stream": True,
    }
    if llm_top_k is not None and llm_top_k > 0:
        payload["top_k"] = llm_top_k
    payload = {k: v for k, v in payload.items() if v is not None}

    logger.debug(f"[{request_id}] LLM payload: {json.dumps(payload, indent=2)}")
    resp = None
    t0 = time.time()

    try:
        resp = requests.post(
            LMSTUDIO_API_ENDPOINT, json=payload, headers=STREAM_HEADERS,
            stream=True, timeout=STREAM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        logger.info(f"[{request_id}] LLM stream connected in {time.time() - t0:.3f}s.")

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                dl = line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if not dl.startswith(SSE_DATA_PREFIX):
                continue
            js = dl[len(SSE_DATA_PREFIX):].strip()
            if js == SSE_DONE_MARKER:
                break
            if not js:
                continue
            try:
                data = json.loads(js)
            except json.JSONDecodeError:
                continue

            delta_content = None
            if "choices" in data and data["choices"]:
                ch = data["choices"][0]
                if "delta" in ch and ch["delta"] is not None:
                    delta_content = ch["delta"].get("content")
                fr = ch.get("finish_reason")
                if fr and fr not in ("stop", "length", None):
                    logger.warning(f"[{request_id}] LLM non-standard finish_reason: {fr}")

            if delta_content is not None:
                yield delta_content
            elif "error" in data:
                msg = data.get("error", {}).get("message", "Unknown LLM error")
                err = f"{LLM_FAILED_PREFIX} (Stream Error: {msg})"
                logger.error(f"[{request_id}] {err}")
                yield err
                return

            if "choices" in data and data["choices"] and data["choices"][0].get("finish_reason"):
                break

        logger.info(f"[{request_id}] LLM stream done in {time.time() - t0:.3f}s.")

    except requests.exceptions.Timeout:
        logger.error(f"[{request_id}] LLM timed out.", exc_info=True)
        yield f"{LLM_FAILED_PREFIX} (LLM stream request timed out)"
    except requests.exceptions.RequestException as e:
        logger.exception(f"[{request_id}] LLM request failed: {e}")
        err = f"{LLM_FAILED_PREFIX} (Error connecting to LLM server)"
        if hasattr(e, "response") and e.response is not None:
            try:
                ej = e.response.json()
                detail = ej.get("error", {}).get("message") or ej.get("detail", e.response.text)
                err = f"{LLM_FAILED_PREFIX} (LLM Server Error: {str(detail)[:200]})"
            except json.JSONDecodeError:
                err = f"{LLM_FAILED_PREFIX} (LLM Server Error: {e.response.status_code} - {e.response.text[:200]})"
        yield err
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected LLM error: {e}")
        yield f"{LLM_FAILED_PREFIX} (Unexpected Error: {e})"
    finally:
        if resp:
            resp.close()
