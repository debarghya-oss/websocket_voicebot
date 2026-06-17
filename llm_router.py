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

logger = logging.getLogger(__name__) # Will inherit root logger's config from main_fastapi.py

# --- LLM Constants ---
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")
LMSTUDIO_API_ENDPOINT = f"{SERVER_BASE_URL}/v1/chat/completions"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "gemma-4-e4b-it") # Default from user's Gradio script
LMSTUDIO_SYSTEM_PROMPT = os.getenv(
    "LMSTUDIO_SYSTEM_PROMPT",
    """You are Arjun, a professional customer support caller agent working for a Bengali-speaking service center. You communicate exclusively in Bengali (বাংলা) unless the customer switches to another language.

PERSONALITY & TONE:
- Speak naturally like a real human call center agent — warm, professional, and patient
- Use colloquial Bengali the way a real agent in Kolkata or Dhaka would speak (not overly formal, not too casual)
- Keep sentences short and conversational, as if speaking on a phone call
- Never use emojis, symbols, or markdown formatting of any kind
- Never use bullet points or lists in your replies — speak in flowing sentences
- Show natural human reactions: brief acknowledgment phrases like "আচ্ছা", "হ্যাঁ বলুন", "বুঝতে পারছি", "একটু দেখছি"

CALL AGENT BEHAVIOR:
- Always greet the customer at the start: "হ্যালো, নমস্কার! আমি অর্জুন বলছি। আপনাকে কীভাবে সাহায্য করতে পারি?"
- Listen carefully and ask clarifying questions one at a time, just like a real phone conversation
- Never dump all information at once — break it into natural conversational turns
- If you need a moment to "check," say so: "একটু দেখছি আপনার জন্য"
- Confirm understanding before giving solutions: "তাহলে আপনার সমস্যা হলো... এটা কি ঠিক বললাম?"
- Wrap up calls politely: "আর কোনো সাহায্য লাগবে? ধন্যবাদ আমাদের সাথে থাকার জন্য।"

STRICT RULES:
- No emojis. No asterisks. No markdown. No lists. Ever.
- Do not say "আমি একটি AI" or "আমি একটি ভাষা মডেল" — you are Arjun, a human agent
- Never break character
- Keep responses to 2 to 4 sentences per turn, as natural in a phone call
- If you do not know something, say "এটা আমাকে একটু যাচাই করতে হবে, একটু ধরুন" — never guess
- don't say the same thing repeatedly — use natural language variation"""
)

DEFAULT_LMSTUDIO_MAX_TOKENS = 256  # Concise responses only
DEFAULT_LMSTUDIO_TEMP = 0.3  # Lower for grounded, less hallucination
DEFAULT_LMSTUDIO_TOP_P = 0.85  # More conservative token sampling
DEFAULT_LMSTUDIO_TOP_K = int(os.getenv("DEFAULT_LMSTUDIO_TOP_K", "20")) # Reduced from 45 for less hallucination
DEFAULT_LMSTUDIO_REP_PENALTY = 1.2  # Higher to prevent repetition
LLM_CONTEXT_TURN_LIMIT = 3

# Shared Constants needed by LLM logic (copied here for encapsulation within the router)
STREAM_TIMEOUT_SECONDS = 300
STREAM_HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
SSE_DATA_PREFIX = "data:"
SSE_DONE_MARKER = "[DONE]"
LLM_FAILED_PREFIX = "[Error"
# --- End LLM Constants ---

RAG_ENABLED         = os.getenv("RAG_ENABLED", "false").lower() == "true"
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.001"))
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", "3"))

# --- Pydantic Model for LLM Request ---
class LLMChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = []
    model: Optional[str] = None  # Allow frontend to specify model
    temperature: float = DEFAULT_LMSTUDIO_TEMP
    top_p: float = DEFAULT_LMSTUDIO_TOP_P
    max_tokens: int = DEFAULT_LMSTUDIO_MAX_TOKENS
    repetition_penalty: float = DEFAULT_LMSTUDIO_REP_PENALTY
    top_k: Optional[int] = DEFAULT_LMSTUDIO_TOP_K
# --- End Pydantic Model ---

def _retrieve_context(prompt: str) -> tuple[str, bool]:
    """
    Try to retrieve relevant context from the RAG knowledge base.

    Returns:
        (context_str, rag_hit)
        - context_str : formatted context to prepend to the user message,
                        empty string when RAG is disabled or no relevant chunks found.
        - rag_hit     : True only when at least one chunk clears RAG_SCORE_THRESHOLD,
                        i.e. the LLM should ground its answer in the retrieved context.
                        False means fall through to normal conversational LLM behaviour.
    """
    if not RAG_ENABLED:
        return "", False

    try:
        from rag.retriever import retrieve_chunks

        chunks = retrieve_chunks(question=prompt, top_k=RAG_TOP_K)

        if not chunks:
            logger.info("RAG: no chunks retrieved — normal chat mode")
            return "", False

        best_score = max(c.get("score", 0.0) for c in chunks)

        if best_score < RAG_SCORE_THRESHOLD:
            logger.info(
                "RAG miss: best score %.3f < threshold %.2f — normal chat mode",
                best_score,
                RAG_SCORE_THRESHOLD,
            )
            return "", False

        qualified = [c for c in chunks if c.get("score", 0.0) >= RAG_SCORE_THRESHOLD]

        context_parts = []
        for i, c in enumerate(qualified, start=1):
            context_parts.append(
                f"[Source {i} | {c['document']} | {c['topic']}]\n{c['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        logger.info(
            "RAG hit: injecting %d chunk(s) (best score=%.3f)",
            len(qualified),
            best_score,
        )

        context_block = (
            "Use ONLY the following verified knowledge-base context to answer "
            "the question. If the context does not cover the question, say so "
            "and do not guess.\n\n"
            f"{context}\n\n---\n\n"
        )
        return context_block, True

    except Exception as e:
        logger.warning("RAG retrieval failed: %s — falling back to normal chat", e)
        return "", False
    
# --- LLM Stream Generator Function ---
def generate_llm_text_stream(
    prompt: str,
    history: List[Dict[str, str]],
    llm_temperature: float,
    llm_top_p: float,
    llm_max_tokens: int,
    llm_repetition_penalty: float,
    llm_top_k: Optional[int] = None,
    model: Optional[str] = None
) -> Generator[str, None, None]:
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] LLM Router: Initiating LLM stream request.")

    # Use provided model or fall back to default
    selected_model = model if model else LMSTUDIO_MODEL
    logger.info(f"[{request_id}] LLM Router: Using model: {selected_model}")

    # --- RAG context retrieval (tool-style) ---
    # Injects KB context into the user message only when a relevant chunk is found.
    # Falls through to normal conversational LLM behaviour on a miss.
    rag_context, rag_hit = _retrieve_context(prompt)
    if rag_hit:
        logger.info(f"[{request_id}] RAG HIT — grounding reply in knowledge-base context")
        user_content = rag_context + prompt
    else:
        logger.info(f"[{request_id}] RAG MISS — normal conversational LLM mode")
        user_content = prompt

    messages = [{"role": "system", "content": LMSTUDIO_SYSTEM_PROMPT}]
    if history:
        limited_history = history[-(LLM_CONTEXT_TURN_LIMIT * 2):]
        messages.extend(limited_history)
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": llm_temperature,
        "top_p": llm_top_p,
        "max_tokens": llm_max_tokens if llm_max_tokens != -1 else None, # API might expect null for no limit
        "repeat_penalty": llm_repetition_penalty,
        "stream": True
    }
    if llm_top_k is not None and llm_top_k > 0: # top_k=0 often means disabled
        payload["top_k"] = llm_top_k
    
    # Remove None values from payload, especially if max_tokens became None
    payload = {k: v for k, v in payload.items() if v is not None}

    logger.debug(f"[{request_id}] LLM Router: Sending LLM Payload: {json.dumps(payload, indent=2)}")
    response_obj = None
    stream_start_time = time.time()

    try:
        response_obj = requests.post(
            LMSTUDIO_API_ENDPOINT, json=payload, headers=STREAM_HEADERS, stream=True, timeout=STREAM_TIMEOUT_SECONDS
        )
        response_obj.raise_for_status() # Check for HTTP errors (4xx or 5xx)
        logger.info(f"[{request_id}] LLM Router: LLM API Stream connected after {time.time() - stream_start_time:.3f}s.")
        error_occurred_in_stream = False

        for line in response_obj.iter_lines():
            if error_occurred_in_stream: break # Stop if an error was already processed from the stream
            if not line: continue

            try:
                decoded_line = line.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(f"[{request_id}] LLM Router: Skipping undecodable line in LLM stream: {line[:50]}..."); continue

            if decoded_line.startswith(SSE_DATA_PREFIX):
                json_str = decoded_line[len(SSE_DATA_PREFIX):].strip()
                if json_str == SSE_DONE_MARKER:
                    logger.debug(f"[{request_id}] LLM Router: Received LLM SSE_DONE_MARKER."); break
                if not json_str: continue

                try:
                    data = json.loads(json_str)
                    delta_content = None
                    finish_reason = None

                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "delta" in choice and choice["delta"] is not None:
                            delta_content = choice["delta"].get("content")
                        finish_reason = choice.get("finish_reason")
                    
                    if delta_content is not None:
                        yield delta_content
                    elif "error" in data: # Check for error objects in the stream
                        error_msg_detail = data.get('error', {}).get('message', 'Unknown LLM error from stream data')
                        formatted_error = f"{LLM_FAILED_PREFIX} (Stream Error: {error_msg_detail})"
                        logger.error(f"[{request_id}] LLM Router: {formatted_error}")
                        yield formatted_error; error_occurred_in_stream = True; break
                    
                    if finish_reason:
                        logger.debug(f"[{request_id}] LLM Router: LLM stream finish_reason: {finish_reason}")
                        # Typically, SSE_DONE_MARKER is the primary signal to stop.
                        # Some models might send finish_reason and then SSE_DONE.
                        # If finish_reason indicates an error or abnormal stop, might need specific handling.
                        if finish_reason not in ["stop", "length", None]: # "length" is a common valid reason
                            logger.warning(f"[{request_id}] LLM Router: Stream finished with non-standard reason: {finish_reason}")
                            # Depending on API, may want to break here or yield an indicator.

                except json.JSONDecodeError:
                    logger.warning(f"[{request_id}] LLM Router: Skipping invalid JSON in LLM stream: {json_str[:100]}..."); continue
                except Exception as e_proc:
                    logger.exception(f"[{request_id}] LLM Router: Error processing LLM stream chunk data: '{json_str[:100]}...'")
                    yield f"{LLM_FAILED_PREFIX} (Processing Error: {str(e_proc)})"; error_occurred_in_stream = True; break
        logger.info(f"[{request_id}] LLM Router: LLM stream processing loop finished after {time.time() - stream_start_time:.3f}s.")

    except requests.exceptions.Timeout:
        logger.error(f"[{request_id}] LLM Router:  LLM API stream request timed out after {STREAM_TIMEOUT_SECONDS} seconds.", exc_info=True)
        yield f"{LLM_FAILED_PREFIX} (LLM stream request timed out)"
    except requests.exceptions.RequestException as req_e:
        logger.exception(f"[{request_id}] LLM Router: LLM API stream request failed: {req_e}")
        err_yield = f"{LLM_FAILED_PREFIX} (Error connecting to LLM server)"
        if hasattr(req_e, 'response') and req_e.response is not None:
            try:
                err_json = req_e.response.json()
                detail = err_json.get('error',{}).get('message') or err_json.get('detail', req_e.response.text)
                err_yield = f"{LLM_FAILED_PREFIX} (LLM Server Error: {str(detail)[:200]})" # Truncate potentially long errors
            except json.JSONDecodeError: # If error response is not JSON
                err_yield = f"{LLM_FAILED_PREFIX} (LLM Server Error: {req_e.response.status_code} - {req_e.response.text[:200]})"
        yield err_yield
    except Exception as e: # Catch-all for any other unexpected errors
        logger.exception(f"[{request_id}] LLM Router: Unexpected error during LLM stream generation: {e}")
        yield f"{LLM_FAILED_PREFIX} (Unexpected Error in LLM stream: {str(e)})"
    finally:
        if response_obj:
            response_obj.close()
            logger.debug(f"[{request_id}] LLM Router: Closed LLM API response connection.")
# --- End LLM Stream Generator ---

# --- APIRouter instance ---
router = APIRouter()
# --- End APIRouter instance ---

# --- LLM Chat Endpoint Definition ---
@router.post("/chat/stream", summary="Stream LLM Chat Completions", tags=["LLM"])
async def llm_chat_stream_endpoint_router(request_data: LLMChatRequest):
    logger.info(f"LLM Router: Received POST request to /chat/stream")
    logger.info(f"LLM Router: Request Payload: {request_data.model_dump_json(indent=2)}")
    
    text_generator = generate_llm_text_stream(
        prompt=request_data.prompt,
        history=request_data.history,
        llm_temperature=request_data.temperature,
        llm_top_p=request_data.top_p,
        llm_max_tokens=request_data.max_tokens,
        llm_repetition_penalty=request_data.repetition_penalty,
        llm_top_k=request_data.top_k,
        model=request_data.model
    )
    return StreamingResponse(text_generator, media_type="text/event-stream")
# --- End LLM Chat Endpoint Definition ---