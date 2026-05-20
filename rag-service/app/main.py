import time
import logging
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .routers.ingest import router as ingest_router
from .routers.modelfile import router as modelfile_router
from .retriever import retrieve_chunks
from .llm import generate_answer
from .vectorstore import init_collection, get_collection_stats
from .modelfile_store import get_status as modelfile_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Knowledge Base",
    version="2.2.0",
    description=(
        "Local RAG service — Milvus HNSW retrieval + LM Studio (gemma3 4b). "
        "Upload a Modelfile to define the assistant persona. "
        "Use /retrieve for chunks only, /ask for a full grounded answer with citations."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(modelfile_router)


@app.on_event("startup")
async def startup():
    logger.info("Initializing Milvus collection...")
    init_collection()
    logger.info("RAG Knowledge Base ready.")


# ── Schemas ──────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 5
    topic_filter: Optional[str] = None


class ChunkResult(BaseModel):
    text: str
    document: str
    topic: str
    source: str
    score: float
    text_preview: str


class RetrieveResponse(BaseModel):
    chunks: List[ChunkResult]
    total_found: int
    latency_ms: float


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    topic_filter: Optional[str] = None


class CitationSource(BaseModel):
    index: int
    document: str
    topic: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[CitationSource]
    total_chunks_retrieved: int
    latency_ms: float
    modelfile_active: bool          # tells caller which persona generated this answer


# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return get_collection_stats()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """
    Embed the question, search Milvus HNSW, return top-K ranked chunks.
    No LLM call — use this when the orchestrator builds its own prompt.
    """
    t0 = time.perf_counter()

    chunks = retrieve_chunks(
        question=req.question,
        top_k=req.top_k,
        topic_filter=req.topic_filter,
    )

    results = [
        ChunkResult(
            text=c["text"],
            document=c["document"],
            topic=c["topic"],
            source=c["source"],
            score=round(c["score"], 4),
            text_preview=c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        )
        for c in chunks
    ]

    return RetrieveResponse(
        chunks=results,
        total_found=len(results),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Full RAG pipeline in one call:
      1. Embed question → HNSW search → top-K chunks
      2. Assemble context-grounded prompt (truncated to MAX_CONTEXT_CHARS)
      3. Call LM Studio (gemma3 4b instruct) → answer
      4. Return answer + citations

    Use this endpoint for the STT → RAG → TTS voice pipeline.
    """
    t0 = time.perf_counter()

    chunks = retrieve_chunks(
        question=req.question,
        top_k=req.top_k,
        topic_filter=req.topic_filter,
    )

    if not chunks:
        return AskResponse(
            answer="No relevant documents found in the knowledge base.",
            sources=[],
            total_chunks_retrieved=0,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    result = generate_answer(question=req.question, chunks=chunks)

    return AskResponse(
        answer=result["answer"],
        sources=[CitationSource(**s) for s in result["sources"]],
        total_chunks_retrieved=len(chunks),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        modelfile_active=modelfile_status()["active"],
    )