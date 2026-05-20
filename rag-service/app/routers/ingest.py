import os
import uuid
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..ingestion import ingest_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# In-memory job store (swap for Redis/DB in production)
_jobs: dict[str, dict] = {}


# ── Schemas ──────────────────────────────────────────────────


class IngestResponse(BaseModel):
    job_id: str
    filename: str
    topic: str
    status: str          # queued | processing | done | failed
    chunks_ingested: int
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


class IngestStatusResponse(BaseModel):
    jobs: list[IngestResponse]
    total: int


# ── Helpers ───────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    return suffix


def _run_ingest(job_id: str, file_path: str, filename: str, topic: str):
    """Background task: ingest a single document and update job state."""
    _jobs[job_id]["status"] = "processing"
    try:
        count = ingest_document(file_path=file_path, filename=filename, topic=topic)
        _jobs[job_id].update(
            status="done",
            chunks_ingested=count,
            finished_at=_now(),
        )
        logger.info(f"[{job_id}] Done — {count} chunks from '{filename}'")
    except Exception as exc:
        _jobs[job_id].update(
            status="failed",
            finished_at=_now(),
            error=str(exc),
        )
        logger.error(f"[{job_id}] Failed: {exc}")
    finally:
        # Clean up temp file
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────


@router.post(
    "/",
    response_model=IngestResponse,
    summary="Ingest a single document",
    description=(
        "Upload one PDF, DOCX, or TXT file. "
        "Ingestion runs in the background; poll `/ingest/status/{job_id}` for progress."
    ),
)
async def ingest_single(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF, DOCX, or TXT file to ingest"),
    topic: str = Form(default="general", description="Topic/category tag stored as metadata"),
):
    suffix = _validate_extension(file.filename)
    job_id = str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / f"{job_id}{suffix}"

    # Stream file to disk
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    save_path.write_bytes(content)

    job = {
        "job_id": job_id,
        "filename": file.filename,
        "topic": topic,
        "status": "queued",
        "chunks_ingested": 0,
        "started_at": _now(),
        "finished_at": None,
        "error": None,
    }
    _jobs[job_id] = job

    background_tasks.add_task(_run_ingest, job_id, str(save_path), file.filename, topic)
    logger.info(f"[{job_id}] Queued '{file.filename}' (topic={topic})")

    return IngestResponse(**job)


@router.post(
    "/batch",
    response_model=list[IngestResponse],
    summary="Ingest multiple documents at once",
    description="Upload up to 10 files in one request. Each gets its own background job.",
)
async def ingest_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Up to 10 PDF/DOCX/TXT files"),
    topic: str = Form(default="general", description="Shared topic tag for all files in this batch"),
):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch request.")

    responses = []
    for file in files:
        suffix = _validate_extension(file.filename)
        job_id = str(uuid.uuid4())
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_path = UPLOAD_DIR / f"{job_id}{suffix}"

        content = await file.read()
        if not content:
            logger.warning(f"Skipping empty file: {file.filename}")
            continue
        save_path.write_bytes(content)

        job = {
            "job_id": job_id,
            "filename": file.filename,
            "topic": topic,
            "status": "queued",
            "chunks_ingested": 0,
            "started_at": _now(),
            "finished_at": None,
            "error": None,
        }
        _jobs[job_id] = job
        background_tasks.add_task(_run_ingest, job_id, str(save_path), file.filename, topic)
        logger.info(f"[{job_id}] Batch queued '{file.filename}'")
        responses.append(IngestResponse(**job))

    if not responses:
        raise HTTPException(status_code=400, detail="No valid files to ingest.")

    return responses


@router.get(
    "/status/{job_id}",
    response_model=IngestResponse,
    summary="Get status of a single ingest job",
)
def ingest_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return IngestResponse(**job)


@router.get(
    "/status",
    response_model=IngestStatusResponse,
    summary="List all ingest jobs",
    description="Returns all jobs, optionally filtered by status.",
)
def list_jobs(status: Optional[str] = None):
    jobs = list(_jobs.values())
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    return IngestStatusResponse(
        jobs=[IngestResponse(**j) for j in jobs],
        total=len(jobs),
    )


@router.delete(
    "/status/{job_id}",
    summary="Remove a completed/failed job from history",
)
def delete_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] in ("queued", "processing"):
        raise HTTPException(status_code=409, detail="Cannot delete an active job.")
    del _jobs[job_id]
    return {"deleted": job_id}