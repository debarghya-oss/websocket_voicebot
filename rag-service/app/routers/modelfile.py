"""
routers/modelfile.py — Modelfile upload, view, and clear endpoints.

POST /modelfile/upload   — upload a plain text file as the active Modelfile
GET  /modelfile/active   — view current Modelfile status and preview
DELETE /modelfile/active — clear the Modelfile, revert to default behaviour
"""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..modelfile_store import set_modelfile, clear_modelfile, get_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/modelfile", tags=["modelfile"])

# Max Modelfile size — 32 KB is more than enough for any system prompt
MAX_MODELFILE_BYTES = 32_768


class ModelfileStatus(BaseModel):
    active:      bool
    filename:    Optional[str]
    uploaded_at: Optional[str]
    preview:     Optional[str]
    char_count:  int


@router.post(
    "/upload",
    response_model=ModelfileStatus,
    summary="Upload a Modelfile",
    description=(
        "Upload a plain text file describing how the assistant should behave. "
        "No special syntax required — just write naturally what you want the assistant to do. "
        "This replaces any previously active Modelfile immediately."
    ),
)
async def upload_modelfile(
    file: UploadFile = File(..., description="Plain text Modelfile"),
):
    # Accept .txt, .md, or files with no extension (bare 'Modelfile')
    filename  = file.filename or "Modelfile"
    content   = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_MODELFILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Modelfile too large ({len(content)} bytes). Max is {MAX_MODELFILE_BYTES} bytes.",
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Modelfile must be UTF-8 encoded plain text.",
        )

    try:
        set_modelfile(text, filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("Modelfile uploaded: '%s'", filename)
    return ModelfileStatus(**get_status())


@router.get(
    "/active",
    response_model=ModelfileStatus,
    summary="View the active Modelfile",
    description="Returns the current Modelfile status, preview, and upload timestamp.",
)
def active_modelfile():
    return ModelfileStatus(**get_status())


@router.delete(
    "/active",
    response_model=ModelfileStatus,
    summary="Clear the active Modelfile",
    description=(
        "Removes the current Modelfile. "
        "The assistant reverts to default neutral RAG behaviour."
    ),
)
def delete_modelfile():
    clear_modelfile()
    return ModelfileStatus(**get_status())
