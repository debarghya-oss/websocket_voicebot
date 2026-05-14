import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from .embedder import get_embedder
from .vectorstore import get_collection

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_document(file_path: str) -> List[str]:
    """Load raw text from PDF, DOCX, or TXT."""
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)
    elif suffix == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    return [d.page_content for d in docs if d.page_content.strip()]


def chunk_texts(texts: List[str]) -> List[str]:
    """Split texts into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    # Deduplicate and filter empties
    seen = set()
    unique = []
    for c in chunks:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def ingest_document(file_path: str, filename: str, topic: str = "general") -> int:
    """
    Full ingestion pipeline:
    1. Load document
    2. Chunk
    3. Embed
    4. Upsert to Milvus
    Returns number of chunks ingested.
    """
    suffix = Path(filename).suffix.lower().lstrip(".")
    logger.info(f"Loading: {filename}")
    texts = load_document(file_path)

    if not texts:
        raise ValueError("Document appears to be empty or unreadable.")

    logger.info(f"Chunking {len(texts)} pages/sections...")
    chunks = chunk_texts(texts)
    logger.info(f"Generated {len(chunks)} chunks.")

    embedder = get_embedder()
    logger.info("Embedding chunks...")
    embeddings = embedder.embed_documents(chunks)

    collection = get_collection()

    batch_size = 128
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]

        data = [
            batch_chunks,                        # text
            [filename] * len(batch_chunks),      # document
            [suffix] * len(batch_chunks),        # source
            [topic] * len(batch_chunks),         # topic
            batch_embs,                          # embedding
        ]
        collection.insert(data)
        total += len(batch_chunks)
        logger.info(f"Inserted batch {i // batch_size + 1}: {len(batch_chunks)} chunks")

    collection.flush()
    logger.info(f"Ingestion complete: {total} chunks for '{filename}'")
    return total
