import os
import logging
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")  # set to "cuda" if GPU available


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} on {DEVICE}")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,  # Required for COSINE similarity with BGE
            "batch_size": 64,
        },
    )
    logger.info("Embedding model loaded.")
    return embedder
