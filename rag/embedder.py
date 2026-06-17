import os
import logging
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Multilingual embedder (384-dim, same as bge-small — Milvus schema unchanged).
# Maps Bengali queries and English documents into the SAME vector space, so a
# Bengali spoken question retrieves the English context chunks directly.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # set to "cuda" if GPU available

# e5-family models require these prefixes for correct retrieval quality
_IS_E5 = "e5" in EMBEDDING_MODEL.lower()
_QUERY_PREFIX = "query: " if _IS_E5 else ""
_PASSAGE_PREFIX = "passage: " if _IS_E5 else ""


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

def embed_query_text(question: str):
    """Embed a (possibly Bengali) user question with the correct e5 prefix."""
    return get_embedder().embed_query(_QUERY_PREFIX + question)


def embed_passages(chunks):
    """Embed document chunks with the correct e5 prefix."""
    return get_embedder().embed_documents([_PASSAGE_PREFIX + c for c in chunks])
