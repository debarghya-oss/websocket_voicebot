import os
import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

logger = logging.getLogger(__name__)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "rag_chunks")
EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5


def get_connection():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)


def init_collection():
    get_connection()

    if utility.has_collection(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
        col = Collection(COLLECTION_NAME)
        col.load()
        return col

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields=fields, description="RAG document chunks")
    col = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 400},
    }
    col.create_index(field_name="embedding", index_params=index_params)
    col.load()
    logger.info(f"Collection '{COLLECTION_NAME}' created with HNSW index.")
    return col


def get_collection() -> Collection:
    get_connection()
    col = Collection(COLLECTION_NAME)
    col.load()
    return col


def get_collection_stats() -> dict:
    try:
        get_connection()
        if not utility.has_collection(COLLECTION_NAME):
            return {"collection": COLLECTION_NAME, "num_entities": 0, "status": "not_initialized"}
        col = Collection(COLLECTION_NAME)
        col.flush()
        return {
            "collection": COLLECTION_NAME,
            "num_entities": col.num_entities,
            "status": "ready",
            "index_type": "HNSW",
            "embedding_dim": EMBEDDING_DIM,
        }
    except Exception as e:
        return {"error": str(e)}