import logging
from pathlib import Path

import chromadb

from app.embedder import embed

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("./chroma_data")
COLLECTION_NAME = "documents"

client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)


def add_documents(chunks: list[str]):
    try:
        collection = get_collection()
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"chunk{i}"], embeddings=[embed(chunk)], documents=[chunk]
            )
        logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise RuntimeError(f"Failed to index documents: {e}")


def search(question: str, n: int = 2) -> list[str]:
    try:
        collection = get_collection()
        results = collection.query(query_embeddings=[embed(question)], n_results=n)
        return results["documents"][0]
    except Exception as e:
        logger.error(f"ChromaDB search failed: {e}")
        raise RuntimeError(f"Search failed: {e}")


def delete_document():
    try:
        client.delete_collection("my_doc")
        client.create_collection("my_doc")
        logger.info("Collection cleared successfully")
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise RuntimeError(f"Failed to delete with error: {e}")
