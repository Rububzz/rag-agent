import logging
from pathlib import Path

import chromadb

from app.embedder import embed

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("./chroma_data")
COLLECTION_NAME = "documents"
PREVIEW_SIZE = 20

client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection():
    return client.get_or_create_collection(COLLECTION_NAME)


def add_documents(chunks: list[str], filename: str):
    try:
        collection = get_collection()
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"{filename}chunk{i}"],
                embeddings=[embed(chunk)],
                documents=[chunk],
                metadatas=[
                    {
                        "filename": filename,
                        "chunk_index": i,
                        "preview": chunk[:PREVIEW_SIZE],
                    }
                ],
            )
        logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise RuntimeError(f"Failed to index documents: {e}")


def search(question: str, n: int = 2) -> dict:
    try:
        collection = get_collection()
        results = collection.query(
            query_embeddings=[embed(question)],
            n_results=n,
            include=["documents", "metadatas"],
        )
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
        }
    except Exception as e:
        logger.error(f"ChromaDB search failed: {e}")
        raise RuntimeError(f"Search failed: {e}")


def delete_document():
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Collection cleared successfully")
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise RuntimeError(f"Failed to delete with error: {e}")


def multi_search(questions: list[str], n: int):
    cache = set()
    results = []
    for question in questions:
        query_results = search(question, n)
        metadatas = query_results["metadatas"]
        for i in range(len(metadatas)):
            if (metadatas[i]["chunk_index"], metadatas[i]["filename"]) in cache:
                continue
            else:
                results.append(
                    {
                        "document": query_results["documents"][i],
                        "metadata": query_results["metadatas"][i],
                    }
                )
                cache.add((metadatas[i]["chunk_index"], metadatas[i]["filename"]))
    return {
        "documents": [result["document"] for result in results],
        "metadatas": [result["metadata"] for result in results],
    }
