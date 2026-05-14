import logging

import chromadb

from app.embedder import embed

client = chromadb.Client()
collection = client.create_collection("my_doc")
logger = logging.getLogger(__name__)


def add_documents(chunks: list[str]):
    try:
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"chunks{i}"],
                embeddings=[embed(chunk)],
                documents=[chunk],
            )
        logger.info(f"Added chunks into ChromaDB")
    except Exception as e:
        logger.warning(f"Failed to add chunks to ChromaDB")
        raise RuntimeError(f"Failed to add chunk with message {e}")


def search(question: str, n: int = 2) -> list[str]:
    try:
        results = collection.query(query_embeddings=[embed(question)], n_results=n)
        return results["documents"][0]
    except Exception as e:
        logger.warning(f"Failed to search context")
        raise RuntimeError(f"Failed with error: {e}")
