import chromadb

from app.embedder import embed

client = chromadb.Client()
collection = client.create_collection("my_doc")


def add_document(chunks: list[str]):
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunks{i}"],
            embeddings=[embed(chunk)],
            documents=chunk,
        )


def search(question: str, n: int = 2) -> list[str]:
    results = collection.query(query_embeddings=[embed(question)], n_results=n)
    return results["documents"][0]
