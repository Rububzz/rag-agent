import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("my_docs")
groqAi = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask(question: str) -> str:
    response = groqAi.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


def chunk_text(text: str, size: int) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = words[i : i + size]
        chunks.append(" ".join(chunk))
    return chunks


def embed(text: str) -> list:
    return model.encode(text).tolist()


def add_documents(docs: list[str]):
    for i, doc in enumerate(docs):
        collection.add(ids=[f"doc{i}"], embeddings=[embed(doc)], documents=[doc])


def search(question: str, n: int = 2) -> list[str]:
    results = collection.query(query_embeddings=[embed(question)], n_results=n)
    return results["documents"][0]


def main():
    with open("sample.txt", "r") as f:
        file = f.read()
    chunks = chunk_text(file, 10)
    print("Adding documents to ChromaDB...")
    add_documents(chunks)
    question = "How does retrieval augmented generation improve language model answers?"
    print(f"\nQuestion: {question}")
    results = search(question)
    context = " ".join(results)
    answer = ask(
        f"Answer this question using only the context provided. Context: {context} Question: {question}"
    )
    print(f"\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
