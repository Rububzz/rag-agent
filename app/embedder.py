from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> list:
    return model.encode(text).tolist()
