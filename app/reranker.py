from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(question: str, retrieved_result: dict, top_n: int = 2) -> dict:
    scores = model.predict([[question, var] for var in retrieved_result["documents"]])
    ranked = sorted(
        [
            [chunk, metadata, score]
            for chunk, metadata, score in zip(
                retrieved_result["documents"], retrieved_result["metadatas"], scores
            )
        ],
        key=lambda x: x[2],
        reverse=True,
    )[:top_n]
    return {
        "documents": [r[0] for r in ranked],
        "metadatas": [r[1] for r in ranked],
        "scores": [float(r[2]) for r in ranked],
    }
