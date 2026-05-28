from dataclasses import dataclass

from sentence_transformers import CrossEncoder


@dataclass
class RerankConfig:
    top_n: int = 2
    score_threshold: float = 0.0


model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(question: str, retrieved_result: dict, rerank_config: RerankConfig) -> dict:
    top_n = rerank_config.top_n
    score_threshold = rerank_config.score_threshold
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
    if score_threshold > 0.0:
        result = [ranked[0]]
        i = 1
        while i < len(ranked):
            if ranked[i][2] > score_threshold:
                result.append(ranked[i])
                i += 1
            else:
                break
    else:
        result = ranked

    return {
        "documents": [r[0] for r in result],
        "metadatas": [r[1] for r in result],
        "scores": [float(r[2]) for r in result],
    }
