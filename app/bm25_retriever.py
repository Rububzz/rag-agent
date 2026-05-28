from rank_bm25 import BM25Okapi


def build_index(chunks: list[str]) -> BM25Okapi:
    data = BM25Okapi([chunk.split() for chunk in chunks])
    return data


def bm25_search(index: BM25Okapi, chunks: list[str], question: str, n: int) -> dict:
    tokenized_query = question.split(" ")
    scores = index.get_scores(tokenized_query)
    top_scores = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:n]
    return top_scores
