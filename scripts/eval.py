import json
import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8000"


def load_questions(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def upload_document(filepath: str):
    with open(filepath, "rb") as f:
        response = requests.post(f"{API_URL}/upload", files={"file": f})
    print(response.json())


def query(
    question: str,
    search_k: int = 2,
    rerank_top_k: int = 10,
    use_rerank: bool = False,
    use_rewrite_query: bool = False,
    use_hybrid: bool = False,
) -> dict:
    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": question,
            "search_k": search_k,
            "rerank_top_k": rerank_top_k,
            "use_rerank": use_rerank,
            "use_rewrite_query": use_rewrite_query,
            "use_hybrid": use_hybrid,
        },
    )
    return response.json()


def delete_document():
    response = requests.delete(f"{API_URL}/document")
    print(response.json())


def evaluate(answer: str, expected_keywords: list) -> dict:
    result = {"passed": False, "keywords_found": [], "keywords_missing": []}
    isPassed = True
    for words in expected_keywords:
        if words in answer.lower():
            result["keywords_found"].append(words)
        else:
            isPassed = False
            result["keywords_missing"].append(words)
    result["passed"] = isPassed
    return result


def main(
    search_k: int = 10,
    rerank_top_k: int = 2,
    use_rerank: bool = False,
    use_rewrite_query: bool = False,
    use_hybrid: bool = False,
):
    questions = load_questions("scripts/questions.json")
    delete_document()
    upload_document("documents/Flower - Wikipedia.pdf")
    results = []
    for question in questions:
        query_result = query(
            question["question"],
            search_k,
            rerank_top_k,
            use_rerank,
            use_rewrite_query,
            use_hybrid,
        )

        eval_result = evaluate(query_result["answer"], question["expected_keywords"])
        expected_chunks = question.get("expected_chunks")
        eval_result["expected_chunks"] = expected_chunks
        eval_result["id"] = question["id"]
        eval_result["question"] = question["question"]
        eval_result["sources"] = query_result.get("sources", [])
        retrieved_chunks = [source["chunk_index"] for source in eval_result["sources"]]
        eval_result["retrieved_chunks"] = retrieved_chunks
        if expected_chunks is not None:
            eval_result["retrieval_hit"] = any(
                chunk in retrieved_chunks for chunk in expected_chunks
            )
        else:
            eval_result["retrieval_hit"] = None
        eval_result["duration_ms"] = query_result.get("duration_ms")
        eval_result["token_usage"] = query_result.get("token_usage")
        eval_result["cached"] = query_result.get("cached")
        results.append(eval_result)
        print(
            f"Q{question['id']}: {'✓' if eval_result['passed'] else '✗'} {question['question'][:50]}"
            f"\n sources: {eval_result['sources']}"
            f"\n duration_ms: {eval_result['duration_ms']}"
            f"\n token_usage: {eval_result['token_usage']}"
            f"\n retrieved_chunks: {retrieved_chunks}"
            f"\n expected_chunks: {expected_chunks}"
            f"\n retrieval_hit: {eval_result['retrieval_hit']}"
        )

    durations = [r["duration_ms"] for r in results]
    tokens = [r["token_usage"]["total_tokens"] for r in results]
    average_duration_ms = sum(durations) / len(durations)
    average_total_tokens = sum(tokens) / len(tokens)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n=== EVAL SUMMARY ===")
    print(f"Score: {passed}/{len(results)}")
    print(f"Pass rate: {round(passed/len(results)*100)}%")
    print(f"\n average_duration_ms: {average_duration_ms} ")
    print(f"\n average_total_tokens: {average_total_tokens}")

    retrieval_result = [r for r in results if r["retrieval_hit"] is not None]
    print(f"\n=== Retrieval Hit Rate===")
    if retrieval_result:
        retrieval_hit = sum(1 for r in retrieval_result if r["retrieval_hit"])
        retrieval_hit_rate = round(retrieval_hit / len(retrieval_result) * 100)
        print(f"\n Retrieval Hit Rate: {retrieval_hit_rate}%")
        print(f"\n Retrieval Hit:{retrieval_hit} / {len(retrieval_result)}")
    else:
        retrieval_hit_rate = None
        print(f"\n Retrieval Hit Rate: N/A")

    failed_results = [r for r in results if not r["passed"]]
    print(f"\n === Failed Results")
    for fr in failed_results:
        print(f"Q{fr['id']}:{fr['question']}")
        print(f"Missing Keywords:{fr['keywords_missing']}")

    output = {
        "search_k": search_k,
        "rerank_top_k": rerank_top_k,
        "use_rerank": use_rerank,
        "use_rewrite_query": use_rewrite_query,
        "use_hybrid": use_hybrid,
        "pass_rate": round(passed / len(results) * 100),
        "retrieval_hit_rate": retrieval_hit_rate,
        "average_duration_ms": average_duration_ms,
        "average_total_tokens": average_total_tokens,
        "results": results,
    }

    outputdir = Path("eval_results")
    outputdir.mkdir(exist_ok=True)
    output_path = (
        outputdir
        / f"search_{search_k}_rerank_{rerank_top_k}_use_{int(use_rerank)}_rewrite_{int(use_rewrite_query)}_{int(use_hybrid)}.json"
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved eval results to {output_path}")


if __name__ == "__main__":
    search_k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rerank_top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    use_rerank = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    use_rewrite_query = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    use_hybrid = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

    main(search_k, rerank_top_k, use_rerank, use_rewrite_query, use_hybrid)
