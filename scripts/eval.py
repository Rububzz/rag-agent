import json
import sys

import requests

API_URL = "http://localhost:8000"


def load_questions(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def upload_document(filepath: str):
    with open(filepath, "rb") as f:
        response = requests.post(f"{API_URL}/upload", files={"file": f})
    print(response.json())


def query(question: str, top_k: int = 2) -> dict:
    response = requests.post(
        f"{API_URL}/query", json={"question": question, "top_k": top_k}
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


def main(top_k: int = 2):
    questions = load_questions("scripts/questions.json")
    delete_document()
    upload_document("documents/Flower - Wikipedia.pdf")
    results = []
    for question in questions:
        query_result = query(question["question"], top_k=top_k)
        eval_result = evaluate(query_result["answer"], question["expected_keywords"])
        eval_result["id"] = question["id"]
        eval_result["question"] = question["question"]
        eval_result["sources"] = query_result.get("sources", [])
        eval_result["duration_ms"] = query_result.get("duration_ms")
        eval_result["token_usage"] = query_result.get("token_usage")
        eval_result["cached"] = query_result.get("cached")
        results.append(eval_result)
        print(
            f"Q{question['id']}: {'✓' if eval_result['passed'] else '✗'} {question['question'][:50]}"
            f"\n sources: {eval_result['sources']}"
            f"\n duration_ms: {eval_result['duration_ms']}"
            f"\n token_usage: {eval_result['token_usage']}"
        )

    passed = sum(1 for r in results if r["passed"])
    print(f"\n=== EVAL SUMMARY ===")
    print(f"Score: {passed}/{len(results)}")
    print(f"Pass rate: {round(passed/len(results)*100)}%")


if __name__ == "__main__":
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(top_k)
