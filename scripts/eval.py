import json

import requests

API_URL = "http://localhost:8000"


def load_questions(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def upload_document(filepath: str):
    with open(filepath, "rb") as f:
        response = requests.post(f"{API_URL}/upload", files={"file": f})
    print(response.json())


def query(question: str) -> dict:
    response = requests.post(f"{API_URL}/query", json={"question": question})
    return response.json()


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


def main():
    questions = load_questions("scripts/questions.json")
    upload_document("documents/flower.txt")
    results = []
    for question in questions:
        query_result = query(question["question"])
        eval_result = evaluate(query_result["answer"], question["expected_keywords"])
        eval_result["id"] = question["id"]
        eval_result["question"] = question["question"]
        results.append(eval_result)
        print(
            f"Q{question['id']}: {'✓' if eval_result['passed'] else '✗'} {question['question'][:50]}"
        )

    passed = sum(1 for r in results if r["passed"])
    print(f"\n=== EVAL SUMMARY ===")
    print(f"Score: {passed}/{len(results)}")
    print(f"Pass rate: {round(passed/len(results)*100)}%")


if __name__ == "__main__":
    main()
