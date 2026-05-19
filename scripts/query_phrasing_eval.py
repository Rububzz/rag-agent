from eval import evaluate, query, upload_document

questions = [
    {
        "id": 4,
        "expected_keywords": ["pollen", "ovules", "carpels"],
        "phrasings": {
            "short": "What is the function of the gynoecium?",
            "medium": "What role does the gynoecium play in plant reproduction?",
            "descriptive": "Explain the function of the gynoecium in the reproductive system of a flower.",
        },
    },
    {
        "id": 8,
        "expected_keywords": ["genes", "petals", "development"],
        "phrasings": {
            "short": "What is the ABC model of flower development?",
            "medium": "What does the ABC model explain in flower development?",
            "descriptive": "What is the ABC model of flower development, and why is it important in flowering plants?",
        },
    },
    {
        "id": 12,
        "expected_keywords": ["attract", "pollinators"],
        "phrasings": {
            "short": "What role do petals play in pollination?",
            "medium": "What is the function of petals during pollination?",
            "descriptive": "Explain how petals contribute to the pollination process in flowers.",
        },
    },
    {
        "id": 15,
        "expected_keywords": ["male", "female", "pollen", "egg"],
        "phrasings": {
            "short": "What happens during fertilisation in flowering plants?",
            "medium": "How does fertilisation occur in flowering plants?",
            "descriptive": "What happens during fertilisation in flowering plants, and how does it lead to seed formation?",
        },
    },
    {
        "id": 16,
        "expected_keywords": ["fruit"],
        "phrasings": {
            "short": "What structures develop from the ovary after fertilisation?",
            "medium": "How does the ovary change after fertilisation in flowering plants?",
            "descriptive": "What happens to the ovary after fertilisation, and which reproductive structures are formed from it?",
        },
    },
]


def main():
    upload_document("documents/flower.txt")
    for question in questions:
        short_query_result = query(question["phrasings"]["short"])
        medium_query_result = query(question["phrasings"]["medium"])
        descriptive_query_result = query(question["phrasings"]["descriptive"])

        short_eval_result = evaluate(
            short_query_result["answer"], question["expected_keywords"]
        )
        medium_eval_result = evaluate(
            medium_query_result["answer"], question["expected_keywords"]
        )
        descriptive_eval_result = evaluate(
            descriptive_query_result["answer"], question["expected_keywords"]
        )
        print(
            f"Q{question['id']} - short: {'✓'if short_eval_result['passed'] else '✗'}"
        )
        print(
            f"Q{question['id']} - medium: {'✓'if medium_eval_result['passed'] else '✗'}"
        )
        print(
            f"Q{question['id']} - descriptive: {'✓'if descriptive_eval_result['passed'] else '✗'}"
        )


if __name__ == "__main__":
    main()
