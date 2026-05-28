import logging

from app.llm import groqAi

logger = logging.getLogger(__name__)


def rewrite_query(question: str, n: int = 4) -> list[str]:
    try:
        logger.info(f"Calling Groq to give alternative questions")
        response = groqAi.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate exactly 4 alternative queries using the question given. 1 being a semantic paraphrase, 2 being a keyword only search query, 3 being a broader contextual question and 4 being a technical terminolgy version of the question. Return one phrasing per line with no numbering, bullets, or extra text.",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        raw_questions = response.choices[0].message.content
        questions = [
            line.strip() for line in raw_questions.strip().split("\n") if line.strip()
        ]
        filtered = questions[:n]
        logger.info(f"Rewritten queries: {filtered}")
        return filtered
    except Exception as e:
        logger.error(f"Failed to call Groq for rewrite:{e}")
        return []
