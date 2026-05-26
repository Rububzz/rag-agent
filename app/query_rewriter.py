import logging

from app.llm import groqAi

logger = logging.getLogger(__name__)


def rewrite_query(question: str, n: int) -> list[str]:
    try:
        logger.info(f"Calling Groq to give alternative questions")
        response = groqAi.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate exactly {n} alternative phrasings of the user's question to improve document retrieval. Return one phrasing per line with no numbering, bullets, or extra text.",
                },
                {
                    "role": "user",
                    "content": f"Give me a list of {n} number of alternative phrasings of this question: {question}",
                },
            ],
        )
        raw_questions = response.choices[0].message.content
        questions = [
            line.strip() for line in raw_questions.strip().split("\n") if line.strip()
        ]
        filtered = questions[:n]
        logger.info(f"Generated {len(filtered)} number of alternative phrasings")
        return filtered
    except Exception as e:
        logger.error(f"Failed to call Groq for rewrite:{e}")
        return []
