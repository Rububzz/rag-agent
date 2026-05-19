import logging
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groqAi = Groq(api_key=os.getenv("GROQ_API_KEY"))
logger = logging.getLogger(__name__)


def ask(question: str, context: str) -> str:
    try:
        logger.info(f"Calling Groq with {question}")
        response = groqAi.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"Answer questions using only the context provided. Context: {context}",
                },
                {"role": "user", "content": question},
            ],
        )
        return {
            "answer": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    except Exception as e:
        logger.error(f"Failed to cal Groq")
        raise RuntimeError(f"LLM request failed: {e}")
