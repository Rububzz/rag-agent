import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groqAi = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask(question: str, context: str) -> str:
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
    return response.choices[0].message.content
