import redis

client = redis.Redis(host="localhost", port=6379)

def get_cached(question: str) -> str | None:
    data = client.get(question)
    return data.decode("utf-8") if data else None


def set_cached(question: str, answer: str) -> None:
    client.set(question, answer)


def clear_cache() -> None:
    client.flushdb()
