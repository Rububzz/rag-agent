import json

import redis

client = redis.Redis(host="redis", port=6379)


def build_cache_key(question: str, top_k: int) -> str:
    return f"{question.lower().strip()}:{top_k}"


def get_cached(question: str, top_k: int) -> dict | None:
    key = build_cache_key(question, top_k)
    data = client.get(key)
    return json.loads(data) if data else None


def set_cached(question: str, top_k: int, context: dict) -> None:
    key = build_cache_key(question, top_k)
    client.set(key, json.dumps(context))


def clear_cache() -> None:
    client.flushdb()
