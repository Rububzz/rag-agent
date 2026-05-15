def chunk_text(text: str, size: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = words[i : i + size]
        chunks.append(" ".join(chunk))
    return chunks
