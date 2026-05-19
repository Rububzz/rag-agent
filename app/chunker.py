def chunk_text(text: str, size: int = 200, overlap: int = 0) -> list[str]:
    if size <= 0:
        raise ValueError("size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= size:
        raise ValueError("overlap cannot be more than size")
    words = text.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = words[i : i + size]
        chunks.append(" ".join(chunk))
    return chunks
