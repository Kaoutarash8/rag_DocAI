def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Découpe en fenêtres de `size` mots avec `overlap` mots de recouvrement."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks
