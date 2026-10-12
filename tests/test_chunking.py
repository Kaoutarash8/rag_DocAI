from app.services.chunking import chunk_text


def test_empty():
    assert chunk_text("   ", 10, 2) == []


def test_overlap_and_coverage():
    text = " ".join(str(i) for i in range(25))
    chunks = chunk_text(text, 10, 3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    assert chunks[-1].split()[-1] == "24"
    assert all(len(c.split()) <= 10 for c in chunks)


def test_short_text_single_chunk():
    assert chunk_text("a b c", 100, 20) == ["a b c"]
