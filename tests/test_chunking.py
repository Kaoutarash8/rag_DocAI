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


def test_chunk_pages_keep_page_and_section():
    from app.services.chunking import chunk_pages
    from app.services.content import PageContent, Segment
    pages = [PageContent(1, [Segment("a " * 30, "text", "Intro")]),
             PageContent(2, [Segment("b " * 30, "text", "Méthode > Données")])]
    chunks = chunk_pages(pages, 20, 5)
    assert {c.page for c in chunks} == {1, 2}
    assert all("a" not in c.text for c in chunks if c.page == 2)  # jamais à cheval sur 2 pages
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].section == "Méthode > Données"


def test_table_split_repeats_header():
    from app.services.chunking import split_table
    from app.services.content import rows_to_markdown
    md = rows_to_markdown([["Produit", "CA"]] + [[f"p{i}", i * 10] for i in range(60)])
    parts = split_table(md, 40)
    assert len(parts) > 1
    assert all(p.startswith("| Produit | CA |\n|---|---|") for p in parts)
    assert sum(p.count("\n") - 1 for p in parts) == 60  # aucune ligne perdue


def _pages():
    from app.services.content import PageContent, Segment
    return [PageContent(1, [Segment("Para un mot " * 5, "text", "A")]),
            PageContent(2, [Segment("Para deux mot " * 5, "text", "A"), Segment("Autre " * 10, "text", "B")]),
            PageContent(3, [Segment("Fin " * 5, "text", "B")])]


def test_section_strategy_crosses_pages_but_not_sections():
    from app.services.chunking import chunk_pages
    chunks = chunk_pages(_pages(), 60, 0, strategy="section")
    assert [(c.section, c.page, c.page_end) for c in chunks] == [("A", 1, 2), ("B", 2, 3)]


def test_section_strategy_same_result_in_batches():
    from app.services.chunking import Chunker, chunk_pages
    whole = chunk_pages(_pages(), 60, 0, strategy="section")
    ck = Chunker("section", 60, 0)
    parts = []
    for p in _pages():  # un lot par page : la section ouverte survit entre les lots
        parts += ck.feed([p])
    parts += ck.finish()
    assert [(c.text, c.page, c.index) for c in parts] == [(c.text, c.page, c.index) for c in whole]


def test_page_strategy_one_chunk_per_page():
    from app.services.chunking import chunk_pages
    chunks = chunk_pages(_pages(), 100, 0, strategy="page")
    assert [c.page for c in chunks] == [1, 2, 3]


def test_sentence_strategy_never_cuts_a_sentence_and_overlaps():
    from app.services.chunking import chunk_sentences
    text = "Un deux trois. Quatre cinq six. Sept huit neuf. Dix onze douze."
    assert chunk_sentences(text, 6, 3) == [
        "Un deux trois. Quatre cinq six.",
        "Quatre cinq six. Sept huit neuf.",
        "Sept huit neuf. Dix onze douze.",
    ]
    assert chunk_sentences("Une phrase.", 6, 3) == ["Une phrase."]


def test_unknown_strategy_rejected():
    import pytest
    from app.services.chunking import Chunker
    with pytest.raises(ValueError):
        Chunker("magic", 100, 10)
