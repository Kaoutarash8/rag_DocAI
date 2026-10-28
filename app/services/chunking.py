"""Stratégies de découpage : fixed | page | section | sentence.

Toutes produisent des `Chunk` portant `page`, `section` et `kind`, de sorte que les citations
restent exactes quelle que soit la stratégie. Les tableaux et figures sont des chunks à part
(jamais mélangés au texte courant).

- fixed    : fenêtres de `size` mots avec `overlap` mots de recouvrement, confinées à un segment
             (un chunk ne franchit jamais une page ni une section).
- page     : une page = un chunk (découpée par paragraphes si elle dépasse `size` mots).
- section  : une section = un chunk (paragraphes regroupés jusqu'à `size` mots) ; peut franchir
             une frontière de page, `page` = première page, `page_end` = dernière.
- sentence : phrases regroupées jusqu'à `size` mots ; le chevauchement reprend les dernières
             phrases tenant dans `overlap` mots (jamais de phrase coupée en deux).

Le découpage "section" est le seul à garder un état entre les pages : on l'utilise via `Chunker`
(feed par lots de pages, puis finish()), ce qui garde la mémoire bornée sur les gros documents.
"""

import re

from app.services.content import Chunk, PageContent

STRATEGIES = ("fixed", "page", "section", "sentence")

_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"«(\[]?[A-ZÀ-ÖØ-Ý0-9])")
_PARA_RE = re.compile(r"\n\s*\n")


# ---------------------------------------------------------------- primitives

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


def split_table(markdown: str, size: int) -> list[str]:
    """Découpe un tableau Markdown par lignes en répétant l'en-tête dans chaque morceau.

    Une ligne n'est jamais coupée ; un tableau court reste en un seul chunk.
    """
    lines = markdown.splitlines()
    if len(markdown.split()) <= size or len(lines) <= 3:
        return [markdown]
    header, body = lines[:2], lines[2:]
    head_words = len(" ".join(header).split())
    parts, current, words = [], [], head_words
    for line in body:
        n = len(line.split())
        if current and words + n > size:
            parts.append("\n".join(header + current))
            current, words = [], head_words
        current.append(line)
        words += n
    if current:
        parts.append("\n".join(header + current))
    return parts


def split_sentences(text: str) -> list[str]:
    """Phrases : coupe sur les retours à la ligne puis sur la ponctuation forte + majuscule.

    Heuristique volontairement simple (une abréviation suivie d'une majuscule peut couper à tort).
    """
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.extend(s.strip() for s in _SENT_RE.split(line) if s.strip())
    return out


def chunk_sentences(text: str, size: int, overlap: int) -> list[str]:
    """Regroupe des phrases entières jusqu'à `size` mots, avec chevauchement en phrases."""
    units: list[str] = []
    for s in split_sentences(text):
        units.extend(chunk_text(s, size, 0) if len(s.split()) > size else [s])

    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    fresh = False  # `cur` contient-il du contenu non encore émis ?
    for u in units:
        k = len(u.split())
        if cur and n + k > size:
            if fresh:
                chunks.append(" ".join(cur))
            tail: list[str] = []
            m = 0
            for prev in reversed(cur):
                pk = len(prev.split())
                if m + pk > overlap:
                    break
                tail.insert(0, prev)
                m += pk
            while tail and m + k > size:
                m -= len(tail.pop(0).split())
            cur, n, fresh = tail, m, False
        cur.append(u)
        n += k
        fresh = True
    if cur and fresh:
        chunks.append(" ".join(cur))
    return chunks


Para = tuple[int | None, str]  # (page, texte du paragraphe)


def pack_paragraphs(paras: list[Para], size: int) -> list[list[Para]]:
    """Regroupe des paragraphes consécutifs en groupes de <= `size` mots.

    Un paragraphe plus long que `size` est découpé en fenêtres (un groupe chacune, sans overlap).
    """
    groups: list[list[Para]] = []
    cur: list[Para] = []
    n = 0
    for page, text in paras:
        w = len(text.split())
        if w > size:
            if cur:
                groups.append(cur)
            cur, n = [], 0
            groups.extend([(page, part)] for part in chunk_text(text, size, 0))
            continue
        if cur and n + w > size:
            groups.append(cur)
            cur, n = [], 0
        cur.append((page, text))
        n += w
    if cur:
        groups.append(cur)
    return groups


def _paragraphs(page: int | None, text: str) -> list[Para]:
    return [(page, p.strip()) for p in _PARA_RE.split(text) if p.strip()]


def _group_to_chunk(group: list[Para], section: str) -> Chunk:
    first, last = group[0][0], group[-1][0]
    return Chunk("\n\n".join(t for _, t in group), first, section, "text",
                 page_end=last if last != first else None)


# ---------------------------------------------------------------- Chunker (par lots de pages)

class Chunker:
    """Découpe un document page par page ; `feed` peut être appelé par lots, puis `finish`."""

    def __init__(self, strategy: str, size: int, overlap: int, start_index: int = 0) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Stratégie de chunking inconnue : {strategy} ({', '.join(STRATEGIES)})")
        self.strategy, self.size, self.overlap = strategy, size, overlap
        self._index = start_index
        self._group: list[Para] = []  # état de la stratégie "section"
        self._section = ""

    # -- API publique
    def feed(self, pages: list[PageContent]) -> list[Chunk]:
        out: list[Chunk] = []
        for p in pages:
            out += self._section_page(p) if self.strategy == "section" else self._page(p)
        return self._number(out)

    def finish(self) -> list[Chunk]:
        return self._number(self._drain(final=True))

    # -- internes
    def _number(self, chunks: list[Chunk]) -> list[Chunk]:
        for c in chunks:
            c.index = self._index
            self._index += 1
        return chunks

    def _special(self, seg, page: int | None) -> list[Chunk]:
        """Tableaux et figures : toujours des chunks à part."""
        if seg.kind == "table":
            parts = split_table(seg.text, self.size)
        else:
            parts = chunk_text(seg.text, self.size * 2, self.overlap)
        return [Chunk(t, page, seg.section, seg.kind) for t in parts]

    def _page(self, p: PageContent) -> list[Chunk]:
        out: list[Chunk] = []
        segs = [s for s in p.segments if s.text.strip()]
        if self.strategy == "page":
            paras = [pp for s in segs if s.kind == "text" for pp in _paragraphs(p.page, s.text)]
            section = next((s.section for s in segs if s.kind == "text" and s.section), "")
            out += [_group_to_chunk(g, section) for g in pack_paragraphs(paras, self.size)]
            for s in segs:
                if s.kind != "text":
                    out += self._special(s, p.page)
            return out
        for s in segs:
            if s.kind != "text":
                out += self._special(s, p.page)
            elif self.strategy == "sentence":
                out += [Chunk(t, p.page, s.section, "text")
                        for t in chunk_sentences(s.text, self.size, self.overlap)]
            else:  # fixed
                out += [Chunk(t, p.page, s.section, "text")
                        for t in chunk_text(s.text, self.size, self.overlap)]
        return out

    def _section_page(self, p: PageContent) -> list[Chunk]:
        out: list[Chunk] = []
        for s in p.segments:
            if not s.text.strip():
                continue
            if s.kind != "text":
                out += self._special(s, p.page)
                continue
            if s.section != self._section:
                out += self._drain(final=True)  # nouvelle section : on clôt la précédente
                self._section = s.section
            self._group += _paragraphs(p.page, s.text)
            out += self._drain(final=False)
        return out

    def _drain(self, final: bool) -> list[Chunk]:
        """Émet les groupes complets ; garde le dernier (partiel) tant que la section continue."""
        if not self._group:
            return []
        groups = pack_paragraphs(self._group, self.size)
        if final:
            self._group = []
            emit = groups
        else:
            self._group = list(groups[-1])
            emit = groups[:-1]
        return [_group_to_chunk(g, self._section) for g in emit]


def chunk_pages(pages: list[PageContent], size: int, overlap: int,
                start_index: int = 0, strategy: str = "fixed") -> list[Chunk]:
    """Découpe une liste de pages en une fois (comportement historique : stratégie `fixed`)."""
    ck = Chunker(strategy, size, overlap, start_index)
    return ck.feed(pages) + ck.finish()
