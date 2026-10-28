"""Structures communes à tous les extracteurs (PDF, images, Office, texte)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Segment:
    text: str
    kind: str = "text"   # text | table | figure
    section: str = ""    # fil d'Ariane : "1 Introduction > 1.2 Méthode"


@dataclass
class PageContent:
    """Une unité de traitement : page PDF, diapositive, groupe de paragraphes, lot de lignes."""
    page: int | None     # None si le format n'a pas de pages (TXT, XLSX, DOCX non paginé)
    segments: list[Segment] = field(default_factory=list)
    ocr: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments if s.text.strip())


@dataclass
class Chunk:
    text: str
    page: int | None
    section: str
    kind: str
    index: int = 0       # position globale dans le document (assignée à l'indexation)
    page_end: int | None = None  # dernière page couverte (stratégies section/page ; None = même page)


class HeadingStack:
    """Suit la hiérarchie des titres rencontrés au fil du document (persiste entre les pages)."""

    def __init__(self) -> None:
        self._stack: list[str] = []

    def push(self, level: int, title: str) -> None:
        level = max(1, level)
        self._stack = self._stack[: level - 1] + [title.strip()]

    @property
    def path(self) -> str:
        return " > ".join(self._stack)


# ---------------------------------------------------------------- tableaux -> Markdown

def _cell(value) -> str:
    s = "" if value is None else str(value)
    return re.sub(r"\s+", " ", s.replace("|", "\\|")).strip()


def rows_to_markdown(rows) -> str:
    """Convertit une grille (liste de lignes) en tableau Markdown ; '' si inutilisable."""
    grid = [[_cell(c) for c in r] for r in rows or []]
    grid = [r for r in grid if any(r)]
    if len(grid) < 1:
        return ""
    width = max(len(r) for r in grid)
    if width < 2 and len(grid) < 2:
        return ""
    grid = [r + [""] * (width - len(r)) for r in grid]
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(lines)


# ---------------------------------------------------------------- police -> niveaux de titres

def font_profile(size_counts: Counter, max_levels: int = 3,
                 ratio: float = 1.12) -> tuple[float, list[float]]:
    """(taille du corps de texte, tailles de titres triées de la plus grande à la plus petite).

    `size_counts` : taille de police (arrondie à 0,5 pt) -> nombre de caractères.
    Le corps est la taille la plus fréquente ; un titre est nettement plus grand (>= 12 %).
    """
    if not size_counts:
        return 0.0, []
    body = size_counts.most_common(1)[0][0]
    bigger = sorted({s for s in size_counts if s >= body * ratio}, reverse=True)
    return body, bigger[:max_levels]
