"""Reciprocal Rank Fusion : score(d) = somme_i  w_i / (k + rang_i(d)), rangs à partir de 1."""

from collections.abc import Hashable, Sequence


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60,
        weights: Sequence[float] | None = None) -> list[tuple[Hashable, float]]:
    """Fusionne plusieurs classements (listes de clés, meilleur en premier).

    Ne dépend que des rangs : pas besoin de normaliser cosinus et BM25, qui n'ont pas la même
    échelle. Égalités départagées par l'ordre de première apparition (déterministe).
    """
    ws = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: dict[Hashable, float] = {}
    for w, ranking in zip(ws, rankings):
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
