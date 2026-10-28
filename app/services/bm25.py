"""BM25 (Okapi, variante Lucene) sans dépendance : même formule que `rank_bm25.BM25Okapi`,
mais avec index inversé (rapide), filtre par document et critère de couverture des termes."""

import math
import re
import unicodedata
from collections import Counter, defaultdict

STOPWORDS = frozenset("""
le la les un une des de du d l et ou en au aux a à y ne pas plus que qu qui quoi dont où ce cet cette
ces se sa son ses sur par pour dans avec sans est sont etre ete etait ont avoir il elle ils elles on
nous vous je tu me te c j n s m t si mais donc car ni comme quel quelle quels quelles comment
the of and to in is are for on with a an by at from
""".split())
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Minuscules, accents retirés, mots vides français/anglais écartés."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN.findall(text)
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25Index:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.lengths = [len(t) for t in corpus_tokens]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(corpus_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))

    def search(self, query_tokens: list[str], top_n: int, allowed: set[int] | None = None,
               min_coverage: float = 0.0) -> list[tuple[int, float]]:
        """[(indice du document, score BM25)] triés par score décroissant.

        `min_coverage` : fraction minimale des termes DISTINCTS de la requête présents dans le
        passage (0 = désactivé). Signal absolu, contrairement au score BM25 lui-même.
        """
        terms = set(query_tokens)
        if not terms or not self.n:
            return []
        scores: dict[int, float] = defaultdict(float)
        matched: Counter = Counter()
        for term in terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                if allowed is not None and i not in allowed:
                    continue
                norm = 1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1.0)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
                matched[i] += 1
        need = min_coverage * len(terms)
        ranked = [(i, s) for i, s in scores.items() if matched[i] >= need]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked[:top_n]
