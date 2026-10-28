from app.services import bm25, embeddings, reranker, retrieval, vectorstore
from app.services.fusion import rrf
from app.services.profiles import Profile
from app.services.vectorstore import Hit


def test_rrf_rewards_agreement_between_rankings():
    fused = dict(rrf([["a", "b", "c"], ["c", "a", "d"]], k=60))
    assert fused["a"] > fused["c"] > fused["b"] > fused["d"]
    assert abs(fused["a"] - (1 / 61 + 1 / 62)) < 1e-9


def test_bm25_ranks_exact_terms_and_strips_accents():
    docs = ["Le chiffre d'affaires 2023 est de 12 millions", "Politique de congés payés",
            "Résultat net en hausse"]
    ix = bm25.BM25Index([bm25.tokenize(d) for d in docs])
    assert ix.search(bm25.tokenize("CONGES payes"), 3)[0][0] == 1
    assert ix.search(bm25.tokenize("chiffre 2023"), 3, allowed={1, 2}) == []  # filtre par document
    # couverture : "chiffre" seul ne suffit pas si la requête a 3 termes distincts
    assert ix.search(bm25.tokenize("chiffre congés payés"), 3, min_coverage=1.0) == []


def _profile(**kw):
    base = dict(name="t", collection="c", embedding_model="m", min_score=0.3, top_k=3, candidates=5)
    return Profile(**{**base, **kw})


def _patch(monkeypatch, dense, lexical_texts=()):
    monkeypatch.setattr(embeddings, "embed_query", lambda m, q: [0.0])
    monkeypatch.setattr(vectorstore, "search", lambda col, v, n, ids=None: [h for h in dense][:n])
    monkeypatch.setattr(vectorstore, "get_all", lambda col: (
        [h.text for h in dense] + list(lexical_texts),
        [{"document_id": h.document_id, "chunk_index": h.chunk_index, "source": h.source}
         for h in dense] + [{"document_id": 9, "chunk_index": i, "source": "lex.pdf"}
                            for i in range(len(lexical_texts))]))
    monkeypatch.setattr(vectorstore, "version", lambda c: object().__hash__())  # force la reconstruction


def test_dense_only_applies_min_score(monkeypatch):
    dense = [Hit("bon", "a.pdf", 1, 0, 0.8), Hit("faible", "b.pdf", 2, 0, 0.1)]
    _patch(monkeypatch, dense)
    assert [h.text for h in retrieval.retrieve("q", _profile(), allowed_ids=None)] == ["bon"]
    # avec des documents choisis, pas de seuil
    assert len(retrieval.retrieve("q", _profile(), document_ids=[1, 2], allowed_ids=None)) == 2


def test_hybrid_brings_keyword_match_missed_by_dense(monkeypatch):
    dense = [Hit("texte générique sans rapport", "a.pdf", 1, 0, 0.9)]
    _patch(monkeypatch, dense, ["Le code erreur XJ-4412 apparaît au démarrage"])
    hits = retrieval.retrieve("erreur XJ-4412", _profile(hybrid=True, bm25_min_coverage=0.5), allowed_ids=None)
    assert {h.source for h in hits} == {"a.pdf", "lex.pdf"}
    assert all(h.score_kind == "rrf" for h in hits)


def test_rerank_reorders_and_gates(monkeypatch):
    dense = [Hit("moyen", "a.pdf", 1, 0, 0.9), Hit("excellent", "b.pdf", 2, 0, 0.5)]
    _patch(monkeypatch, dense)
    fake = {"moyen": 0.2, "excellent": 0.95}

    class CE:
        def predict(self, pairs, **kw):
            return [fake[p[1]] for p in pairs]
    monkeypatch.setattr(reranker, "get_model", lambda name: CE())
    hits = retrieval.retrieve("q", _profile(rerank=True, rerank_min_score=0.5), allowed_ids=None)
    assert [(h.text, h.score_kind) for h in hits] == [("excellent", "rerank")]
