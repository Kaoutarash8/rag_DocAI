"""Compare des profils de recherche sur un jeu de questions annotées (sans appeler le LLM).

    python -m scripts.benchmark data/benchmark/questions.jsonl --profiles baseline,e5-fixed,e5-hybrid
    python -m scripts.benchmark data/benchmark/questions.jsonl --profiles all --k 5 --out report.json

Prérequis : les documents sont indexés dans la collection de chaque profil comparé
(INDEX_PROFILES=... puis `python -m scripts.reindex`).

Une ligne du JSONL = une question :
  {"question": "Quel est le CA 2023 ?", "source": "rapport.pdf", "page": 12, "contains": ["12 millions"]}
  - source / page / contains : critères de pertinence d'un passage (au moins un ; tous ceux fournis
    doivent être vrais). `page` accepte une liste. Les pages d'un passage couvrent page..page_end.
  - {"question": "...", "answerable": false} : doit ne rien renvoyer (mesure le seuil min_score).
Métriques : hit@1, hit@k, MRR@k, nDCG@k (pertinence binaire), refus corrects, latences p50/p95.
"""
import argparse
import json
import math
import statistics
import time
import unicodedata
from pathlib import Path


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())


def is_relevant(hit, q: dict) -> bool:
    checks = []
    if q.get("source"):
        checks.append(norm(hit.source) == norm(q["source"]))
    if q.get("page") is not None:
        pages = q["page"] if isinstance(q["page"], list) else [q["page"]]
        lo, hi = hit.page, (hit.page_end or hit.page)
        checks.append(lo is not None and any(lo <= p <= hi for p in pages))
    if q.get("contains"):
        text = norm(hit.text)
        checks.append(all(norm(c) in text for c in q["contains"]))
    return bool(checks) and all(checks)


def score_ranking(rels: list[bool], k: int) -> dict:
    rels = rels[:k]
    first = next((i for i, r in enumerate(rels) if r), None)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rels) if r)
    ideal = 1 / math.log2(2)  # une seule bonne réponse attendue : idéal = rang 1
    return {"hit1": bool(rels and rels[0]), "hitk": first is not None,
            "mrr": 1 / (first + 1) if first is not None else 0.0, "ndcg": dcg / ideal if first is not None else 0.0}


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def evaluate(profile, questions: list[dict], k: int) -> dict:
    from app.services import retrieval

    rows, lat, stages, refusals = [], [], {}, []
    for q in questions:
        timings: dict = {}
        t0 = time.perf_counter()
        hits = retrieval.retrieve(q["question"], profile, None, k, timings, allowed_ids=None)  # hors API : pas de contrôle d'accès
        lat.append((time.perf_counter() - t0) * 1000)
        for name, ms in timings.items():
            stages.setdefault(name, []).append(ms)
        if q.get("answerable", True) is False:
            refusals.append(not hits)
        else:
            rows.append(score_ranking([is_relevant(h, q) for h in hits], k))
    mean = lambda key: round(statistics.fmean(r[key] for r in rows), 3) if rows else None  # noqa: E731
    return {"profile": profile.name, "n": len(rows), f"hit@1": mean("hit1"), f"hit@{k}": mean("hitk"),
            f"mrr@{k}": mean("mrr"), f"ndcg@{k}": mean("ndcg"),
            "refus_corrects": round(statistics.fmean(refusals), 3) if refusals else None,
            "p50_ms": round(pct(lat, 0.5)), "p95_ms": round(pct(lat, 0.95)),
            "stages_p50_ms": {n: round(pct(v, 0.5)) for n, v in stages.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions")
    ap.add_argument("--profiles", default="all", help="liste séparée par des virgules, ou 'all'")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", help="écrit aussi les résultats en JSON")
    args = ap.parse_args()

    from app.services import profiles, vectorstore

    questions = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    profs = profiles.all_profiles()
    names = list(profs) if args.profiles == "all" else [n.strip() for n in args.profiles.split(",")]
    results = []
    for name in names:
        p = profiles.get_profile(name)
        n_chunks = vectorstore.count(p.collection)
        if n_chunks == 0:
            print(f"- {name} : collection vide (non indexée), ignoré")
            continue
        r = evaluate(p, questions, args.k)
        r["chunks"] = n_chunks
        results.append(r)

    k = args.k
    cols = ["profile", "chunks", "hit@1", f"hit@{k}", f"mrr@{k}", f"ndcg@{k}", "refus_corrects", "p50_ms", "p95_ms"]
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in sorted(results, key=lambda r: -(r[f"mrr@{k}"] or 0)):
        print("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in cols) + " |")
    print("\nLatence p50 par étape (ms) :")
    for r in results:
        print(f"  {r['profile']}: {r['stages_p50_ms']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
