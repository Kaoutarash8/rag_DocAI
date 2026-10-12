import re

from app.core.config import get_settings
from app.services import embeddings, llm, vectorstore

NO_ANSWER = "Je ne trouve pas d'information pertinente dans les documents indexés."
SUMMARY_RE = re.compile(
    r"(r[ée]sum|synth[èe]se|summar|de quoi (parle|traite)|overview|contenu (du|de ce))", re.I)
SUMMARY_CHUNKS = 12  # passages max envoyés au LLM pour un résumé


def answer_question(question: str, top_k: int | None = None,
                    document_ids: list[int] | None = None) -> tuple[str, list[dict]]:
    s = get_settings()
    top_k = top_k or s.rag_top_k

    if document_ids and SUMMARY_RE.search(question):
        # Résumé : une recherche par similarité n'a pas de sens, on lit le début des documents.
        ids = document_ids[:5]
        per_doc = max(3, SUMMARY_CHUNKS // len(ids))
        hits = []
        for d in ids:
            hits += vectorstore.get_document_chunks(d, per_doc)
        hits = hits[:SUMMARY_CHUNKS]
    else:
        raw = vectorstore.search(embeddings.embed([question])[0], top_k, document_ids)
        # Si l'utilisateur a choisi des documents, on garde leurs meilleurs passages sans seuil.
        hits = raw if document_ids else [h for h in raw if h.score >= s.rag_min_score]

    if not hits:
        return NO_ANSWER, []

    chunks = [{"source": h.source, "text": h.text} for h in hits]
    answer = llm.generate_answer(question, chunks)
    sources = [
        {"source": h.source, "document_id": h.document_id, "chunk_index": h.chunk_index,
         "score": h.score, "snippet": h.text[:300]}
        for h in hits
    ]
    return answer, sources