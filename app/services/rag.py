import re
from collections.abc import Collection

from app.core import metrics
from app.core.config import get_settings
from app.services import llm, profiles, retrieval

NO_ANSWER = "Je ne trouve pas d'information pertinente dans les documents indexés."
MAX_HISTORY = 6  # derniers messages conservés (3 échanges)
_PAGE_WORDS = re.compile(r"\b(page|pages|chapitre|section|où)\b", re.IGNORECASE)
_CITED = re.compile(r"\[([\d,;\s]+)\]")


def _label(h) -> str:
    """Référence lisible transmise au LLM : fichier, page(s), section."""
    parts = [h.source]
    if h.page is not None:
        parts.append(f"p. {h.page}-{h.page_end}" if h.page_end else f"p. {h.page}")
    if h.section:
        parts.append(h.section)
    if h.kind != "text":
        parts.append("tableau" if h.kind == "table" else "figure")
    return ", ".join(parts)


def needs_rewrite(question: str, history: list[dict]) -> bool:
    mode = get_settings().query_rewrite
    if mode == "off":
        return False
    if mode == "always":
        return True
    return bool(history) or bool(_PAGE_WORDS.search(question))  # suivi de conversation ou question « de localisation »


def cited_refs(answer: str) -> set[int]:
    """Numéros d'extraits cités dans la réponse ([1], [2, 3]). Ne prouve PAS que l'extrait justifie la phrase."""
    return {int(n) for m in _CITED.finditer(answer) for n in re.findall(r"\d+", m.group(1))}


def answer_question(question: str, top_k: int | None = None,
                    document_ids: list[int] | None = None,
                    profile: str | None = None,
                    history: list[dict] | None = None, *,
                    allowed_ids: Collection[int] | None) -> tuple[str, list[dict], str, str]:
    """(réponse, sources, nom du profil, requête de recherche effectivement utilisée).

    `allowed_ids` : documents lisibles par l'appelant (voir retrieval.retrieve ; None = scripts seulement).
    `history` : derniers messages [{role, content}] ; sert à la réécriture et au prompt, jamais aux droits d'accès.
    """
    p = profiles.get_profile(profile)
    history = (history or [])[-MAX_HISTORY:]
    search_query = llm.condense_question(question, history) if needs_rewrite(question, history) else question

    timings: dict = {}
    hits = retrieval.retrieve(search_query, p, document_ids, top_k, timings, allowed_ids=allowed_ids)
    metrics.observe_retrieval(p.name, timings)
    if not hits:
        return NO_ANSWER, [], p.name, search_query

    chunks = [{"source": _label(h), "text": h.text} for h in hits]
    answer = llm.generate_answer(question, chunks, history) if history else llm.generate_answer(question, chunks)
    used = cited_refs(answer)
    sources = [
        {"ref": i, "cited": i in used, "source": h.source, "document_id": h.document_id,
         "chunk_index": h.chunk_index, "page": h.page, "page_end": h.page_end, "section": h.section,
         "kind": h.kind, "score": h.score, "score_kind": h.score_kind, "snippet": h.text[:300]}
        for i, h in enumerate(hits, 1)
    ]
    return answer, sources, p.name, search_query
