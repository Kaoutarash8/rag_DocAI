from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import current_user, rate_limit
from app.api.schemas import AskRequest, AskResponse
from app.database import get_db
from app.models.user import User
from app.services import access, rag

router = APIRouter(tags=["rag"], dependencies=[Depends(rate_limit)])


@router.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Questions sur VOS documents et ceux partagés avec vous (jamais ceux des autres)."""
    allowed = access.accessible_ids(db, user, req.scope)  # SQL : propriétaire + partages
    if req.document_ids and not set(req.document_ids) <= set(allowed):
        raise HTTPException(404, "Document introuvable")  # même réponse qu'un document inexistant
    history = [turn.model_dump() for turn in req.history]
    answer, sources, profile, search_query = rag.answer_question(
        req.question, req.top_k, req.document_ids, req.profile, history, allowed_ids=allowed)
    return {"question": req.question, "answer": answer, "sources": sources, "profile": profile,
            "search_query": search_query}
