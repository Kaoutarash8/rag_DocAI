from fastapi import APIRouter, Depends

from app.api.schemas import AskRequest, AskResponse
from app.core.security import rate_limit, require_api_key
from app.services import rag

router = APIRouter(tags=["rag"], dependencies=[Depends(require_api_key), Depends(rate_limit)])


@router.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    answer, sources = rag.answer_question(req.question, req.top_k, req.document_ids)
    return {"question": req.question, "answer": answer, "sources": sources}
