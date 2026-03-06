# app/api/routes/query.py
from fastapi import APIRouter
from app.rag.citations import build_citations
from app.rag.formatting import format_answer_with_sources

router = APIRouter()

@router.post("/query")
def query(payload: dict):
    question = payload["question"]

    # 1) retrieve
    docs = retriever.get_relevant_documents(question)  # returns list[Document]

    # 2) build context
    context = "\n\n".join([d.page_content for d in docs])

    # 3) call LLM (your existing code)
    answer_text = llm_answer(question, context)

    # 4) citations
    citations = build_citations(docs, max_sources=4)

    # Option A: single formatted string
    formatted = format_answer_with_sources(answer_text, citations)
    return {"answer": formatted, "sources": citations}

    # Option B: return answer + sources separately (recommended)
    # return {"answer": answer_text, "sources": citations}