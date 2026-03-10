from fastapi import FastAPI
from fastapi import APIRouter

from app.api.routes.query import router as query_router






app = FastAPI()

router = APIRouter()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@router.post("/query")
def query(payload: dict):
    question = payload["question"]

    return {
        "answer": f"You asked: {question}",
        "sources": []
    }
# include query router
app.include_router(query_router)

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")