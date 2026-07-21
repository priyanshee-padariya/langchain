"""
main.py — A tiny FastAPI web server that exposes the chatbot.

Run it with:
    uvicorn main:app --reload

Then POST to http://localhost:8000/chat with JSON: {"question": "..."}
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import answer_question

app = FastAPI(title="RAG Chatbot")

# Allow the Vue dev server (running on port 5173) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


@app.post("/chat")
def chat(query: Query):
    return answer_question(query.question)


@app.get("/")
def health():
    return {"status": "ok"}
