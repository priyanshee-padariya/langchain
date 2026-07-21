# RAG Chatbot (LangChain + FastAPI + Vue)

A chatbot that answers questions about *your own documents*. Drop files into a
folder, and the bot retrieves the relevant parts and answers from them instead of
guessing.

## How it works

```
Your docs ──▶ ingest.py ──▶ [chunks + embeddings] ──▶ Chroma vector DB
                                                            │
You ask a question ─────────────────────────────────────────┤
                                                            ▼
                          retrieve top matches ──▶ LLM writes answer ──▶ Vue chat UI
```

RAG = **R**etrieval **A**ugmented **G**eneration. The "retrieval" step finds
relevant text; the "generation" step (the LLM) writes the answer using it.

## Project layout

```
rag-chatbot/
├── backend/
│   ├── docs/            ← put YOUR .txt/.md/.pdf files here
│   ├── ingest.py        ← builds the vector store (run once)
│   ├── rag.py           ← retrieval + answer logic
│   ├── main.py          ← FastAPI server
│   ├── requirements.txt
│   └── .env.example
└── frontend/            ← Vue 3 + Vite chat interface
```

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then paste your OpenAI key into .env
```

Add your own documents to `backend/docs/`, then build the vector store:

```bash
python ingest.py
```

Start the API:

```bash
uvicorn main:app --reload
```

It runs on http://localhost:8000

### 2. Frontend

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open the URL it prints (usually http://localhost:5173) and start chatting.

## Things to try once it works

- Change `k` in `rag.py` (`search_kwargs={"k": 4}`) to retrieve more/fewer chunks.
- Change `chunk_size` / `chunk_overlap` in `ingest.py` and re-run it.
- Edit the prompt in `rag.py` to change the bot's tone or rules.
- Swap the model: `gpt-4o-mini` → `gpt-4o` for higher quality (higher cost).

## Don't want to pay for the OpenAI API?

You can run everything locally and free with [Ollama](https://ollama.com):

1. `ollama pull llama3` and `ollama pull nomic-embed-text`
2. `pip install langchain-ollama`
3. In `ingest.py` and `rag.py`, replace the OpenAI imports:
   ```python
   from langchain_ollama import OllamaEmbeddings, ChatOllama
   embeddings = OllamaEmbeddings(model="nomic-embed-text")
   llm = ChatOllama(model="llama3")
   ```
   (and delete the OpenAI embedding/LLM lines)

Everything else stays the same.
```
