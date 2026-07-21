"""
rag.py — The core Retrieval-Augmented Generation logic.

Flow when a question comes in:
  1. Turn the question into an embedding and find the most similar chunks
     in the vector store (retrieval).
  2. Stuff those chunks into a prompt as "context".
  3. Ask the LLM to answer using only that context (generation).
  4. Return the answer plus which files it came from.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import config
from config import get_embeddings, load_vector_store

load_dotenv()

# OpenRouter is OpenAI-compatible: same client, different base URL + key.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def get_retriever():
    try:
        embeddings = get_embeddings()
        vectorstore = load_vector_store(embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4}), None
    except Exception as exc:
        return None, str(exc)


def get_llm():
    if OPENROUTER_API_KEY:
        try:
            return ChatOpenAI(
                model="openrouter/free",
                temperature=0,
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            ), None
        except Exception as exc:
            return None, str(exc)

    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3")), None
    except Exception as exc:
        return None, str(exc)


retriever, retriever_error = get_retriever()
llm, llm_error = get_llm()

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know — do not make things up.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question: str) -> dict:
    if retriever is None:
        return {
            "answer": (
                "The vector store is not available yet. Run ingest.py first or check the embedding setup. "
                f"Details: {retriever_error}"
            ),
            "sources": [],
        }

    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "No relevant documents were found in the knowledge base.", "sources": []}

    if llm is None:
        return {
            "answer": (
                "The LLM is not configured. Set OPENROUTER_API_KEY or make Ollama available. "
                f"Details: {llm_error}"
            ),
            "sources": [
                doc.metadata.get("source", "unknown") for doc in docs
            ],
        }

    # 2 + 3. Build the prompt and ask the model
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})

    # 4. Collect unique source filenames for transparency
    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))

    return {"answer": answer, "sources": sources}