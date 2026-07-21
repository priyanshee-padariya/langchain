"""
rag.py — The core Retrieval-Augmented Generation logic.

Flow when a question comes in:
  1. Turn the question into an embedding and find the most similar chunks
     in the vector store (retrieval).
  2. Stuff those chunks into a prompt as "context".
  3. Ask the LLM to answer using only that context (generation).
  4. Return the answer plus which files it came from.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from config import get_embeddings, load_vector_store

load_dotenv()

# --- LLM provider settings -------------------------------------------------
# Primary: Hugging Face Inference API (same token used for embeddings).
# Fallback: OpenRouter, if HF isn't configured or fails to initialize.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_CHAT_MODEL = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "meta-llama/llama-3.3-70b-instruct:free")


def get_retriever():
    try:
        embeddings = get_embeddings()
        vectorstore = load_vector_store(embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4}), None
    except Exception as exc:
        return None, str(exc)


def get_llm():
    """Try Hugging Face first, fall back to OpenRouter, then Ollama."""

    # 1. Hugging Face Inference API
    if HF_TOKEN:
        try:
            from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

            endpoint = HuggingFaceEndpoint(
                repo_id=HF_CHAT_MODEL,
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                provider="auto",
                huggingfacehub_api_token=HF_TOKEN,
            )
            return ChatHuggingFace(llm=endpoint), None
        except Exception as exc:
            hf_error = str(exc)
        else:
            hf_error = None
    else:
        hf_error = "HUGGINGFACEHUB_API_TOKEN not set"

    # 2. OpenRouter (fallback)
    if OPENROUTER_API_KEY:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=OPENROUTER_CHAT_MODEL,
                temperature=0,
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            ), None
        except Exception as exc:
            return None, f"HuggingFace failed ({hf_error}); OpenRouter failed ({exc})"

    # 3. Ollama (local, last resort)
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3")), None
    except Exception as exc:
        return None, f"HuggingFace failed ({hf_error}); no OpenRouter key; Ollama failed ({exc})"


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
                "The LLM is not configured. Set HUGGINGFACEHUB_API_TOKEN or OPENROUTER_API_KEY, "
                f"or make Ollama available. Details: {llm_error}"
            ),
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
        }

    # 2 + 3. Build the prompt and ask the model
    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))
    chain = prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({"context": format_docs(docs), "question": question})
    except Exception as exc:
        return {
            "answer": (
                "The LLM call failed (rate limit, unavailable model, or network issue). "
                f"Details: {exc}"
            ),
            "sources": sources,
        }

    # 4. Return the answer plus which files it came from
    return {"answer": answer, "sources": sources}