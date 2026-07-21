"""Shared configuration for the RAG chatbot."""

DOCS_DIR = "docs"
DB_DIR = "db"

import os

# Embedding provider options: huggingface, huggingface_api, openai, google, ollama
EMBEDDING_PROVIDER = "huggingface_api"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Vector store provider options: chroma, faiss
VECTOR_STORE_PROVIDER = "faiss"


def get_embeddings():
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
    if EMBEDDING_PROVIDER == "huggingface_api":
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(
            model=HUGGINGFACE_EMBEDDING_MODEL,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    if EMBEDDING_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def create_vector_store(documents, embeddings):
    if VECTOR_STORE_PROVIDER == "faiss":
        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(DB_DIR)
        return vectorstore
    if VECTOR_STORE_PROVIDER == "chroma":
        from langchain_chroma import Chroma

        return Chroma.from_documents(documents, embeddings, persist_directory=DB_DIR)
    raise ValueError(f"Unsupported vector store: {VECTOR_STORE_PROVIDER}")


def load_vector_store(embeddings):
    if VECTOR_STORE_PROVIDER == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    if VECTOR_STORE_PROVIDER == "chroma":
        from langchain_chroma import Chroma

        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    raise ValueError(f"Unsupported vector store: {VECTOR_STORE_PROVIDER}")
