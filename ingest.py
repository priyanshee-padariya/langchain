"""
ingest.py — Run this ONCE (or whenever your docs change).

It reads every file in the docs/ folder, splits them into chunks,
turns each chunk into an embedding vector, and stores them in a
local Chroma database (the chroma_db/ folder).

Usage:
    python ingest.py
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from config import DOCS_DIR, DB_DIR, EMBED_MODEL

load_dotenv()


def load_documents():
    """Load .txt, .md, and .pdf files from the docs/ folder."""
    loaders = [
        DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(DOCS_DIR, glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def main():
    docs = load_documents()
    if not docs:
        print("No documents found. Put some .txt/.md/.pdf files in the docs/ folder.")
        return
    print(f"Loaded {len(docs)} document(s).")

    # Split long documents into overlapping chunks so retrieval is precise.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s).")

    # Embed the chunks and persist them to disk.
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print(f"Done. Vector store saved to ./{DB_DIR}/")


if __name__ == "__main__":
    main()