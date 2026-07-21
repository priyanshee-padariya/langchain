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

# Load the vector store that ingest.py created.
# IMPORTANT: this embedding model MUST match the one used in ingest.py.
# Embeddings run locally and free.
embeddings = get_embeddings()
vectorstore = load_vector_store(embeddings)

# A retriever fetches the top-k most relevant chunks for a query.
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# The chat model that writes the final answer.
# "openrouter/free" is OpenRouter's auto-router: it picks an available free
# model for you, so your code keeps working even when free models rotate out.
# To pin a specific model instead, swap in e.g.
#   "meta-llama/llama-3.3-70b-instruct:free"   (free, verify it's still live)
#   "openai/gpt-4o-mini"                        (paid, needs credits)
# Browse everything at https://openrouter.ai/models
llm = ChatOpenAI(
    model="openrouter/free",
    temperature=0,
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

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
    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)

    # 2 + 3. Build the prompt and ask the model
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": question})

    # 4. Collect unique source filenames for transparency
    sources = list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))

    return {"answer": answer, "sources": sources}