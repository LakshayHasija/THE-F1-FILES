"""
THE-F1-FILES — Day 5: Embed and store the RAG corpus

Embeds Documents (from rag_documents.py) and stores them in a local,
persistent Chroma vector store. Embedding provider is swappable via
EMBEDDING_PROVIDER below — currently "local" (sentence-transformers,
free, runs on your machine, no API key). Set it to "gemini" later
when you're ready to pay for Google's API; the rest of the pipeline
(Chroma, retrieval, chunking) doesn't care which model produced the
vectors it's searching.

Note: local and Gemini embeddings are NOT interchangeable — different
models, different vector dimensions (384 vs 768). Switching providers
means re-embedding into a fresh collection, not reusing the old one.
That's handled below by naming the collection after the provider.

Usage:
    python src/rag_embed.py
"""

import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag_documents import build_driver_documents, build_race_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_PROVIDER = "local"  # "local" or "gemini" — flip this when ready

# Lightweight, fast, good enough for a portfolio RAG demo (~80MB download,
# cached locally after the first run — no internet needed after that).
# "sentence-transformers/all-mpnet-base-v2" is a slower, higher-quality
# alternative if retrieval quality ever feels lacking.
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# text-embedding-004 is stable and free-tier friendly, for when you
# switch back. Requires GOOGLE_API_KEY / GEMINI_API_KEY in the environment.
GEMINI_MODEL_NAME = "models/text-embedding-004"

BATCH_SIZE = 50
MAX_RETRIES = 3
# No point pausing between batches for a local model — nothing to rate-limit.
BATCH_DELAY_SECONDS = 1 if EMBEDDING_PROVIDER != "local" else 0

def get_embeddings():
    """Return an Embeddings instance for the configured provider."""
    if EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=GEMINI_MODEL_NAME)

    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings):
    return Chroma(
        collection_name=f"f1_files_{EMBEDDING_PROVIDER}",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def embed_in_batches(vector_store, documents, batch_size = BATCH_SIZE):
    """
    Add documents in small batches with a short pause and basic retry, so
    one rate-limit hiccup partway through a few hundred docs doesn't kill
    the whole run.
    """
    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector_store.add_documents(batch)
                print(f"  [{min(start + batch_size, total)}/{total}] embedded")
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"  [FAILED] batch {start}-{start + len(batch)}: {e}")
                else:
                    wait = 2 ** attempt
                    print(f"  [RETRY {attempt}/{MAX_RETRIES}] {e} — waiting {wait}s")
                    time.sleep(wait)
        time.sleep(BATCH_DELAY_SECONDS)

def build_and_persist(start_year = 2023):
    """Build the corpus, embed it, and persist to data/chroma_db."""
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    driver_docs = build_driver_documents()
    race_docs = build_race_documents(start_year=start_year)
    all_docs = driver_docs + race_docs

    print(
        f"Embedding {len(driver_docs)} driver docs + {len(race_docs)} race "
        f"docs ({start_year}+) = {len(all_docs)} total..."
    )
    embed_in_batches(vector_store, all_docs)
    print(f"Done. Persisted to {CHROMA_DIR}")
    return vector_store


if __name__ == "__main__":
    store = build_and_persist(start_year=2023)

    for query in [
        "Which driver dominated the 2023 season?",
        "Tell me about Lewis Hamilton's career",
    ]:
        print(f"\n--- Query: {query!r} ---")
        for doc in store.similarity_search(query, k=3):
            print(f"\n[{doc.metadata['doc_type']}] {doc.page_content}")