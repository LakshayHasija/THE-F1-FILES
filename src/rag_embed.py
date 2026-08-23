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

import os
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag_documents import build_driver_documents, build_race_documents, build_season_documents

# Same F1_FILES_DATA_DIR override as data_access.py — see that file's
# comment for why. Keeps this working both as a dev script and as a
# packaged Claude Desktop extension pointed at an external data folder.
_env_data_dir = os.environ.get("F1_FILES_DATA_DIR")
if _env_data_dir:
    CHROMA_DIR = Path(_env_data_dir) / "chroma_db"
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_PROVIDER = "local"  # "local" or "gemini" — flip this when ready

# Tested all-mpnet-base-v2 (768-dim, ~420MB) against this same query set —
# it did NOT meaningfully improve retrieval quality (see git history /
# progress log). The bottleneck turned out to be near-identical sentence
# templates across driver docs causing name-token overlap to dominate,
# not embedding model capacity. MiniLM is 5x smaller and faster for the
# same result, so that's what's staying — revisit if doc templates get
# diversified later, not by upgrading the model again.
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


def _collection_name():
    """
    Key the Chroma collection on provider AND model, not just provider —
    different models produce vectors of different dimensions (MiniLM:
    384, mpnet: 768, Gemini text-embedding-004: 768 but a different
    space entirely). Mixing any of these in one collection breaks
    similarity search silently or errors on dimension mismatch. Each
    distinct model gets its own collection instead.
    """
    model_id = GEMINI_MODEL_NAME if EMBEDDING_PROVIDER == "gemini" else LOCAL_MODEL_NAME
    model_slug = model_id.split("/")[-1].replace("-", "_").replace(".", "_")
    return f"f1_files_{EMBEDDING_PROVIDER}_{model_slug}"


def get_vector_store(embeddings):
    return Chroma(
        collection_name=_collection_name(),
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


def build_and_persist(start_year = 1950):
    """
    Build the corpus, embed it, and persist to data/chroma_db.

    Always does a full rebuild of the collection first. The corpus is
    fully regenerable from SQLite, and Chroma doesn't reliably dedupe
    re-added documents even with matching IDs (see langchain-chroma
    issue #24005) — so treating every run as append-only would silently
    duplicate entries on re-runs. Clean slate each time avoids that
    entirely, at the cost of always re-embedding everything (fine at
    this corpus size on a local model — no API cost either way).
    """
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    try:
        vector_store.delete_collection()
        print(f"Cleared existing collection: {_collection_name()}")
    except Exception:
        pass  # collection didn't exist yet — nothing to clear

    vector_store = get_vector_store(embeddings)  # fresh, empty collection

    driver_docs = build_driver_documents()
    race_docs = build_race_documents(start_year=start_year)
    season_docs = build_season_documents(start_year=start_year)
    all_docs = driver_docs + race_docs + season_docs

    print(
        f"Embedding {len(driver_docs)} driver docs + {len(race_docs)} race "
        f"docs + {len(season_docs)} season docs ({start_year}+) = "
        f"{len(all_docs)} total..."
    )
    embed_in_batches(vector_store, all_docs)
    print(f"Done. Persisted to {CHROMA_DIR}")
    return vector_store


if __name__ == "__main__":
    store = build_and_persist(start_year=1950)

    for query in [
        "Which driver dominated the 2023 season?",
        "Tell me about Lewis Hamilton's career",
    ]:
        print(f"\n--- Query: {query!r} ---")
        for doc in store.similarity_search(query, k=3):
            print(f"\n[{doc.metadata['doc_type']}] {doc.page_content}")