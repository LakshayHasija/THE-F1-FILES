"""
THE-F1-FILES — Day 6: RAG answer generation (retrieve + generate)

Wires the Chroma retriever (src/rag_embed.py) to Groq as the generation
LLM: pull the top-k most relevant documents for a question, hand them to
the model as context, and have it synthesize an answer grounded in that
context — not its own training knowledge, which may be stale or wrong
for anything beyond what's actually in the corpus.

Note: this uses Groq (console.groq.com, LPU inference hardware company,
free tier), not xAI's Grok (console.x.ai) — the two are unrelated
companies despite the near-identical names. Requires GROQ_API_KEY set
in the environment.

Usage:
    python src/rag_chat.py                              # interactive
    python src/rag_chat.py "who won the 2023 title?"     # one-off
"""

import sys
from langchain_groq import ChatGroq
from rag_embed import get_embeddings, get_vector_store

# Check console.groq.com/docs/models for the current lineup if this gets
# retired. gpt-oss-120b is OpenAI's flagship open-weight model, hosted
# on Groq's fast inference — strong reasoning benchmarks, free tier
# available (rate-limited, not credit-metered). Swap to "openai/gpt-oss-20b"
# if you hit rate limits and want a faster/lighter fallback.
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are an assistant answering questions about Formula 1 history, "
    "using only the context provided below. The context comes from a "
    "database with three document types: driver career summaries (full "
    "history), season championship standings (drivers' and constructors' "
    "titles, per year), and individual race result summaries (winner, "
    "podium, notable retirements).\n\n"
    "Rules:\n"
    "- Answer only using the provided context. Do not use outside "
    "knowledge, even if you're confident about it — the corpus may be "
    "incomplete or the person may be testing what you actually retrieved.\n"
    "- For championship/title questions, prefer season-standings "
    "documents over inferring from individual race wins — race wins "
    "alone don't determine the champion (points from podiums matter too).\n"
    "- If the context doesn't contain enough information to answer, say "
    "so plainly rather than guessing.\n"
    "- Retrieved context sometimes includes irrelevant documents — ignore "
    "them rather than forcing them into your answer.\n"
    "- Be concise and specific (names, numbers), not vague."
)


def get_llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0)

# Cached lazily and reused across calls. Fine for the CLI (one call per
# process anyway), but matters once this runs inside the long-lived MCP
# server (src/mcp_server.py) — without caching, every question would
# reload the embedding model and reopen the vector store from scratch.
_embeddings = None
_vector_store = None
_llm = None


def _cached_vector_store():
    global _embeddings, _vector_store
    if _vector_store is None:
        _embeddings = get_embeddings()
        _vector_store = get_vector_store(_embeddings)
    return _vector_store


def _cached_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def retrieve(question, k = TOP_K):
    return _cached_vector_store().similarity_search(question, k=k)

def ask(question, k = TOP_K):
    """Retrieve context, ask Groq, return the answer plus sources used."""
    docs = retrieve(question, k=k)
    context = "\n\n".join(
        f"[{doc.metadata.get('doc_type', 'unknown')}] {doc.page_content}"
        for doc in docs
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = _cached_llm().invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    return {
        "question": question,
        "answer": response.content,
        "sources": docs,
    }


def _source_tag(doc):
    """Human-readable label for a retrieved source, by doc type."""
    meta = doc.metadata
    doc_type = meta.get("doc_type")
    if doc_type == "driver":
        return meta.get("driver_name", "unknown driver")
    if doc_type == "season":
        return f"{meta.get('year')} season"
    if doc_type == "race":
        return f"{meta.get('year')} R{meta.get('round')}"
    return "unknown source"


def _print_result(result):
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"\n  ({len(result['sources'])} source(s) retrieved)")
    for doc in result["sources"]:
        print(f"   - [{doc.metadata.get('doc_type')}] {_source_tag(doc)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _print_result(ask(question))
    else:
        print("THE-F1-FILES — ask a question (Ctrl+C to quit)\n")
        while True:
            try:
                question = input("> ").strip()
                if not question:
                    continue
                _print_result(ask(question))
            except KeyboardInterrupt:
                print("\nBye.")
                break