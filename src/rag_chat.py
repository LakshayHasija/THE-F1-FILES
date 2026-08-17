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
# retired — openai/gpt-oss-120b is a solid, free-tier text model as
# of writing. Groq's free tier is rate-limited, not credit-metered.
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are an assistant answering questions about Formula 1 history, "
    "using only the context provided below. The context comes from a "
    "database of driver career summaries (full history) and race result "
    "summaries (2023 season onward only).\n\n"
    "Rules:\n"
    "- Answer only using the provided context. Do not use outside "
    "knowledge, even if you're confident about it — the corpus may be "
    "incomplete or the person may be testing what you actually retrieved.\n"
    "- If the context doesn't contain enough information to answer, say "
    "so plainly rather than guessing.\n"
    "- Retrieved context sometimes includes irrelevant documents — ignore "
    "them rather than forcing them into your answer.\n"
    "- Be concise and specific (names, numbers), not vague."
)

def get_llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0)

def retrieve(question, k = TOP_K):
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    return vector_store.similarity_search(question, k=k)

def ask(question, k = TOP_K):
    """Retrieve context, ask Grok, return the answer plus sources used."""
    docs = retrieve(question, k=k)
    context = "\n\n".join(
        f"[{doc.metadata.get('doc_type', 'unknown')}] {doc.page_content}"
        for doc in docs
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    llm = get_llm()
    response = llm.invoke(
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


def _print_result(result):
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"\n  ({len(result['sources'])} source(s) retrieved)")
    for doc in result["sources"]:
        meta = doc.metadata
        tag = meta.get("driver_name") or f"{meta.get('year')} R{meta.get('round')}"
        print(f"   - [{meta.get('doc_type')}] {tag}")

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