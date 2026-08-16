# F1 PitWall Agent 🏎️

An agentic assistant for Formula 1 analytics — combining **MCP** (Model Context Protocol) tools over structured race data with a **RAG** layer over F1 regulations and season reviews.

> Ask it things like *"How did Verstappen's pit strategy compare to Norris in the 2023 season?"* or *"What does the sporting regulation say about parc fermé conditions?"* — and it'll route to live stats or grounded document retrieval as needed.

## Why this project

Most portfolio projects touching MCP/RAG use e-commerce or generic support-ticket data. This one uses real F1 race data (1950–2024, sourced from Ergast) to build a system that mirrors how production agentic tools actually get built — structured tool-calling + document grounding in one agent.

## Architecture

```
User question
     │
     ▼
LangChain + Gemini Agent  ──┬──▶ MCP Server (FastMCP) ──▶ SQLite (race data)
                             │
                             └──▶ RAG Retriever ──▶ Vector DB (FIA regs, season reviews)
     │
     ▼
Streamlit UI
```

## Milestones

- [ ] **M1 — Data layer**: Load Ergast F1 dataset into SQLite
- [ ] **M2 — MCP server**: Expose 4 tools over structured race data
- [ ] **M3 — RAG layer**: Chunk + embed FIA regulations / season reviews
- [ ] **M4 — Agent orchestration**: LangChain + Gemini routing between tools and retriever
- [ ] **M5 — Streamlit UI**: Chat interface with source attribution
- [ ] **M6 — Polish**: Docs, tests, sample Q&A

## Tech stack

- **Data**: SQLite, Pandas
- **MCP**: FastMCP
- **RAG**: LangChain, ChromaDB/FAISS, sentence-transformers (or Gemini embeddings)
- **Agent**: LangChain + Gemini 2.5 Flash
- **UI**: Streamlit

## Dataset

[F1 World Championship 1950–2024](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — 14 relational CSVs sourced from the Ergast API (races, results, qualifying, lap times, pit stops, drivers, constructors, standings).

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Project status

🚧 Under active daily development — see commit history for progress.
