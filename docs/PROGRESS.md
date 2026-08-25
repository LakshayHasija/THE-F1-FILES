# Progress log

| Day | Date | What I did |
|---|---|---|
| 1 | | Repo scaffold, README, .gitignore, requirements.txt, folder structure |
| 2 | 2026-08-26 | Loaded Ergast CSVs into SQLite (data_loader.py) — 14 tables, ~700K+ rows verified |
| 3 | 2026-08-26 | Built data access layer — query functions for standings, race results, qualifying, driver search/career stats |
| 4 | 2026-08-26 | Built MCP server with 7 structured tools (FastMCP), verified via in-memory Client |
| 5 | 2026-08-26 | Built RAG corpus (driver/race/season documents) + local embedding pipeline (MiniLM + Chroma), full 1950-2024 history |
| 6 | 2026-08-26 | Added RAG answer generation (Groq) — grounded, correctly refuses out-of-corpus questions |
| 7 | 2026-08-26 | Unified MCP + RAG into one 8-tool server, packaged as a Claude Desktop Extension (.mcpb), verified working end-to-end |
| 8 | 2026-08-26 | Added pytest suite — 31 tests, deterministic coverage + a skipped-by-default LLM integration test |
| 9 | 2026-08-26 | Started analytics layer (analysis/era_insights.py) — 5 insights across F1 regulation eras (reliability, competitiveness, constructor dominance, 2023 vs 2024 trajectory) |
| 10 | | |
| 11 | | |
| 12 | | |
| 13 | | |
| 14 | | |
| 15 | | |
| 16 | | |