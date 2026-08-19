"""
THE-F1-FILES — MCP Server

Wraps two things as MCP tools:
  - src/data_access.py functions: precise, structured lookups straight
    from SQLite (exact standings, exact race classifications).
  - src/rag_chat.py's ask(): semantic search + LLM synthesis over the
    embedded corpus, for open-ended/narrative questions a structured
    query can't answer directly.

Type hints generate the JSON schema an LLM client sees; docstrings
become the tool descriptions — so both need to be precise, not just
correct for a human reader. Each tool's docstring below also signals
*when* to prefer it over the others, since an LLM host choosing between
7 structured tools and 1 RAG tool needs that distinction to pick well.

Usage:
    python src/mcp_server.py
"""
from fastmcp import FastMCP
from data_access import (
    get_driver_standings,
    get_constructor_standings,
    get_race_calendar,
    get_race_results,
    get_qualifying_results,
    search_driver,
    get_driver_career_summary,
)
from rag_chat import ask as rag_ask
mcp = FastMCP("F1 Files")

@mcp.tool()
def driver_standings(year: int) -> dict:
    """
    Get final (or current) F1 driver championship standings for a season.
    Args:
        year: Season year, e.g. 2023.
    Returns:
        Dict with `count` and `standings` — a list of drivers ordered by
        championship position: position, driver, constructor, points, wins.
    """
    standings = get_driver_standings(year)
    return {"count": len(standings), "standings": standings}

@mcp.tool()
def constructor_standings(year):
    """
    Get final (or current) F1 constructor (team) championship standings.
    Args:
        year: Season year, e.g. 2023.
    Returns:
        Dict with `count` and `standings` — constructors ordered by
        championship position: position, name, points, wins.
    """
    standings = get_constructor_standings(year)
    return {"count": len(standings), "standings": standings}

@mcp.tool()
def race_calendar(year):
    """
    Get the full F1 race calendar for a season, in round order.
    Args:
        year: Season year, e.g. 2023.
    Returns:
        Dict with `count` and `races` — round number, race name, date,
        and circuit ID for each race.
    """
    races = get_race_calendar(year)
    return {"count": len(races), "races": races}

@mcp.tool()
def race_results(year, round_number):
    """
    Get the finishing classification for a specific F1 race.
    Args:
        year: Season year, e.g. 2023.
        round_number: Round within that season (1 = first race of the year).
    Returns:
        Dict with `count` and `results` — drivers in finishing order:
        position, driver, constructor, points, laps, time, status
        (Finished, Retired, DNF reason, etc.).
    """
    results = get_race_results(year, round_number)
    return {"count": len(results), "results": results}

@mcp.tool()
def qualifying_results(year, round_number):
    """
    Get qualifying results for a specific F1 race.
    Args:
        year: Season year, e.g. 2023.
        round_number: Round within that season.
    Returns:
        Dict with `count` and `results` — drivers in qualifying position
        order with Q1/Q2/Q3 lap times where applicable.
    """
    results = get_qualifying_results(year, round_number)
    return {"count": len(results), "results": results}

@mcp.tool()
def find_driver(name_fragment):
    """
    Search F1 drivers by partial name match (forename or surname).
    Args:
        name_fragment: Partial name, e.g. "Ham" or "Verstappen". Matches
            anywhere in forename or surname, not just the start.
    Returns:
        Dict with `count` and `drivers` — each with driverId, forename,
        surname, nationality, dob. Use driverId with driver_career_summary
        for career stats.
    """
    drivers = search_driver(name_fragment)
    return {"count": len(drivers), "drivers": drivers}

@mcp.tool()
def driver_career_summary(driver_id):
    """
    Get career totals for one driver: races, wins, podiums, points.
    Args:
        driver_id: Internal driver ID — look it up first with find_driver.
    Returns:
        Dict with driver name, races, wins, podiums, total_points.
    """
    return get_driver_career_summary(driver_id)

@mcp.tool()
def ask_f1_question(question):
    """
    Answer an open-ended, natural-language question about F1 history
    using semantic search over an embedded corpus (driver careers,
    season championships, race narratives from 1950-2024), with the
    answer grounded strictly in retrieved context — not general
    knowledge, so it will say "I don't know" rather than guess if the
    corpus doesn't cover something.

    Prefer this tool for questions like "who dominated the 2023 season",
    "tell me about a driver's career", or "who won the 1994 title" —
    narrative or synthesis questions without one exact structured answer.

    For questions with one precise correct value — an exact standings
    table, an exact race classification, a specific lap time — prefer
    the structured tools instead (driver_standings, race_results,
    qualifying_results, etc.). Those return exact database values;
    this tool returns an LLM's synthesis of retrieved text, which is
    more flexible but less precise.

    Args:
        question: A natural-language question about F1 history.

    Returns:
        Dict with `answer` (the generated response) and `sources`
        (the retrieved documents it was grounded in — doc_type, the
        underlying text, and identifying metadata — so the answer can
        be checked against what was actually retrieved).
    """
    result = rag_ask(question)
    sources = [
        {
            "doc_type": doc.metadata.get("doc_type"),
            "metadata": doc.metadata,
            "content": doc.page_content,
        }
        for doc in result["sources"]
    ]
    return {"answer": result["answer"], "sources": sources}


if __name__ == "__main__":
    mcp.run()