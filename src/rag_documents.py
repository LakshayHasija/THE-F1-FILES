"""
THE-F1-FILES — RAG corpus builder

The Ergast dataset is entirely relational (IDs, foreign keys, numbers) —
there's no prose to embed. This module manufactures the text corpus by
turning rows into readable paragraphs, since semantic search is only as
good as the text you feed it. Three document types:

  - Driver documents: one paragraph per driver's career.
  - Race documents:   one paragraph per race (winner, podium, notable
                       retirements), for races from `start_year` onward.
  - Season documents: one paragraph per season's final championship
                       standings (drivers' and constructors' titles).
                       Added after discovering the model correctly
                       refused to answer "who won the 1994 title" from
                       individual race documents alone — race wins
                       aren't the same as championship points, and
                       there was no season-level document to retrieve.

Usage:
    python src/rag_documents.py          # prints a few sample docs
"""

from langchain_core.documents import Document
from data_access import (
    get_all_drivers,
    get_driver_career_summary,
    get_driver_standings,
    get_constructor_standings,
    get_races_since,
    get_race_info,
    get_race_results,
)


def build_driver_documents():
    """One Document per driver who has at least one race on record."""
    documents = []
    for driver in get_all_drivers():
        summary = get_driver_career_summary(driver["driverId"])
        if not summary or not summary.get("races"):
            continue  # no results on record — nothing to say

        text = (
            f"{driver['forename']} {driver['surname']} is a "
            f"{driver['nationality']} Formula 1 driver, born {driver['dob']}. "
            f"Over a career of {summary['races']} races, they won "
            f"{summary['wins']} race(s), finished on the podium "
            f"{summary['podiums']} time(s), and scored "
            f"{summary['total_points']} championship points."
        )

        documents.append(
            Document(
                id=f"driver_{driver['driverId']}",
                page_content=text,
                metadata={
                    "doc_type": "driver",
                    "driver_id": driver["driverId"],
                    "driver_name": f"{driver['forename']} {driver['surname']}",
                },
            )
        )

    return documents

def _summarize_race(year, round_):
    """Build one narrative paragraph for a single race, or None if no data."""
    info = get_race_info(year, round_)
    results = get_race_results(year, round_)
    if not info or not results:
        return None

    winner = results[0]
    podium = results[1:3]
    podium_text = " and ".join(f"{p['driver']} ({p['constructor']})" for p in podium)

    retirements = [
        r for r in results
        if r["status"] not in ("Finished",) and not r["status"].startswith("+")
    ]

    text = (
        f"The {year} {info['race_name']} (Round {round_}) was held at "
        f"{info['circuit_name']} in {info['location']}, {info['country']} "
        f"on {info['date']}. The race was won by {winner['driver']} "
        f"driving for {winner['constructor']}, with {podium_text} "
        f"rounding out the podium."
    )

    if retirements:
        dnf_text = "; ".join(f"{r['driver']} ({r['status']})" for r in retirements)
        text += f" Notable retirements: {dnf_text}."

    return text

def build_race_documents(start_year = 1950):
    """One Document per race, from start_year (1950 = full history) onward."""
    documents = []

    for race in get_races_since(start_year):
        text = _summarize_race(race["year"], race["round"])
        if text is None:
            continue

        documents.append(
            Document(
                id=f"race_{race['year']}_{race['round']}",
                page_content=text,
                metadata={
                    "doc_type": "race",
                    "year": race["year"],
                    "round": race["round"],
                },
            )
        )

    return documents


def _summarize_season(year):
    """Build one narrative paragraph for a season's final standings."""
    driver_standings = get_driver_standings(year)
    if not driver_standings:
        return None

    champion = driver_standings[0]
    runner_up = driver_standings[1] if len(driver_standings) > 1 else None
    third = driver_standings[2] if len(driver_standings) > 2 else None

    text = (
        f"In the {year} Formula 1 World Championship season, "
        f"{champion['driver']} ({champion['constructor']}) won the "
        f"Drivers' Championship with {champion['points']} points and "
        f"{champion['wins']} race win(s)."
    )
    if runner_up:
        text += (
            f" {runner_up['driver']} finished second with "
            f"{runner_up['points']} points."
        )
    if third:
        text += (
            f" {third['driver']} finished third with "
            f"{third['points']} points."
        )

    # Constructors' Championship wasn't awarded until 1958 — Ergast has
    # no constructor_standings rows before that, so this is skipped
    # gracefully for early seasons rather than fabricating a sentence.
    constructor_standings = get_constructor_standings(year)
    if constructor_standings:
        constructor_champion = constructor_standings[0]
        text += (
            f" The Constructors' Championship was won by "
            f"{constructor_champion['constructor']} with "
            f"{constructor_champion['points']} points."
        )

    return text


def build_season_documents(start_year = 1950):
    """One Document per season, summarizing final championship standings."""
    documents = []
    years = sorted({race["year"] for race in get_races_since(start_year)})

    for year in years:
        text = _summarize_season(year)
        if text is None:
            continue

        documents.append(
            Document(
                id=f"season_{year}",
                page_content=text,
                metadata={"doc_type": "season", "year": year},
            )
        )

    return documents


if __name__ == "__main__":
    drivers = build_driver_documents()
    print(f"Built {len(drivers)} driver documents. Sample:\n")
    print(drivers[0].page_content)
    print(drivers[0].metadata)

    races = build_race_documents(start_year=2023)
    print(f"\nBuilt {len(races)} race documents (2023+ sample). Sample:\n")
    print(races[0].page_content)
    print(races[0].metadata)

    seasons = build_season_documents(start_year=1994)
    print(f"\nBuilt {len(seasons)} season documents (1994+ sample). Sample:\n")
    print(seasons[0].page_content)
    print(seasons[0].metadata)