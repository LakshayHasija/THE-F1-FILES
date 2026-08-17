"""
THE-F1-FILES — Day 5: RAG corpus builder

The Ergast dataset is entirely relational (IDs, foreign keys, numbers) —
there's no prose to embed. This module manufactures the text corpus by
turning rows into readable paragraphs, since semantic search is only as
good as the text you feed it. Two document types for now:

  - Driver documents: one paragraph per driver's career.
  - Race documents:   one paragraph per race (winner, podium, notable
                       retirements), for races from `start_year` onward.

Race documents are windowed by year (not all 1,125 races) to keep the
initial corpus a manageable size for embedding — widen the window once
this is working end to end.

Usage:
    python src/rag_documents.py          # prints a few sample docs
"""

from langchain_core.documents import Document

from data_access import (
    get_all_drivers,
    get_driver_career_summary,
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
            continue
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

def build_race_documents(start_year=2015):
    """One Document per race, from start_year to the most recent season."""
    documents = []

    for race in get_races_since(start_year):
        text = _summarize_race(race["year"], race["round"])
        if text is None:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "doc_type": "race",
                    "year": race["year"],
                    "round": race["round"],
                },
            )
        )

    return documents


if __name__ == "__main__":
    drivers = build_driver_documents()
    print(f"Built {len(drivers)} driver documents. Sample:\n")
    print(drivers[0].page_content)
    print(drivers[0].metadata)

    races = build_race_documents(start_year=2023)
    print(f"\nBuilt {len(races)} race documents (2023+). Sample:\n")
    print(races[0].page_content)
    print(races[0].metadata)