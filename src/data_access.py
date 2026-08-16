"""
THE-F1-FILES — Day 3: Data Access Layer

Thin, well-typed query functions over data/f1.db. These are the
functions your FastMCP tools will wrap in a later milestone — so
everything here returns plain JSON-serializable Python (list[dict] /
dict), not DataFrames, keeping the MCP layer a thin pass-through.

Usage:
    python src/data_access.py   # runs a few sanity-check queries
"""

import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "f1.db"

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _rows_to_dicts(rows):
    return [dict(r) for r in rows]


def get_driver_standings(year):
    """Final (or current) driver standings for a season, most recent race."""
    query = """
        SELECT ds.position, d.forename || ' ' || d.surname AS driver,
               c.name AS constructor, ds.points, ds.wins
        FROM driver_standings ds
        JOIN races r        ON ds.raceId = r.raceId
        JOIN drivers d       ON ds.driverId = d.driverId
        JOIN results res     ON res.raceId = r.raceId AND res.driverId = d.driverId
        JOIN constructors c  ON res.constructorId = c.constructorId
        WHERE r.year = ?
          AND r.raceId = (
              SELECT raceId FROM races WHERE year = ? ORDER BY round DESC LIMIT 1
          )
        GROUP BY d.driverId
        ORDER BY ds.position ASC;
    """
    with _connect() as conn:
        rows = conn.execute(query, (year, year)).fetchall()
    return _rows_to_dicts(rows)

def get_constructor_standings(year):
    """Final (or current) constructor standings for a season."""
    query = """
        SELECT cs.position, c.name AS constructor, cs.points, cs.wins
        FROM constructor_standings cs
        JOIN races r ON cs.raceId = r.raceId
        JOIN constructors c ON cs.constructorId = c.constructorId
        WHERE r.raceId = (
            SELECT raceId FROM races WHERE year = ? ORDER BY round DESC LIMIT 1
        )
        ORDER BY cs.position ASC;
    """
    with _connect() as conn:
        rows = conn.execute(query, (year,)).fetchall()
    return _rows_to_dicts(rows)

# ---------------------------------------------------------------------
# Race-level
# ---------------------------------------------------------------------

def get_race_calendar(year):
    """All races in a season, in round order."""
    query = """
        SELECT round, name AS race_name, date, circuitId
        FROM races
        WHERE year = ?
        ORDER BY round ASC;
    """
    with _connect() as conn:
        rows = conn.execute(query, (year,)).fetchall()
    return _rows_to_dicts(rows)

def get_race_results(year, round_):
    """Full classification for a specific race."""
    query = """
        SELECT res.positionOrder AS position,
               d.forename || ' ' || d.surname AS driver,
               c.name AS constructor,
               res.points, res.laps, res.time, st.status
        FROM results res
        JOIN races r        ON res.raceId = r.raceId
        JOIN drivers d       ON res.driverId = d.driverId
        JOIN constructors c  ON res.constructorId = c.constructorId
        JOIN status st       ON res.statusId = st.statusId
        WHERE r.year = ? AND r.round = ?
        ORDER BY res.positionOrder ASC;
    """
    with _connect() as conn:
        rows = conn.execute(query, (year, round_)).fetchall()
    return _rows_to_dicts(rows)

def get_qualifying_results(year, round_):
    """Qualifying classification for a specific race."""
    query = """
        SELECT q.position,
               d.forename || ' ' || d.surname AS driver,
               c.name AS constructor,
               q.q1, q.q2, q.q3
        FROM qualifying q
        JOIN races r        ON q.raceId = r.raceId
        JOIN drivers d       ON q.driverId = d.driverId
        JOIN constructors c  ON q.constructorId = c.constructorId
        WHERE r.year = ? AND r.round = ?
        ORDER BY q.position ASC;
    """
    with _connect() as conn:
        rows = conn.execute(query, (year, round_)).fetchall()
    return _rows_to_dicts(rows)

# ---------------------------------------------------------------------
# Driver-level
# ---------------------------------------------------------------------

def search_driver(name_fragment):
    """Fuzzy search drivers by surname or forename fragment."""
    query = """
        SELECT driverId, forename, surname, nationality, dob
        FROM drivers
        WHERE surname LIKE ? OR forename LIKE ?
        ORDER BY surname ASC;
    """
    like = f"%{name_fragment}%"
    with _connect() as conn:
        rows = conn.execute(query, (like, like)).fetchall()
    return _rows_to_dicts(rows)

def get_driver_career_summary(driver_id):
    """Career totals for a single driver: races, wins, podiums, points."""
    query = """
        SELECT d.forename || ' ' || d.surname AS driver,
               COUNT(res.raceId) AS races,
               SUM(CASE WHEN res.positionOrder = 1 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN res.positionOrder <= 3 THEN 1 ELSE 0 END) AS podiums,
               SUM(res.points) AS total_points
        FROM results res
        JOIN drivers d ON res.driverId = d.driverId
        WHERE d.driverId = ?
        GROUP BY d.driverId;
    """
    with _connect() as conn:
        row = conn.execute(query, (driver_id,)).fetchone()
    return dict(row) if row else {}

if __name__ == "__main__":
    print("Driver standings — 2023 (top 5):")
    for row in get_driver_standings(2023)[:5]:
        print(f"  {row['position']:>2}. {row['driver']:<20} {row['constructor']:<15} {row['points']} pts")

    print("\nSearching drivers matching 'Ver':")
    for row in search_driver("Ver"):
        print(f"  {row['driverId']:>4}  {row['forename']} {row['surname']} ({row['nationality']})")

    print("\n2023 Round 1 race results (top 5):")
    for row in get_race_results(2023, 1)[:5]:
        print(f"  {row['position']:>2}. {row['driver']:<20} {row['constructor']:<15} {row['points']} pts")