"""
THE-F1-FILES — Day 2: Load Ergast CSVs into SQLite

Scans data/raw/*.csv and loads each file into a SQLite table of the
same name (e.g. drivers.csv -> table `drivers`). Re-running this script
is safe — it replaces existing tables rather than duplicating rows.

Usage:
    python src/data_loader.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# --- Config -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT/"data"/"raw"
DB_PATH = PROJECT_ROOT/"data"/"f1.db"

def load_csvs_to_sqlite(raw_dir, db_path):
    if not raw_dir.exists():
        print(f"[ERROR] Raw data folder not found: {raw_dir}")
        print("        Create it and drop your Ergast CSVs there first.")
        sys.exit(1)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {raw_dir}\n")

    conn = sqlite3.connect(db_path)

    try:
        for csv_path in csv_files:
            table_name = csv_path.stem  # e.g. "drivers.csv" -> "drivers"

            try:
                # low_memory=False avoids dtype-guessing warnings on
                # mixed-type columns (Ergast has a few, e.g. driver 'code')
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                print(f"  [SKIP] {csv_path.name}: failed to read ({e})")
                continue

            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"  [OK]   {table_name:<25} {len(df):>7} rows, {len(df.columns):>2} cols")

        conn.commit()
    finally:
        conn.close()

    print(f"\nDone. Database written to: {db_path}")


def verify(db_path: Path) -> None:
    """Quick sanity check: list tables and row counts."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]

    print(f"\n{len(tables)} table(s) in {db_path.name}:")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t};")
        count = cur.fetchone()[0]
        print(f"  - {t:<25} {count:>7} rows")

    conn.close()


if __name__ == "__main__":
    load_csvs_to_sqlite(RAW_DIR, DB_PATH)
    verify(DB_PATH)