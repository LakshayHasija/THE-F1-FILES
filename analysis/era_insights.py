# %% [markdown]
# # THE-F1-FILES — Era-Based Analysis
#
# Exploring how Formula 1 has changed across major regulatory eras
# (1950–2024): reliability and championship competitiveness.
#
# **Methodology note:** the era boundaries below are a simplified
# categorization based on major regulation changes (engine formulas,
# aerodynamic rules) — a deliberate simplification to make cross-era
# comparison possible, not a definitive historical taxonomy. Real F1
# history has more nuance than 7 buckets can capture.
#
# Run this in VS Code's Interactive Window (each `# %%` block runs as
# a cell, charts render inline) or convert to a notebook if preferred.

# %%
import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "f1.db"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

# %% [markdown]
# ## Define eras

# %%
ERAS = [
    (1950, 1960, "Front-Engine Era"),
    (1961, 1982, "Rear-Engine & Early Aero Era"),
    (1983, 1988, "Turbo Era"),
    (1989, 2005, "Naturally Aspirated Era"),
    (2006, 2013, "V8 Era"),
    (2014, 2021, "Hybrid V6 Turbo Era"),
    (2022, 2024, "Ground Effect Era"),
]
ERA_ORDER = [e[2] for e in ERAS]


def era_for_year(year: int) -> str:
    for start, end, name in ERAS:
        if start <= year <= end:
            return name
    return "Unknown"


# %% [markdown]
# ## Insight 1: Reliability by era (DNF rate)
#
# A DNF here means anything other than "Finished" or a lapped finish
# (e.g. "+1 Lap") — genuine retirements, not race position.

# %%
results = pd.read_sql_query(
    """
    SELECT r.year, res.positionOrder, st.status
    FROM results res
    JOIN races r ON res.raceId = r.raceId
    JOIN status st ON res.statusId = st.statusId
    """,
    conn,
)
results["era"] = results["year"].apply(era_for_year)


def is_dnf(status: str) -> bool:
    return status != "Finished" and not status.startswith("+")


results["dnf"] = results["status"].apply(is_dnf)

dnf_by_era = (results.groupby("era")["dnf"].mean() * 100).reindex(ERA_ORDER)

fig, ax = plt.subplots(figsize=(10, 5))
dnf_by_era.plot(kind="bar", ax=ax, color="#c0392b")
ax.set_ylabel("DNF Rate (%)")
ax.set_title("F1 Reliability Has Transformed: DNF Rate by Era")
ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dnf_rate_by_era.png", dpi=150)
plt.show()

print(dnf_by_era.round(1))

# %% [markdown]
# ## Insight 2: Championship competitiveness by era
#
# Raw point margins between P1 and P2 aren't comparable across eras —
# the scoring system has changed multiple times (e.g. the old 10-6-4-
# 3-2-1 system vs today's 25-18-15-...-1, plus sprint points, fastest-
# lap points at various points in history). Instead, measuring the
# winning margin as a **percentage of the champion's total points** —
# an era-agnostic competitiveness measure that doesn't get distorted
# by points inflation.

# %%
standings_query = """
WITH final_races AS (
    SELECT year, MAX(round) AS max_round FROM races GROUP BY year
)
SELECT r.year, ds.position, ds.points
FROM driver_standings ds
JOIN races r ON ds.raceId = r.raceId
JOIN final_races fr ON r.year = fr.year AND r.round = fr.max_round
WHERE ds.position IN (1, 2)
ORDER BY r.year, ds.position
"""
standings = pd.read_sql_query(standings_query, conn)

pivot = standings.pivot(index="year", columns="position", values="points")
pivot.columns = ["champion_points", "runnerup_points"]
pivot = pivot.dropna()  # a handful of very early years may lack a clean P2

pivot["margin_pct"] = (
    (pivot["champion_points"] - pivot["runnerup_points"])
    / pivot["champion_points"]
    * 100
)
pivot["era"] = pivot.index.to_series().apply(era_for_year)

competitiveness_by_era = pivot.groupby("era")["margin_pct"].mean().reindex(ERA_ORDER)

fig, ax = plt.subplots(figsize=(10, 5))
competitiveness_by_era.plot(kind="bar", ax=ax, color="#2980b9")
ax.set_ylabel("Avg. Winning Margin (% of champion's points)")
ax.set_title("How Close Were F1 Title Fights? By Era")
ax.set_xlabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "competitiveness_by_era.png", dpi=150)
plt.show()

print(competitiveness_by_era.round(1))

# %% [markdown]
# ## Insight 3: Is the Ground Effect Era's high margin a real trend,
# or one outlier season?
#
# That era-level average (32.3%) spans only 3 seasons — a much smaller
# sample than every other era (15-20+ seasons each), so it's worth
# checking whether all three years actually trend similarly high, or
# whether one season alone is pulling the average up.

# %%
ground_effect = pivot[pivot["era"] == "Ground Effect Era"].sort_index()

fig, ax = plt.subplots(figsize=(6, 5))
ground_effect["margin_pct"].plot(kind="bar", ax=ax, color="#8e44ad")
ax.set_ylabel("Winning Margin (% of champion's points)")
ax.set_title("Ground Effect Era: Winning Margin by Season")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ground_effect_margin_by_season.png", dpi=150)
plt.show()

print(ground_effect[["champion_points", "runnerup_points", "margin_pct"]].round(1))

# %% [markdown]
# ## Insight 4: Constructor dominance by era
#
# Championship margin (Insight 2) measures how close *drivers* were —
# but two closely-matched teammates from the same dominant team (e.g.
# Mercedes 2014–16: Hamilton vs. Rosberg was genuinely tight, while
# Mercedes still won almost every race) can produce a close title
# fight while one constructor still dominates. This measures dominance
# at the *team* level instead, using the Herfindahl-Hirschman Index
# (HHI) — a standard concentration metric from economics: the sum of
# each constructor's squared win-share within an era. HHI approaches
# 1.0 when one team wins nearly everything; it approaches 1/n when
# wins are spread evenly across many teams.
#
# Simplification worth naming: this groups by literal constructor name
# in the dataset, so ownership-continuity cases (e.g. Jaguar -> Red
# Bull, Honda -> Brawn -> Mercedes) count as separate constructors
# rather than one continuous team lineage. Same spirit as the era
# boundaries — a deliberate simplification, not an oversight.

# %%
wins = pd.read_sql_query(
    """
    SELECT r.year, c.name AS constructor
    FROM results res
    JOIN races r ON res.raceId = r.raceId
    JOIN constructors c ON res.constructorId = c.constructorId
    WHERE res.positionOrder = 1
    """,
    conn,
)
wins["era"] = wins["year"].apply(era_for_year)


def hhi(group: pd.Series) -> float:
    shares = group.value_counts(normalize=True)
    return (shares**2).sum()


def top1_share(group: pd.Series) -> float:
    return group.value_counts(normalize=True).iloc[0] * 100


hhi_by_era = wins.groupby("era")["constructor"].apply(hhi).reindex(ERA_ORDER)
top1_by_era = wins.groupby("era")["constructor"].apply(top1_share).reindex(ERA_ORDER)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

hhi_by_era.plot(kind="bar", ax=ax1, color="#27ae60")
ax1.set_ylabel("HHI (win concentration, 0-1)")
ax1.set_title("Constructor Dominance by Era (HHI)")
ax1.set_xlabel("")
ax1.tick_params(axis="x", rotation=30)
for label in ax1.get_xticklabels():
    label.set_ha("right")

top1_by_era.plot(kind="bar", ax=ax2, color="#16a085")
ax2.set_ylabel("Leading Constructor's Win Share (%)")
ax2.set_title("Top Constructor's Share of Race Wins by Era")
ax2.set_xlabel("")
ax2.tick_params(axis="x", rotation=30)
for label in ax2.get_xticklabels():
    label.set_ha("right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "constructor_dominance_by_era.png", dpi=150)
plt.show()

print(
    pd.DataFrame(
        {"hhi": hhi_by_era.round(3), "top1_win_share_pct": top1_by_era.round(1)}
    )
)

# %% [markdown]
# ## Insight 5: Why did 2023 and 2024 have such different margins?
#
# 50.4% down to 14.4% is the steepest single-season swing in the whole
# dataset. The final margin alone doesn't say WHEN the gap opened or
# closed — tracking cumulative points round-by-round through each
# season does. Both 2023 and 2024 had sprint races, which award points
# through a separate sprint_results table — summing both tables per
# round so the trajectory matches the official season totals.

# %%
def season_trajectory(year: int) -> pd.DataFrame:
    query = """
        WITH combined_points AS (
            SELECT raceId, driverId, points FROM results
            UNION ALL
            SELECT raceId, driverId, points FROM sprint_results
        )
        SELECT r.round, d.forename || ' ' || d.surname AS driver,
               SUM(cp.points) AS points
        FROM combined_points cp
        JOIN races r ON cp.raceId = r.raceId
        JOIN drivers d ON cp.driverId = d.driverId
        WHERE r.year = ?
        GROUP BY r.round, d.driverId
        ORDER BY r.round
    """
    df = pd.read_sql_query(query, conn, params=(year,))
    cumulative = (
        df.pivot_table(index="round", columns="driver", values="points", aggfunc="sum")
        .fillna(0)
        .cumsum()
    )
    return cumulative


for year in [2023, 2024]:
    cumulative = season_trajectory(year)
    final_standings = cumulative.iloc[-1].sort_values(ascending=False)
    top2 = final_standings.index[:2]

    fig, ax = plt.subplots(figsize=(8, 5))
    for driver in top2:
        ax.plot(cumulative.index, cumulative[driver], marker="o", label=driver)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Points")
    ax.set_title(f"{year} Championship Battle: Top 2 Drivers by Round")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{year}_championship_trajectory.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Takeaways
#
# *(Fill this in after actually looking at the charts above — write
# what the data shows, not what you'd expect it to show. If a number
# surprises you, that's usually the most interesting sentence to write,
# not one to smooth over.)*
#
# - Reliability: ...
# - Competitiveness: ...

# %%
conn.close()