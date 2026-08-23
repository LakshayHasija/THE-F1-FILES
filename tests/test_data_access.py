"""
Tests for src/data_access.py

All of these run against the real f1.db — no mocking needed, since the
Ergast dataset is a static historical snapshot (1950-2024) and these
facts don't change. Fast and fully deterministic: no network, no API
keys, no model downloads.
"""

import data_access as da


class TestDriverStandings:
    def test_2023_champion_is_verstappen(self):
        standings = da.get_driver_standings(2023)
        champion = standings[0]
        assert champion["driver"] == "Max Verstappen"
        assert champion["constructor"] == "Red Bull"
        assert champion["points"] == 575.0
        assert champion["wins"] == 19

    def test_2023_standings_are_ordered_by_position(self):
        standings = da.get_driver_standings(2023)
        positions = [row["position"] for row in standings]
        assert positions == sorted(positions)

    def test_unknown_year_returns_empty_list(self):
        # No F1 championship existed in 1900
        assert da.get_driver_standings(1900) == []


class TestConstructorStandings:
    def test_2023_champion_is_red_bull(self):
        standings = da.get_constructor_standings(2023)
        assert standings[0]["constructor"] == "Red Bull"

    def test_pre_1958_returns_empty_list(self):
        # Constructors' Championship wasn't awarded until 1958
        assert da.get_constructor_standings(1955) == []


class TestRaceResults:
    def test_2023_bahrain_winner(self):
        results = da.get_race_results(2023, 1)
        assert results[0]["driver"] == "Max Verstappen"
        assert results[0]["status"] == "Finished"

    def test_results_ordered_by_position(self):
        results = da.get_race_results(2023, 1)
        positions = [row["position"] for row in results]
        assert positions == sorted(positions)

    def test_invalid_round_returns_empty_list(self):
        assert da.get_race_results(2023, 99) == []


class TestDriverCareerSummary:
    def test_verstappen_career_totals(self):
        # driverId 830 = Max Verstappen (confirmed via find_driver)
        summary = da.get_driver_career_summary(830)
        assert summary["driver"] == "Max Verstappen"
        assert summary["races"] > 200
        assert summary["wins"] >= 63  # will only grow as more seasons are added
        assert summary["podiums"] >= 112

    def test_unknown_driver_id_returns_empty_dict(self):
        assert da.get_driver_career_summary(999999) == {}


class TestSearchDriver:
    def test_verstappen_matches_both_jos_and_max(self):
        results = da.search_driver("Verstappen")
        surnames = {row["surname"] for row in results}
        forenames = {row["forename"] for row in results}
        assert "Verstappen" in surnames
        assert {"Jos", "Max"}.issubset(forenames)

    def test_no_match_returns_empty_list(self):
        assert da.search_driver("Zzzznonexistentdriver") == []


class TestRaceInfo:
    def test_2023_bahrain_circuit_info(self):
        info = da.get_race_info(2023, 1)
        assert info["circuit_name"] == "Bahrain International Circuit"
        assert info["country"] == "Bahrain"

    def test_invalid_race_returns_empty_dict(self):
        assert da.get_race_info(2023, 99) == {}