"""
Tests for src/rag_documents.py

These only touch data_access.py (SQLite) to build Document objects —
no embeddings, no vector store, no API calls. Fast and deterministic.
"""

import rag_documents as rd


class TestSeasonDocuments:
    def test_1994_season_summary(self):
        docs = rd.build_season_documents(start_year=1994)
        season_1994 = next(d for d in docs if d.metadata["year"] == 1994)

        assert "Michael Schumacher" in season_1994.page_content
        assert "Benetton" in season_1994.page_content
        assert "92.0 points" in season_1994.page_content
        assert "Damon Hill" in season_1994.page_content
        assert "Williams" in season_1994.page_content  # constructors' champion

    def test_document_id_format(self):
        docs = rd.build_season_documents(start_year=2023)
        ids = {d.id for d in docs}
        assert "season_2023" in ids

    def test_pre_1958_season_has_no_constructors_sentence(self):
        # Constructors' Championship wasn't awarded until 1958 — the
        # summary should degrade gracefully, not crash or fabricate one.
        docs = rd.build_season_documents(start_year=1955)
        season_1955 = next((d for d in docs if d.metadata["year"] == 1955), None)
        assert season_1955 is not None
        assert "Constructors' Championship" not in season_1955.page_content


class TestDriverDocuments:
    def test_lewis_hamilton_bio(self):
        docs = rd.build_driver_documents()
        hamilton = next(d for d in docs if d.metadata["driver_name"] == "Lewis Hamilton")

        assert "British" in hamilton.page_content
        assert "1985-01-07" in hamilton.page_content
        assert hamilton.metadata["doc_type"] == "driver"

    def test_document_id_matches_driver_id(self):
        docs = rd.build_driver_documents()
        hamilton = next(d for d in docs if d.metadata["driver_name"] == "Lewis Hamilton")
        assert hamilton.id == f"driver_{hamilton.metadata['driver_id']}"

    def test_ids_are_unique(self):
        docs = rd.build_driver_documents()
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))


class TestRaceDocuments:
    def test_2023_bahrain_summary(self):
        docs = rd.build_race_documents(start_year=2023)
        bahrain = next(
            d for d in docs
            if d.metadata["year"] == 2023 and d.metadata["round"] == 1
        )
        assert "Max Verstappen" in bahrain.page_content
        assert "Red Bull" in bahrain.page_content
        assert bahrain.id == "race_2023_1"

    def test_ids_are_unique(self):
        docs = rd.build_race_documents(start_year=2020)
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))