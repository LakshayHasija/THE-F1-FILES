"""
Tests for src/server.py (the MCP tool server)

Uses FastMCP's in-memory Client to call tools through the real MCP
protocol — same approach as src/test_mcp_tools.py, but as real
assertions instead of output you eyeball.

The 7 structured tools are fast and deterministic (pure SQLite, no
network). ask_f1_question is marked @pytest.mark.integration since it
needs GROQ_API_KEY, network access, and the local embedding model —
skipped by default (see tests/conftest.py). Run it explicitly with:
pytest --run-integration
"""

import pytest
from fastmcp import Client

from server import mcp


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


class TestToolRegistration:
    async def test_all_eight_tools_registered(self, client):
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {
            "driver_standings",
            "constructor_standings",
            "race_calendar",
            "race_results",
            "qualifying_results",
            "find_driver",
            "driver_career_summary",
            "ask_f1_question",
        }


class TestStructuredTools:
    async def test_driver_standings_2023(self, client):
        result = await client.call_tool("driver_standings", {"year": 2023})
        assert result.is_error is False
        assert result.data["standings"][0]["driver"] == "Max Verstappen"
        assert result.data["standings"][0]["points"] == 575.0

    async def test_find_driver_verstappen(self, client):
        result = await client.call_tool(
            "find_driver", {"name_fragment": "Verstappen"}
        )
        assert result.is_error is False
        forenames = {d["forename"] for d in result.data["drivers"]}
        assert {"Jos", "Max"}.issubset(forenames)

    async def test_driver_career_summary_verstappen(self, client):
        result = await client.call_tool(
            "driver_career_summary", {"driver_id": 830}
        )
        assert result.is_error is False
        assert result.data["driver"] == "Max Verstappen"
        assert result.data["wins"] >= 63

    async def test_race_results_2023_bahrain(self, client):
        result = await client.call_tool(
            "race_results", {"year": 2023, "round_number": 1}
        )
        assert result.is_error is False
        assert result.data["results"][0]["driver"] == "Max Verstappen"

    async def test_race_calendar_2023_not_empty(self, client):
        result = await client.call_tool("race_calendar", {"year": 2023})
        assert result.is_error is False
        assert result.data["count"] > 0

    async def test_constructor_standings_2023(self, client):
        result = await client.call_tool("constructor_standings", {"year": 2023})
        assert result.is_error is False
        assert result.data["standings"][0]["constructor"] == "Red Bull"

    async def test_qualifying_results_2023_bahrain(self, client):
        result = await client.call_tool(
            "qualifying_results", {"year": 2023, "round_number": 1}
        )
        assert result.is_error is False
        assert result.data["count"] > 0


@pytest.mark.integration
class TestRagTool:
    async def test_ask_f1_question_1994_championship(self, client):
        result = await client.call_tool(
            "ask_f1_question", {"question": "Who won the 1994 F1 championship?"}
        )
        assert result.is_error is False
        # Exact phrasing varies since it's LLM-generated — check the
        # grounded fact is present, not an exact string match.
        assert "Schumacher" in result.data["answer"]
        assert len(result.data["sources"]) > 0