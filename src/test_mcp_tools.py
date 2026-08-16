"""
THE-F1-FILES — Day 4: Quick MCP tool check

Verifies every tool on the MCP server actually works, using FastMCP's
in-memory Client — no Claude Desktop config, no subprocess, no network.
Connects directly to the `mcp` server object in the same process.

Usage:
    python src/test_mcp_tools.py
"""
import asyncio
from fastmcp import Client
from server import mcp

def inspect(label, result):
    """Print everything about a CallToolResult so we can see what's really happening."""
    print(f"\n--- {label} ---")
    print(f"  is_error:          {result.is_error}")
    print(f"  data:              {result.data!r}")
    print(f"  structured_content:{result.structured_content!r}")
    print(f"  content:           {result.content!r}")

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print(f"Registered tools ({len(tools)}):")
        for t in tools:
            print(f"  - {t.name}")
        result = await client.call_tool("driver_standings", {"year": 2023})
        inspect("driver_standings(2023)", result)
        assert result.data is not None, "driver_standings returned no structured data"
        print(f"  -> top 3: {result.data['standings'][:3]}")
        result = await client.call_tool("find_driver", {"name_fragment": "Verstappen"})
        inspect("find_driver('Verstappen')", result)
        print(f"  -> matches: {result.data['drivers']}")
        result = await client.call_tool("driver_career_summary", {"driver_id": 830})
        inspect("driver_career_summary(830)", result)
        result = await client.call_tool(
            "race_results", {"year": 2023, "round_number": 1}
        )
        inspect("race_results(2023, 1)", result)
        print(f"  -> top 3: {result.data['results'][:3]}")

if __name__ == "__main__":
    asyncio.run(main())