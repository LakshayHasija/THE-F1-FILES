"""
Shared pytest configuration.

Tests marked @pytest.mark.integration are skipped by default — they
need GROQ_API_KEY and network access, and their assertions are looser
(checking a grounded fact is present, not exact wording) since LLM
output isn't deterministic run to run.

Run them explicitly with:
    pytest --run-integration
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Also run tests marked @pytest.mark.integration (needs GROQ_API_KEY)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return  # flag passed — don't skip anything
    skip_integration = pytest.mark.skip(reason="needs --run-integration flag")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)