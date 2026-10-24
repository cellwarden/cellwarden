"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _private_stage_times(tmp_path_factory, monkeypatch):
    """Stage timings from test runs go to a temporary file, never to ~/.cellwarden/stage_times.json."""
    monkeypatch.setenv("CELLWARDEN_STAGE_TIMES", str(tmp_path_factory.mktemp("stage_times") / "stage_times.json"))
