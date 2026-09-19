"""Shared pytest fixtures for hermetic unit tests."""

import pytest


@pytest.fixture(autouse=True)
def isolate_from_local_env(monkeypatch):
    """Keep unit tests hermetic: never let a developer's local .env point
    tests at a real MongoDB or require a real API key."""
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
