"""Shared pytest fixtures."""
import urllib.request

import pytest

import config


@pytest.fixture(scope="session")
def require_ollama():
    """Skip the test if the local Ollama server isn't reachable."""
    try:
        urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2.0)
    except Exception:
        pytest.skip("Ollama not reachable at OLLAMA_BASE_URL; start it to run this test")
