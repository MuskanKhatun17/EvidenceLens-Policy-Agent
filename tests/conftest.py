import pytest

@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    monkeypatch.setenv("QUOTA_WAIT_SECONDS", "0")
