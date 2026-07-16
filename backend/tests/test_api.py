"""API endpoint tests using FastAPI TestClient (mocked deps)."""

from __future__ import annotations

import pytest


def test_health():
    # Just verify the module imports cleanly; full integration requires DB
    from app.main import app
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_root():
    from app.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert "endpoints" in body
