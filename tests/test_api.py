# tests/test_api.py
import os
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="session", autouse=True)
def set_api_key_env():
    os.environ["HP_API_KEY"] = "test-key"

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_text_tfidf_ok():
    r = client.post(
        "/predict_text",
        headers={"X-API-Key": "test-key"},
        json={"text": "scanner defectueux et erreur systeme", "model": "tfidf", "return_keywords": True},
    )
    assert r.status_code == 200
    js = r.json()
    assert "label" in js and "proba" in js and "model" in js
    assert js["model"] in ("tfidf", "camembert")
