import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from jobtracker.api import Models, create_app


class Stub:
    def __init__(self, labels, rule):
        self.labels, self.rule = labels, rule

    def predict_proba(self, texts):
        return np.array([self.rule(t) for t in texts], dtype="float32")


def _client():
    s1 = Stub(["job_related", "promotional"], lambda t: [0.9, 0.1] if "bewerbung" in t else [0.1, 0.9])
    s2 = Stub(["acknowledgment", "rejection"], lambda t: [0.1, 0.9])
    return TestClient(create_app(Models(s1, s2, backend="stub")))


def test_health_ok():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_classify_real_application():
    r = _client().post("/classify", json={"emails": [
        {"subject": "Deine Bewerbung bei KFC", "body": "leider absage", "sender": "no-reply@concludis.de"},
        {"subject": "Sale", "body": "rabatt", "sender": "a@shop.de"}]})
    assert r.status_code == 200
    first, second = r.json()
    assert first["category"] == "job_related" and first["authenticity"] == "real_job_application"
    assert first["intent"] == "rejection" and first["intent_score"] == pytest.approx(0.9, abs=1e-6)
    assert second["category"] == "promotional" and second["intent"] == "" and second["intent_score"] is None


def test_validation_and_limits():
    c = _client()
    assert c.post("/classify", json={"emails": []}).status_code == 422
    assert c.post("/classify", json={"emails": [{"subject": "x"}] * 51}).status_code == 422
    assert c.post("/classify", json={"nope": 1}).status_code == 422


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("JOBTRACKER_API_KEY", "secret")
    c = _client()
    body = {"emails": [{"subject": "Sale"}]}
    assert c.post("/classify", json=body).status_code == 401
    assert c.post("/classify", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/classify", json=body, headers={"X-API-Key": "secret"}).status_code == 200


def test_service_starts_without_models(tmp_path, monkeypatch):
    from jobtracker import config

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    c = TestClient(create_app(Models(backend="baseline")))
    assert c.get("/health").json()["status"] == "models_not_loaded"
    assert c.post("/classify", json={"emails": [{"subject": "x"}]}).status_code == 503
