"""Small HTTP API around the two-stage pipeline (FastAPI).

  uvicorn jobtracker.api:app --host 0.0.0.0 --port 8000      (from the src/ folder, or with PYTHONPATH=src)

Environment variables
  JOBTRACKER_BACKEND   baseline (default) | cnn | xlmr   which trained model to serve
  JOBTRACKER_API_KEY   optional; when set, every /classify call must send it as header X-API-Key

The service takes text in the request body. It never logs email content and never talks to Gmail.
"""
from __future__ import annotations

import logging
import os
import secrets

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import config, pipeline

log = logging.getLogger("jobtracker.api")
MAX_BATCH = 50


class EmailIn(BaseModel):
    subject: str = Field("", max_length=500)
    body: str = Field("", max_length=20000)
    sender: str = Field("", max_length=300)


class BatchIn(BaseModel):
    emails: list[EmailIn] = Field(..., min_length=1, max_length=MAX_BATCH)


class ResultOut(BaseModel):
    category: str
    category_score: float
    authenticity: str
    intent: str
    intent_score: float | None


class Models:
    """Holds the two predictors; loaded once at start-up (or injected in tests)."""

    def __init__(self, stage1=None, stage2=None, backend: str | None = None):
        self.backend = backend or os.environ.get("JOBTRACKER_BACKEND", "baseline")
        self.stage1, self.stage2, self.error = stage1, stage2, None
        if stage1 is None or stage2 is None:
            try:
                from . import predictors as P

                self.stage1 = P.load_predictor("stage1", self.backend)
                self.stage2 = P.load_predictor("stage2", self.backend)
            except Exception as exc:  # models missing: the service still starts, /health says why
                self.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning("models not loaded: %s", self.error)

    @property
    def ready(self) -> bool:
        return self.stage1 is not None and self.stage2 is not None


def create_app(models: Models | None = None) -> FastAPI:
    app = FastAPI(title="JobTracker email classifier", version="1.0")
    state = models if models is not None else Models()

    def check_key(x_api_key: str | None = Header(default=None)) -> None:
        wanted = os.environ.get("JOBTRACKER_API_KEY")
        if wanted and not secrets.compare_digest(x_api_key or "", wanted):
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok" if state.ready else "models_not_loaded", "backend": state.backend,
                "detail": state.error}

    @app.post("/classify", response_model=list[ResultOut], dependencies=[Depends(check_key)])
    def classify_batch(batch: BatchIn) -> list[ResultOut]:
        if not state.ready:
            raise HTTPException(status_code=503, detail="models are not loaded; see /health")
        frame = pd.DataFrame({
            "Sender": [e.sender for e in batch.emails],
            "Subject": [e.subject for e in batch.emails],
            "Email Text": [e.body[: config.MAX_BODY_CHARS] for e in batch.emails],
        })
        res = pipeline.run(frame, state.stage1, state.stage2)
        return [ResultOut(category=r.category, category_score=float(r.category_score),
                          authenticity=r.authenticity, intent=r.predicted_intent,
                          intent_score=None if pd.isna(r.intent_score) else float(r.intent_score))
                for r in res.itertuples()]

    return app


app = create_app()
