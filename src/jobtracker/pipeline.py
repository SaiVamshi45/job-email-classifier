"""End-to-end pipeline: emails -> Stage 1 -> Stage 1.5 rules -> Stage 2."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config
from .classify import classify
from .preprocess import build_model_text
from .rules import authenticity

log = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["Date", "Sender", "Subject", "category", "category_score", "authenticity",
                  "predicted_intent", "intent_score"]


def prepare(emails: pd.DataFrame) -> pd.DataFrame:
    """Accept either raw columns (Subject + Email Text) or an already-cleaned `preprocessed_text`."""
    df = emails.copy()
    for col in ("Date", "Sender", "Subject"):
        if col not in df:
            df[col] = ""
    if "preprocessed_text" not in df:
        body = df["Email Text"] if "Email Text" in df else pd.Series([""] * len(df), index=df.index)
        df["preprocessed_text"] = [build_model_text(s, b) for s, b in zip(df["Subject"], body)]
    return df


def run(emails: pd.DataFrame, stage1, stage2, stage1_fallback=None, stage2_fallback=None) -> pd.DataFrame:
    df = prepare(emails).reset_index(drop=True)
    texts = df["preprocessed_text"].fillna("").tolist()

    s1 = classify(texts, stage1, stage1_fallback, config.CNN_FALLBACK_BELOW if stage1_fallback else None)
    df["category"], df["category_score"] = s1["label"].values, s1["score"].values
    log.info("stage 1 fallback rate: %.1f%%", 100 * s1.attrs.get("fallback_rate", 0.0))

    df["authenticity"] = ""
    job = df["category"] == "job_related"
    df.loc[job, "authenticity"] = [
        authenticity(r.Sender, r.Subject, r.preprocessed_text) for r in df[job].itertuples()
    ]

    df["predicted_intent"], df["intent_score"] = "", float("nan")
    real = df["authenticity"] == "real_job_application"
    if real.any():
        s2 = classify(df.loc[real, "preprocessed_text"].tolist(), stage2, stage2_fallback,
                      config.CNN_FALLBACK_BELOW if stage2_fallback else None, config.STAGE2_ABSTAIN_BELOW)
        df.loc[real, "predicted_intent"] = s2["label"].values
        df.loc[real, "intent_score"] = s2["score"].values
    return df


def save(df: pd.DataFrame, tag: str, out_dir: Path = config.OUTPUT_DIR) -> dict[str, Path]:
    """Write result files WITHOUT email bodies, so outputs contain no message content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": out_dir / f"results_{tag}.csv",
        "applications": out_dir / f"applications_{tag}.csv",
        "filtered": out_dir / f"filtered_out_{tag}.csv",
    }
    df[OUTPUT_COLUMNS].to_csv(paths["all"], index=False, encoding="utf-8-sig")
    df[df["authenticity"] == "real_job_application"][OUTPUT_COLUMNS].to_csv(
        paths["applications"], index=False, encoding="utf-8-sig")
    df[(df["category"] == "job_related") & (df["authenticity"] != "real_job_application")][OUTPUT_COLUMNS].to_csv(
        paths["filtered"], index=False, encoding="utf-8-sig")
    return paths
