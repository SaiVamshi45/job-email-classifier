"""Combine a primary model with an optional fallback and an abstain threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify(texts, primary, fallback=None, fallback_below: float | None = None,
             abstain_below: float | None = None) -> pd.DataFrame:
    """Return a frame with label, score and source ("primary" / "fallback" / "abstain")."""
    texts = list(texts)
    if not texts:
        return pd.DataFrame(columns=["label", "score", "source"])
    labels = np.array(primary.labels, dtype=object)
    proba = primary.predict_proba(texts)
    idx = proba.argmax(axis=1)
    out = pd.DataFrame({"label": labels[idx], "score": proba.max(axis=1).astype("float64"), "source": "primary"})

    if fallback is not None and fallback_below is not None:
        if list(fallback.labels) != list(primary.labels):
            raise ValueError("primary and fallback models were trained with different label sets")
        weak = np.flatnonzero(out["score"].to_numpy() < fallback_below)
        if len(weak):
            fp = fallback.predict_proba([texts[i] for i in weak])
            out.loc[weak, "label"] = labels[fp.argmax(axis=1)]
            out.loc[weak, "score"] = fp.max(axis=1).astype("float64")
            out.loc[weak, "source"] = "fallback"
    if abstain_below is not None:
        unsure = out["score"] < abstain_below
        out.loc[unsure, "label"] = "uncertain"
        out.loc[unsure, "source"] = "abstain"
    out.attrs["fallback_rate"] = float((out["source"] == "fallback").mean())
    return out
