"""Dataset cleaning and leakage-free splitting."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .preprocess import normalize_subject, sender_domain

log = logging.getLogger(__name__)


def clean_raw(df: pd.DataFrame, text_col: str, label_col: str, min_words: int = 3):
    """Turn a raw labelled export into the clean schema: sender_domain, subject, text, label, group.

    Fixes label typos, drops empty/too-short texts, exact duplicates and texts that carry
    conflicting labels. Sender is reduced to its domain so no personal addresses are stored.
    Returns (clean_df, stats).
    """
    stats = {"rows_in": len(df)}
    out = pd.DataFrame(
        {
            "sender_domain": df["Sender"].map(sender_domain) if "Sender" in df else "",
            "subject": df["Subject"].fillna("").astype(str) if "Subject" in df else "",
            "text": df[text_col],
            "label": df[label_col],
        }
    )
    out = out.dropna(subset=["text", "label"])
    out["label"] = out["label"].astype(str).str.strip().replace(config.LABEL_FIXES)
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.split().str.len() >= min_words]
    stats["after_empty_short"] = len(out)

    n_labels = out.groupby("text")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    out = out[~out["text"].isin(conflicting)]
    stats["conflicting_label_texts_dropped"] = len(conflicting)

    before = len(out)
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    stats["exact_duplicates_dropped"] = before - len(out)

    out["group"] = out["subject"].map(normalize_subject)
    empty_group = out["group"] == ""
    out.loc[empty_group, "group"] = "text:" + out.loc[empty_group, "text"].str[:80]
    stats["rows_out"] = len(out)
    stats["class_counts"] = out["label"].value_counts().to_dict()
    return out, stats


def load_stage(stage: str, path=None) -> pd.DataFrame:
    path = path or (config.DATA_DIR / config.STAGES[stage]["data_file"])
    df = pd.read_csv(path)
    missing = {"text", "label", "group"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; run scripts/clean_data.py")
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)


def cap_groups(df: pd.DataFrame, cap: int | None, seed: int = config.SEED) -> pd.DataFrame:
    """Keep at most `cap` rows per (subject template, label). Automatic replies sent hundreds of times (for example
    one DHL template for 54 applications) would otherwise make up most of a class and decide a whole test split."""
    if not cap:
        return df
    shuffled = df.sample(frac=1.0, random_state=seed)
    keep = shuffled.groupby(["group", "label"]).cumcount() < cap
    return shuffled[keep].sort_index().reset_index(drop=True)


def _one_fold(df: pd.DataFrame, seed: int, n_splits: int = 5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, held_idx = next(sgkf.split(df, df["label"], groups=df["group"]))
    return df.iloc[train_idx], df.iloc[held_idx]


def make_splits(df: pd.DataFrame, seed: int = config.SEED):
    """Train / validation / test (~64/16/20) where no group (same subject template) crosses a split."""
    trainval, test = _one_fold(df, seed)
    train, val = _one_fold(trainval.reset_index(drop=True), seed + 1)
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("%s: %d rows, classes=%s", name, len(part), part["label"].value_counts().to_dict())
    assert not (set(train["group"]) & set(test["group"])), "group leakage train/test"
    assert not (set(val["group"]) & set(test["group"])), "group leakage val/test"
    assert not (set(train["group"]) & set(val["group"])), "group leakage train/val"
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def class_weights(y, n_classes: int) -> dict[int, float]:
    counts = pd.Series(y).value_counts()
    total = len(y)
    return {i: total / (n_classes * counts.get(i, 1)) for i in range(n_classes)}
