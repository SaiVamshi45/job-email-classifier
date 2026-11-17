"""One-off cleanup: drop mailbox-reviewed training rows whose subject template also appears in the gold set.

  python scripts/remove_gold_from_training.py
Rows from the original dataset are kept. Re-train afterwards.
"""
import _bootstrap  # noqa: F401
import pandas as pd

from jobtracker import config
from jobtracker.preprocess import normalize_subject

gold = pd.read_csv(config.DATA_DIR / "gold_seed.csv").fillna("")
gold_subjects = {normalize_subject(x) for x in gold["subject"]}
for name in ("stage1", "stage2"):
    path = config.DATA_DIR / config.STAGES[name]["data_file"]
    df = pd.read_csv(path)
    hit = df["subject"].fillna("").map(normalize_subject).isin(gold_subjects)
    if "source" in df:
        hit &= df["source"] == "mailbox_review"
    df[~hit].to_csv(path, index=False)
    print(f"{name}: removed {int(hit.sum())} rows matching gold subjects -> {int((~hit).sum())} left")