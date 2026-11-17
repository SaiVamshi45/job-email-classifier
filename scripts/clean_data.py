"""Clean the raw labelled exports into data/stage1_clean.csv and data/stage2_clean.csv.

Usage:
  python scripts/clean_data.py --stage1 raw/Stage1_ideal_balanced_final.csv --stage2 raw/stage2_final_clean.csv

Removes exact duplicates and conflicting labels, fixes label typos, and reduces Sender to a
domain. Prints a summary so you can see how much of the original data was duplicated.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import pandas as pd

from jobtracker import config
from jobtracker.data import clean_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    args = ap.parse_args()
    config.DATA_DIR.mkdir(exist_ok=True)

    raw1 = pd.read_csv(args.stage1)
    c1, s1 = clean_raw(raw1, "preprocessed_text", "category")
    c1.to_csv(config.DATA_DIR / "stage1_clean.csv", index=False)

    raw2 = pd.read_csv(args.stage2)
    c2, s2 = clean_raw(raw2, "preprocessed_text", "intent_label")
    c2.to_csv(config.DATA_DIR / "stage2_clean.csv", index=False)

    print(json.dumps({"stage1": s1, "stage2": s2}, indent=2))


if __name__ == "__main__":
    main()
