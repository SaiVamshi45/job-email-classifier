"""Find (and optionally fix) probably-wrong labels in the existing training files.

  python scripts/audit_labels.py                 # writes data/label_audit_stage1.csv and ..._stage2.csv
  python scripts/audit_labels.py --apply         # applies the `human` column (or `--accept-suggested`) back to the data

A row is listed when the independent weak labeller is HIGH confidence and disagrees with the stored label.
Open the CSV, look at `text`, and write the correct label into `human` (or leave empty to keep the old label).
With --accept-suggested, rows with an empty `human` take the suggestion, which is only sensible after you spot-checked
a sample (see the audit precision printed by this script when you fill the `human` column for ~30 rows).
"""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd

from jobtracker import config
from jobtracker.weak_labels import weak_category, weak_intent

STAGES = {"stage1": "category", "stage2": "intent"}


def suggest(stage: str, df: pd.DataFrame) -> pd.Series:
    if stage == "stage1":
        res = [weak_category("x@" + d, s, t) for d, s, t in zip(df["sender_domain"].fillna(""), df["subject"].fillna(""), df["text"])]
    else:
        res = [weak_intent(s, t) for s, t in zip(df["subject"].fillna(""), df["text"])]
    return pd.Series([p if c == "high" else None for p, c in res], index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--accept-suggested", action="store_true")
    ap.add_argument("--stage", nargs="+", choices=list(STAGES), help="limit to some stages (default: both)")
    args = ap.parse_args()

    for stage in args.stage or STAGES:
        data_path = config.DATA_DIR / config.STAGES[stage]["data_file"]
        audit_path = config.DATA_DIR / f"label_audit_{stage}.csv"
        df = pd.read_csv(data_path)
        if not args.apply:
            sug = suggest(stage, df)
            bad = df[sug.notna() & (sug != df["label"])].copy()
            bad.insert(0, "row", bad.index)
            bad["suggested"] = sug[bad.index]
            bad["human"] = ""
            bad[["row", "label", "suggested", "human", "subject", "text"]].sort_values(["label", "suggested"]).to_csv(
                audit_path, index=False, encoding="utf-8-sig")
            print(f"{stage}: {len(bad)} of {len(df)} rows disagree with a high-confidence weak label -> {audit_path}")
            print(pd.crosstab(bad["label"], bad["suggested"]).to_string())
        else:
            a = pd.read_csv(audit_path).fillna("")
            new = a["human"].where(a["human"] != "", a["suggested"] if args.accept_suggested else "")
            changed = 0
            for row, lab in zip(a["row"], new):
                if lab and df.loc[row, "label"] != lab:
                    df.loc[row, "label"] = lab
                    changed += 1
            df.to_csv(data_path, index=False)
            print(f"{stage}: {changed} labels corrected in {data_path}")
            filled = a[a["human"] != ""]
            if len(filled):
                print(f"   audit precision (how often your correction equalled the suggestion): "
                      f"{(filled['human'] == filled['suggested']).mean():.2f} on {len(filled)} reviewed rows")


if __name__ == "__main__":
    main()
