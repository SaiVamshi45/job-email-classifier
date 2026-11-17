"""Merge a reviewed queue into the training files.

  python scripts/merge_reviewed.py --queue data/review_queue.csv [--accept-unreviewed]

Label rule per row:
  human_category / human_intent filled      -> use it (always trusted)
  empty, review_priority == 3 (all agree)  -> use the weak label, only with --accept-unreviewed
  anything else                             -> skipped

Stage 1 gets every labelled row (category). Stage 2 gets rows that are real job applications
(weak_authenticity == real_job_application, or you filled human_intent) with an intent.
Existing rows are kept; duplicates (same text) are removed, newer reviewed labels win.
"""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd

from jobtracker import config
from jobtracker.preprocess import normalize_subject

STAGE1_LABELS = {"job_related", "job_agency_spam", "promotional", "non_job"}
STAGE2_LABELS = {"acknowledgment", "interview", "next_step", "rejection"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(config.DATA_DIR / "review_queue.csv"))
    ap.add_argument("--accept-unreviewed", action="store_true",
                    help="also take agreed, high-confidence weak labels you did not touch")
    args = ap.parse_args()

    q = pd.read_csv(args.queue).fillna("")
    # The gold set is the independent test; the same mails sit in the mailbox, so keep them (by id and by
    # subject template) out of the training files.
    gold_path = config.DATA_DIR / "gold_seed.csv"
    if gold_path.exists():
        gold = pd.read_csv(gold_path).fillna("")
        gold_subjects = {normalize_subject(x) for x in gold["subject"]}
        in_gold = q["Id"].astype(str).isin(set(gold["id"].astype(str))) | q["subject"].map(normalize_subject).isin(gold_subjects)
        print(f"excluded {int(in_gold.sum())} queue rows that match the gold set")
        q = q[~in_gold].reset_index(drop=True)
    auto = (q["review_priority"] == 3) & args.accept_unreviewed

    cat = q["human_category"].where(q["human_category"] != "", q["weak_category"].where(auto, ""))
    intent = q["human_intent"].where(q["human_intent"] != "", q["weak_intent"].where(auto, ""))
    bad = (set(cat) - STAGE1_LABELS - {""}) | (set(intent) - STAGE2_LABELS - {""})
    if bad:
        raise SystemExit(f"unknown labels in queue: {sorted(bad)}")

    base = {"sender_domain": q["sender_domain"], "subject": q["subject"], "text": q["text"], "group": q["group"],
            "source": "mailbox_review"}
    s1 = pd.DataFrame({**base, "label": cat})
    s1 = s1[s1["label"] != ""]
    is_app = (q["weak_authenticity"] == "real_job_application") | (q["human_intent"] != "")
    s2 = pd.DataFrame({**base, "label": intent})[is_app & (intent != "")]

    for name, new in (("stage1", s1), ("stage2", s2)):
        path = config.DATA_DIR / config.STAGES[name]["data_file"]
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=new.columns)
        if "source" not in old:
            old["source"] = "original"
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset="text", keep="last").reset_index(drop=True)
        merged.to_csv(path, index=False)
        print(f"{name}: +{len(new)} new rows -> {len(merged)} total   {merged['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()