"""Turn your own mailbox into a review queue of pre-labelled training examples.

  python scripts/build_dataset.py --gmail-query "after:2025/08/01 -in:sent" --max-messages 3000
  python scripts/build_dataset.py --input-csv emails.csv            # same, from a CSV (Subject, Email Text, Sender)

What it does
1. Fetches the mails (first 1500 characters of each body, the same length the models are trained on).
2. Pre-labels them with the weak labellers (weak_labels.py) and, when a trained model exists, with the model.
3. Writes data/review_queue.csv sorted so the mails that most need a human come first:
   weak and model disagree  >  weak labeller unsure  >  nothing matched  >  everything agrees.
4. You open the CSV in Excel / Sheets and fill ONLY the columns `human_category` and `human_intent`
   for rows you want to correct or confirm (leave empty to accept an agreed, high-confidence label).
5. Run scripts/merge_reviewed.py to add the result to data/stage1_clean.csv and data/stage2_clean.csv.

Reviewing the first 150-200 rows fixes most of the label noise; the rest of the file can be accepted as is.
data/ is git-ignored: the queue contains email text.
"""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd

from jobtracker import config, gmail
from jobtracker import predictors as P
from jobtracker.preprocess import build_model_text, normalize_subject, sender_domain
from jobtracker.weak_labels import weak_label_row


def model_predictions(texts, stage: str, backend: str):
    try:
        pred = P.load_predictor(stage, backend)
    except Exception:  # model not trained yet
        return None, None
    proba = pred.predict_proba(texts)
    return [pred.labels[i] for i in proba.argmax(1)], proba.max(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gmail-query")
    src.add_argument("--input-csv")
    ap.add_argument("--max-messages", type=int, default=3000)
    ap.add_argument("--backend", default="baseline", choices=list(P.BACKENDS), help="model used for disagreement checks")
    ap.add_argument("--out", default=str(config.DATA_DIR / "review_queue.csv"))
    args = ap.parse_args()

    df = gmail.fetch_emails(args.gmail_query, args.max_messages) if args.gmail_query else pd.read_csv(args.input_csv)
    for col in ("Id", "Date", "Sender", "Subject", "Email Text"):
        if col not in df:
            df[col] = ""
    df["Email Text"] = df["Email Text"].fillna("").astype(str).str[: config.MAX_BODY_CHARS]
    df["text"] = [build_model_text(s, b) for s, b in zip(df["Subject"], df["Email Text"])]
    df = df[df["text"].str.split().str.len() >= 3].drop_duplicates(subset="text").reset_index(drop=True)

    weak = pd.DataFrame([weak_label_row(r.Sender, r.Subject, r.text) for r in df.itertuples()])
    out = pd.concat([df[["Id", "Date"]], weak], axis=1)
    out.insert(2, "sender_domain", df["Sender"].map(sender_domain))
    out.insert(3, "subject", df["Subject"])
    out["text"] = df["text"]
    out["group"] = df["Subject"].map(normalize_subject)

    m1, c1 = model_predictions(out["text"], "stage1", args.backend)
    m2, c2 = model_predictions(out["text"], "stage2", args.backend)
    out["model_category"] = m1 if m1 is not None else ""
    out["model_intent"] = m2 if m2 is not None else ""

    cat_disagree = (out["weak_category"].notna() & (out["model_category"] != "") & (out["weak_category"] != out["model_category"]))
    int_disagree = (out["weak_intent"].notna() & (out["model_intent"] != "") & (out["weak_intent"] != out["model_intent"])
                    & (out["weak_authenticity"] == "real_job_application"))
    unsure = (out["weak_category_conf"] != "high") | ((out["weak_authenticity"] == "real_job_application") & (out["weak_intent_conf"] != "high"))
    # lower number = review first
    out["review_priority"] = 3
    out.loc[unsure, "review_priority"] = 2
    out.loc[cat_disagree | int_disagree, "review_priority"] = 1
    out["human_category"] = ""
    out["human_intent"] = ""
    out = out.sort_values(["review_priority", "group"]).reset_index(drop=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"{len(out)} mails -> {args.out}")
    print("review priority (1 = most urgent):", out["review_priority"].value_counts().sort_index().to_dict())
    print("weak categories:", out["weak_category"].fillna("none").value_counts().to_dict())
    print("weak intents:", out["weak_intent"].fillna("none").value_counts().to_dict())


if __name__ == "__main__":
    main()
