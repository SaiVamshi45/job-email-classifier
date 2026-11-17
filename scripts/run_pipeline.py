"""Classify emails end to end.

  python scripts/run_pipeline.py --gmail-query "after:2025/11/01" --backend baseline
  python scripts/run_pipeline.py --input-csv emails.csv --backend cnn --fallback xlmr

--input-csv needs columns Subject and Email Text (or preprocessed_text); Sender/Date are optional.
Outputs contain no email bodies.
"""
import _bootstrap  # noqa: F401
import argparse
import logging
import re

import pandas as pd

from jobtracker import config, gmail, pipeline
from jobtracker import predictors as P


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gmail-query", help='Gmail search, e.g. "after:2025/11/01"')
    src.add_argument("--input-csv")
    ap.add_argument("--max-messages", type=int, default=1000)
    ap.add_argument("--backend", choices=list(P.BACKENDS), default="baseline")
    ap.add_argument("--fallback", choices=list(P.BACKENDS), default=None, help="used when the primary is unsure")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    emails = (gmail.fetch_emails(args.gmail_query, args.max_messages) if args.gmail_query
              else pd.read_csv(args.input_csv))
    s1, s2 = P.load_predictor("stage1", args.backend), P.load_predictor("stage2", args.backend)
    f1 = P.load_predictor("stage1", args.fallback) if args.fallback else None
    f2 = P.load_predictor("stage2", args.fallback) if args.fallback else None

    result = pipeline.run(emails, s1, s2, f1, f2)
    tag = re.sub(r"[^0-9A-Za-z]+", "-", args.gmail_query or "csv").strip("-")
    paths = pipeline.save(result, tag)
    print(result["category"].value_counts().to_string())
    print(result["predicted_intent"].replace("", pd.NA).value_counts().to_string())
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
