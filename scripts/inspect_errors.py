"""Show which test rows a Stage-2 model gets wrong, grouped by subject template (prints to the console only).

  python scripts/inspect_errors.py --backend baseline --true acknowledgment
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from jobtracker import config
from jobtracker import predictors as P
from jobtracker.data import cap_groups, load_stage, make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="stage2")
ap.add_argument("--backend", default="baseline")
ap.add_argument("--true", default="acknowledgment", help="true label to inspect")
ap.add_argument("--group-cap", type=int, default=config.GROUP_CAP)
ap.add_argument("--top", type=int, default=15)
args = ap.parse_args()

df = cap_groups(load_stage(args.stage), args.group_cap, config.SEED)
_, _, test = make_splits(df, config.SEED)
pred = P.load_predictor(args.stage, args.backend)
test = test.reset_index(drop=True)
test["pred"] = [pred.labels[i] for i in pred.predict_proba(test["text"]).argmax(1)]
print(pd.crosstab(test["label"], test["pred"]))
bad = test[(test["label"] == args.true) & (test["pred"] != args.true)]
print(f"\n{len(bad)} of {(test['label'] == args.true).sum()} '{args.true}' test rows predicted as something else")
print("source of these rows:", bad["source"].value_counts().to_dict() if "source" in bad else "n/a")
print(bad.groupby(["group", "pred"]).size().sort_values(ascending=False).head(args.top).to_string())
