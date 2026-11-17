"""Show how much the original evaluation protocol inflated the scores.

Trains the same TF-IDF + logistic-regression model on the RAW Stage-1 file twice:
  A) original protocol: random stratified 80/20 split, duplicates kept
  B) fixed protocol: duplicates removed, split grouped by subject template

  python scripts/leakage_check.py --raw raw/Stage1_ideal_balanced_final.csv --label category
"""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from jobtracker.data import clean_raw, make_splits


def fit_score(tr_x, tr_y, te_x, te_y) -> float:
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))
    pipe.fit(tr_x, tr_y)
    return f1_score(te_y, pipe.predict(te_x), average="macro")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--label", default="category")
    ap.add_argument("--text", default="preprocessed_text")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    orig = raw.dropna(subset=[args.text, args.label])
    a = []
    for seed in (1, 2, 3):
        x_tr, x_te, y_tr, y_te = train_test_split(
            orig[args.text], orig[args.label], stratify=orig[args.label], test_size=0.2, random_state=seed)
        a.append(fit_score(x_tr, y_tr, x_te, y_te))
    clean, stats = clean_raw(raw, args.text, args.label)
    b = []
    for seed in (1, 2, 3):
        tr, _, te = make_splits(clean, seed)
        b.append(fit_score(tr["text"], tr["label"], te["text"], te["label"]))

    print(f"rows {stats['rows_in']} -> {stats['rows_out']}  (exact duplicates dropped: {stats['exact_duplicates_dropped']})")
    print(f"A) original protocol  macro-F1 = {sum(a)/3:.3f}  {['%.3f' % x for x in a]}")
    print(f"B) fixed protocol     macro-F1 = {sum(b)/3:.3f}  {['%.3f' % x for x in b]}")


if __name__ == "__main__":
    main()
