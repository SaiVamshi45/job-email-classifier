"""Evaluate rules, weak labellers and trained models on the hand-labelled gold set.

  python scripts/eval_gold.py                       # rules + weak labellers only
  python scripts/eval_gold.py --backend baseline cnn

The gold set (data/gold_seed.csv) holds recent real emails from the user's own mailbox, labelled by hand.
Texts are subject + Gmail snippet (about 200 characters), so models trained on 1500-character bodies are
tested on shorter input than usual: treat model numbers as a lower bound.
"""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from jobtracker import config
from jobtracker import predictors as P
from jobtracker import tracking
from jobtracker.preprocess import build_model_text
from jobtracker.rules import authenticity
from jobtracker.weak_labels import weak_label_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", nargs="*", default=[], choices=list(P.BACKENDS))
    ap.add_argument("--gold", default=str(config.DATA_DIR / "gold_seed.csv"))
    args = ap.parse_args()

    g = pd.read_csv(args.gold).fillna("")
    g["text"] = [build_model_text(s, t) for s, t in zip(g["subject"], g["snippet"])]
    g["sender"] = "x@" + g["sender_domain"]
    print(f"gold rows: {len(g)}   stage1: {g.stage1.value_counts().to_dict()}   intents: {g[g.intent != ''].intent.value_counts().to_dict()}")

    # Stage 1.5 rules on the rows that are truly job-related
    jr = g[g.stage1 == "job_related"]
    pred = [authenticity(r.sender, r.subject, r.text) for r in jr.itertuples()]
    print(f"\n[Stage 1.5 rules] accuracy {accuracy_score(jr.stage15, pred):.3f} on {len(jr)} job-related mails")
    wrong = jr.assign(pred=pred)[lambda d: d.pred != d.stage15]
    for r in wrong.itertuples():
        print(f"   gold={r.stage15:<22} pred={r.pred:<22} {r.subject[:70]}")

    # weak labellers
    w = pd.DataFrame([weak_label_row(r.sender, r.subject, r.text) for r in g.itertuples()])
    has = w.weak_category.notna()
    print(f"\n[Weak category] coverage {has.mean():.2f}  accuracy on covered {accuracy_score(g.stage1[has], w.weak_category[has]):.3f}")
    hi = has & (w.weak_category_conf == "high")
    print(f"   high-confidence only: coverage {hi.mean():.2f}  accuracy {accuracy_score(g.stage1[hi], w.weak_category[hi]):.3f}")
    gi = g[g.intent != ""].reset_index(drop=True)
    wi = pd.DataFrame([weak_label_row(r.sender, r.subject, r.text) for r in gi.itertuples()])
    from jobtracker.weak_labels import weak_intent
    wi_pred = [weak_intent(r.subject, r.text) for r in gi.itertuples()]
    got = [(p, c) for p, c in wi_pred]
    cov = [p is not None for p, _ in got]
    print(f"[Weak intent] coverage {sum(cov)/len(gi):.2f}  accuracy on covered "
          f"{accuracy_score([a for a, c in zip(gi.intent, cov) if c], [p for (p, _), c in zip(got, cov) if c]):.3f}")
    hc = [c == "high" for _, c in got]
    print(f"   high-confidence only: coverage {sum(hc)/len(gi):.2f}  accuracy "
          f"{accuracy_score([a for a, c in zip(gi.intent, hc) if c], [p for (p, _), c in zip(got, hc) if c]):.3f}")

    for backend in args.backend:
        print(f"\n=== models: {backend} ===")
        import datetime as _dt
        for _st in ("stage1", "stage2"):
            _files = [f for f in P.model_dir(_st, backend).glob("*") if f.is_file()]
            _t = max(f.stat().st_mtime for f in _files)
            print(f"  {_st} model files last written {_dt.datetime.fromtimestamp(_t):%H:%M:%S} ({len(_files)} files)")
        s1 = P.load_predictor("stage1", backend)
        p1 = [s1.labels[i] for i in s1.predict_proba(g["text"]).argmax(1)]
        print(f"[Stage 1] accuracy {accuracy_score(g.stage1, p1):.3f}  macro-F1 {f1_score(g.stage1, p1, average='macro'):.3f}")
        print(classification_report(g.stage1, p1, zero_division=0))
        tracking.log_gold("stage1", backend, accuracy_score(g.stage1, p1), f1_score(g.stage1, p1, average="macro"),
                          len(g), args.gold)
        s2 = P.load_predictor("stage2", backend)
        print(f"  stage2 labels {s2.labels}; mean top probability on gold {s2.predict_proba(gi['text']).max(1).mean():.4f}")
        p2 = [s2.labels[i] for i in s2.predict_proba(gi["text"]).argmax(1)]
        print(f"[Stage 2] accuracy {accuracy_score(gi.intent, p2):.3f}  macro-F1 {f1_score(gi.intent, p2, average='macro'):.3f}")
        print(classification_report(gi.intent, p2, zero_division=0))
        tracking.log_gold("stage2", backend, accuracy_score(gi.intent, p2), f1_score(gi.intent, p2, average="macro"),
                          len(gi), args.gold)


if __name__ == "__main__":
    main()
