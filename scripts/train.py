"""Train and evaluate one stage.

  python scripts/train.py --stage stage1 --backend baseline
  python scripts/train.py --stage stage2 --backend baseline cnn xlmr     # needs requirements-neural.txt

Splits are grouped by subject template so near-identical emails never straddle train/test.
Each model is saved, reloaded from disk and evaluated on the held-out test split, so the
numbers describe exactly what inference will use.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import logging

from jobtracker import config
from jobtracker import predictors as P
from jobtracker import tracking
from jobtracker.data import cap_groups, load_stage, make_splits
from jobtracker.evaluate import evaluate, plot_confusion, save_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(config.STAGES), required=True)
    ap.add_argument("--backend", nargs="+", choices=list(P.BACKENDS), default=["baseline"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--group-cap", type=int, default=config.GROUP_CAP,
                    help="max rows per subject template and label (0 = no cap)")
    ap.add_argument("--cpu", action="store_true", help="force CPU for XLM-R")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = config.STAGES[args.stage]
    df = load_stage(args.stage)
    rows_before = len(df)
    df = cap_groups(df, args.group_cap, args.seed)
    print(f"{args.stage}: {rows_before} rows, {len(df)} after capping each subject template at {args.group_cap}")
    labels = sorted(df["label"].unique())
    train, val, test = make_splits(df, args.seed)
    y_test = test["label"].map({l: i for i, l in enumerate(labels)}).to_numpy()
    report = {"stage": args.stage, "labels": labels,
              "split_sizes": {"train": len(train), "val": len(val), "test": len(test)}, "models": {}}

    for backend in args.backend:
        out = P.model_dir(args.stage, backend)
        print(f"\n=== {args.stage} / {backend} ===")
        if backend == "baseline":
            P.train_baseline(train, labels, out)
        elif backend == "cnn":
            P.train_cnn(train, val, labels, out, cfg["cnn_maxlen"], cfg["cnn_max_words"], seed=args.seed)
        elif backend == "xlmr":
            P.train_xlmr(train, val, labels, out, cfg["xlmr_maxlen"], epochs=cfg["xlmr_epochs"],
                         seed=args.seed, use_cpu=args.cpu)
        predictor = P.load_predictor(args.stage, backend)
        assert predictor.labels == labels
        res = evaluate(y_test, predictor.predict_proba(test["text"]), labels)
        report["models"][backend] = res
        plot_confusion(res["confusion_matrix"], labels, config.REPORTS_DIR / f"{args.stage}_{backend}_cm.png",
                       f"{args.stage} / {backend}")
        print(f"test macro-F1={res['macro_f1']:.3f}  accuracy={res['accuracy']:.3f}  macro-AUC={res['macro_auc_ovr']}")
        tracking.log_training(
            args.stage, backend,
            {"seed": args.seed, "data_file": cfg["data_file"], "n_rows": len(df), "rows_before_cap": rows_before, "group_cap": args.group_cap, "n_train": len(train),
             "n_val": len(val), "n_test": len(test), "labels": labels,
             "train_class_counts": train["label"].value_counts().to_dict()},
            res, [config.REPORTS_DIR / f"{args.stage}_{backend}_cm.png", out / "labels.json"])

    path = config.REPORTS_DIR / f"{args.stage}_report.json"
    if path.exists():  # keep results of backends trained in earlier runs
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("labels") == labels:
            report["models"] = {**previous.get("models", {}), **report["models"]}
    save_report(path, report)
    print(f"\nreport -> {config.REPORTS_DIR / (args.stage + '_report.json')}")


if __name__ == "__main__":
    main()
