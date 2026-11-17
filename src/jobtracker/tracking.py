"""Optional experiment tracking with MLflow (free, local). Does nothing when mlflow is not installed.

Runs are stored in `mlflow.db` + `mlruns/` in the project folder (both git-ignored). View them with
  mlflow ui --backend-store-uri sqlite:///mlflow.db
Only numbers, label names and the confusion-matrix picture are logged: never email text and never model files
(they are derived from private mail).
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
EXPERIMENT = "jobtracker"


def available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"


def _flat_metrics(result: dict, prefix: str) -> dict[str, float]:
    out = {f"{prefix}accuracy": result["accuracy"], f"{prefix}macro_f1": result["macro_f1"]}
    if result.get("macro_auc_ovr") is not None:
        out[f"{prefix}macro_auc"] = result["macro_auc_ovr"]
    for label, row in result.get("per_class", {}).items():
        if isinstance(row, dict) and label in result.get("labels", []):
            out[f"{prefix}f1_{label}"] = float(row["f1-score"])
            out[f"{prefix}recall_{label}"] = float(row["recall"])
    return out


@contextlib.contextmanager
def run(name: str, tags: dict | None = None):
    """Yield an active MLflow run, or None when tracking is unavailable. Tracking errors never stop training."""
    if not available():
        yield None
        return
    import mlflow

    try:
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(EXPERIMENT)
        ctx = mlflow.start_run(run_name=name, tags=tags or {})
    except Exception as exc:  # e.g. a locked database
        log.warning("MLflow disabled for this run: %s", exc)
        yield None
        return
    with ctx as active:
        yield active


def log_training(stage: str, backend: str, params: dict, result: dict, artifacts: list[Path] = ()) -> None:
    with run(f"{stage}/{backend}", {"kind": "train", "stage": stage, "backend": backend}) as active:
        if active is None:
            return
        import mlflow

        mlflow.log_params({k: (v if not isinstance(v, (dict, list)) else str(v)) for k, v in params.items()})
        mlflow.log_metrics(_flat_metrics(result, "test_"))
        for path in artifacts:
            if Path(path).exists():
                mlflow.log_artifact(str(path))


def log_gold(stage: str, backend: str, accuracy: float, macro_f1: float, n: int, gold_file: str) -> None:
    with run(f"gold/{stage}/{backend}", {"kind": "gold", "stage": stage, "backend": backend}) as active:
        if active is None:
            return
        import mlflow

        mlflow.log_params({"stage": stage, "backend": backend, "gold_rows": n, "gold_file": Path(gold_file).name})
        mlflow.log_metrics({"gold_accuracy": accuracy, "gold_macro_f1": macro_f1})
