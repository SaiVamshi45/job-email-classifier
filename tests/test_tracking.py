import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jobtracker import tracking

RESULT = {"accuracy": 0.9, "macro_f1": 0.8, "macro_auc_ovr": None, "labels": ["a", "b"],
          "per_class": {"a": {"f1-score": 0.9, "recall": 0.8}, "b": {"f1-score": 0.7, "recall": 0.6},
                        "macro avg": {"f1-score": 0.8, "recall": 0.7}}}


def test_flat_metrics_skips_missing_auc_and_averages():
    m = tracking._flat_metrics(RESULT, "test_")
    assert m["test_f1_a"] == 0.9 and m["test_recall_b"] == 0.6
    assert "test_macro_auc" not in m and not any("macro avg" in k for k in m)


def test_noop_without_mlflow(monkeypatch):
    monkeypatch.setattr(tracking, "available", lambda: False)
    tracking.log_training("stage1", "baseline", {"seed": 1}, RESULT)   # must not raise
    tracking.log_gold("stage1", "baseline", 0.9, 0.8, 10, "gold.csv")


def test_logs_a_run_when_mlflow_is_installed(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    tracking.log_training("stage2", "baseline", {"seed": 7, "labels": ["a", "b"]}, RESULT)
    mlflow.set_tracking_uri(tracking.tracking_uri())
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.test_macro_f1"] == pytest.approx(0.8)
    assert runs.iloc[0]["params.seed"] == "7"
