"""Evaluation helpers: macro metrics, per-class report, confusion matrix, optional plot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def evaluate(y_true, proba: np.ndarray, labels: list[str]) -> dict:
    y_true = np.asarray(y_true)
    y_pred = proba.argmax(axis=1)
    ids = list(range(len(labels)))
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=ids, target_names=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "labels": labels,
    }
    try:
        result["macro_auc_ovr"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=ids)
        )
    except ValueError:  # a class missing from y_true
        result["macro_auc_ovr"] = None
    return result


def save_report(path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_confusion(cm, labels, path, title="Confusion matrix") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # plotting is optional
        return
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
