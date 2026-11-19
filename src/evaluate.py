"""Shared metrics, confusion-matrix plot and result logging used by every notebook."""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)

from src.config import RESULTS


def evaluate(y_true, y_pred, y_score=None, labels=None):
    """Return (metrics, per_class). y_score columns must follow `labels` (sklearn: labels=clf.classes_)."""
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # One-vs-rest counts per class from the confusion matrix (rows = true, columns = predicted).
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / (tp + fp))
        recall = np.nan_to_num(tp / (tp + fn))  # TP rate / sensitivity
        fpr = np.nan_to_num(fp / (fp + tn))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

    per_class = pd.DataFrame({"precision": precision, "recall_tpr": recall, "fpr": fpr,
                              "f1": f1, "support": cm.sum(axis=1)}, index=labels)
    # Macro averages: every genre counts equally, so a weak genre can't hide behind strong ones.
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
        "macro_fpr": fpr.mean(),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None:
        metrics["macro_auc_ovr"] = roc_auc_score(y_true, y_score, multi_class="ovr",
                                                 average="macro", labels=labels)
    return {k: round(float(v), 4) for k, v in metrics.items()}, per_class.round(4)


def plot_confusion(y_true, y_pred, labels, title=""):
    # Row-normalised, so each row shows the recall for one true genre.
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, normalize="true",
                                            values_format=".2f", cmap="Blues", colorbar=False,
                                            xticks_rotation=45, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def save_result(name, pipeline, metrics, params=None, split="test", notes=""):
    # One JSON per experiment (results/<pipeline>/<name>.json) so teammates never hit merge conflicts.
    out = RESULTS / pipeline
    out.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "pipeline": pipeline, "split": split, "metrics": metrics,
              "params": params or {}, "notes": notes, "saved": datetime.now().isoformat(timespec="seconds")}
    (out / f"{name}.json").write_text(json.dumps(record, indent=2, default=str))


def results_table(pipeline, split="test"):
    rows = []
    for f in sorted((RESULTS / pipeline).glob("*.json")):
        r = json.loads(f.read_text())
        if r["split"] == split:
            rows.append({"model": r["name"], **r["metrics"]})
    return pd.DataFrame(rows).set_index("model").sort_values("macro_f1", ascending=False)


if __name__ == "__main__":
    # Self-test: python -m src.evaluate
    y = np.array(["a", "b", "c", "d"] * 25)
    m, pc = evaluate(y, y)
    assert m["accuracy"] == 1 and m["mcc"] == 1 and m["macro_f1"] == 1 and m["macro_fpr"] == 0
    zeror = np.full_like(y, "a")  # majority-class baseline on 4 balanced classes
    m, pc = evaluate(y, zeror, y_score=np.tile([1, 0, 0, 0], (len(y), 1)), labels=list("abcd"))
    assert m["accuracy"] == 0.25 and m["macro_recall"] == 0.25 and m["macro_auc_ovr"] == 0.5
    assert pc.loc["a", "recall_tpr"] == 1 and pc.loc["a", "fpr"] == 1 and pc.loc["b", "recall_tpr"] == 0
    print("evaluate.py self-check passed")
