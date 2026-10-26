"""Offline evaluation with exact input identity and full candidate coverage."""
import math
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from .common import load, sha, validate_input, validate_predictions, freeze


def evaluate(input_path, labels_path, run_path, out):
    item = validate_input(load(input_path))
    labels = load(labels_path)
    run = Path(run_path)
    receipt = load(run / "receipt.json")
    expected = sha(input_path)
    if labels["input_sha256"] != expected or receipt["input_sha256"] != expected:
        raise ValueError("Inputs do not match: never mix seed=1 and seed=42 or different cohorts")
    if receipt["status"] != "complete" or receipt["smoke_only"]:
        raise ValueError("Incomplete/smoke run cannot be evaluated as N250")
    if receipt["predictions_sha256"] != sha(run / "predictions.json"):
        raise ValueError("Prediction artifact changed")
    by_id = validate_predictions(item, load(run / "predictions.json"))
    gold = {r["id"]: r["label"] for r in labels["labels"]}
    if len(gold) != len(labels["labels"]) or set(gold) != set(by_id):
        raise ValueError("Label coverage mismatch or duplicate label id")
    ids = [c["id"] for c in item["candidates"]]
    x = np.asarray([by_id[i]["delta_fitness"] for i in ids], dtype=float)
    y = np.asarray([gold[i] for i in ids], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite labels")
    varying = bool(np.ptp(x) and np.ptp(y))
    result = {"assay": item["assay"], "method": receipt["method"], "n": len(ids),
              "seed": 42, "input_sha256": expected,
              "spearman": float(spearmanr(x, y).statistic) if varying else None,
              "pearson": float(pearsonr(x, y).statistic) if varying else None,
              "undefined_reason": None if varying else "constant prediction or label",
              "source_type": receipt["source_type"], "comparison_group": receipt["comparison_group"],
              "independent_zero_shot_admitted": receipt["independent_zero_shot_admitted"],
              "endpoint": item["endpoint"], "coverage": "complete; no imputation; signed correlations"}
    if item["endpoint"]["kind"] == "classification":
        from sklearn.metrics import roc_auc_score, average_precision_score
        positive = y if item["endpoint"]["label_sign"] == 1 else 1 + y
        if set(positive) - {0, 1}:
            raise ValueError("Classification labels must be raw 0/1")
        result["auroc"] = float(roc_auc_score(positive, x)) if np.ptp(positive) else None
        result["auprc"] = float(average_precision_score(positive, x)) if np.ptp(positive) else None
    freeze(out, result)
    return result
