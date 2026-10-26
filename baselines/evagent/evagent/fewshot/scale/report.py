"""Isolated offline evaluator; test labels are read only in this module."""
from __future__ import annotations

import csv
import statistics
import time
from collections import Counter, defaultdict

from evagent.fewshot.evaluation import evaluate_predictions
from evagent.fewshot.runner import load, save


def report(root, accounting, state):
    cohort = load(root / "COHORT.json")
    metadata = {e["metadata"]["assay_id"]: e for e in cohort["assays"]}
    expected_tasks = len(metadata) * 3 + sum(e["multi_round_eligible"] * 8 +
                     sum(x["eligible"] for x in e["extrapolation"]) * 3 for e in cohort["assays"])
    prediction_status = Counter(load(p)["status"] for p in (root / "predictions").glob("*/result.json"))
    records, acquisition, statuses, failures = [], [], Counter(), Counter()
    for path in sorted((root / "tasks").rglob("result.json")):
        task = load(path)
        statuses[task["status"]] += 1
        failures.update(task["failures"])
        assay = task["assay"]
        base = root / "inputs" / assay
        if task["split"] != "random":
            base /= task["split"]
        labels = load(base / "evaluation.json")
        methods = {"llm": task["predictions"]["test"], **task["baselines"]}
        for method, predictions in methods.items():
            metrics = evaluate_predictions(predictions, labels) if set(predictions) == set(labels) else {
                "coverage": len(predictions) / len(labels), "spearman": None,
                "top_k_overlap_precision": None, "n_valid": len(predictions), "n_expected": len(labels)}
            records.append({"assay": assay, "domain": metadata[assay]["metadata"]["domain"],
                            "family": metadata[assay]["family"], "study": metadata[assay]["study"],
                            "development": metadata[assay]["development"],
                            "development_background": metadata[assay]["development_background"],
                            "mode": task["mode"], "arm": task["arm"], "shots": task["shots"],
                            "method": method, "split": task["split"],
                            "task_status": task["status"], **metrics})
        if task["mode"].startswith("round/"):
            acquisition.append({"assay": assay, "policy": task["mode"].split("/")[1],
                                "shots": task["shots"], "best_measured": task["best_measured"],
                                "selected_ids": task["selected"], "status": task["status"]})
    paired = []
    for domain in ("protein", "rna"):
        for mode in sorted({r["mode"] for r in records if not r["mode"].startswith("round/")}):
            for arm in ("raw32", "anchored32"):
                by_assay = defaultdict(dict)
                for r in records:
                    if r["domain"] == domain and r["mode"] == mode and r["method"] == "llm" and r["task_status"] == "complete":
                        by_assay[r["assay"]][r["arm"]] = r
                pairs = [(v["zero"], v[arm]) for v in by_assay.values()
                         if "zero" in v and arm in v and v["zero"]["spearman"] is not None and v[arm]["spearman"] is not None]
                paired.append({"domain": domain, "mode": mode, "comparison": arm + " - zero",
                               "n_complete_pairs": len(pairs),
                               "mean_delta_spearman": statistics.mean(b["spearman"] - a["spearman"] for a, b in pairs) if pairs else None,
                               "mean_delta_top25": statistics.mean(b["top_k_overlap_precision"] - a["top_k_overlap_precision"] for a, b in pairs) if pairs else None})
    round_pairs = []
    for k in (4, 8, 16, 32):
        for assay in metadata:
            cells = {r["mode"]: r for r in records if r["assay"] == assay and r["shots"] == k
                     and r["method"] == "llm" and r["task_status"] == "complete"
                     and r["mode"] in {"round/random", "round/greedy"}}
            acq = {r["policy"]: r for r in acquisition if r["assay"] == assay and r["shots"] == k}
            if len(cells) == 2:
                a, b = cells["round/random"], cells["round/greedy"]
                round_pairs.append({"assay": assay, "shots": k,
                     "delta_spearman_greedy_minus_random": b["spearman"] - a["spearman"] if a["spearman"] is not None and b["spearman"] is not None else None,
                     "delta_top25": b["top_k_overlap_precision"] - a["top_k_overlap_precision"],
                     "delta_best_measured": acq["greedy"]["best_measured"] - acq["random"]["best_measured"]})
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    save(output / "metrics.json", {"time": time.time(), "state": state, "accounting": accounting,
         "records": records, "paired_static": paired, "acquisition": acquisition, "paired_rounds": round_pairs,
         "task_statuses": dict(statuses), "failure_reasons": dict(failures),
         "expected_tasks": expected_tasks, "prediction_status": dict(prediction_status)})
    if records:
        fields = sorted({k for r in records for k in r})
        temporary = output / "metrics.csv.tmp"
        with temporary.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(output / "metrics.csv")
    lines = ["# Single-seed few-shot scale v3", "", f"Snapshot UTC epoch: {time.time():.0f}; state: {state}",
             "", f"Planned tasks: {expected_tasks}; task states: {dict(statuses)}. Failure reasons: {dict(failures)}.",
             f"Prediction request states: {dict(prediction_status)}.",
             f"Known tokens: {accounting['known_tokens']:,}; unknown reserve: {accounting['unknown_reserve']:,}; inflight reserve: {accounting['inflight_reserve']:,}; cap: {accounting['budget']:,}.",
             "Token quantities are provider usage / conservative upper bounds, not currency charges. Old campaigns are excluded from the new 200M cap and remain intact.",
             "", "## Held-out prediction (complete paired assays only)", "",
             "|Domain|Mode|Comparison|Pairs|Delta Spearman|Delta top25|", "|---|---|---|---:|---:|---:|"]
    for p in paired:
        lines.append(f"|{p['domain']}|{p['mode']}|{p['comparison']}|{p['n_complete_pairs']}|{p['mean_delta_spearman']}|{p['mean_delta_top25']}|")
    lines += ["", "## Real-measurement acquisition", "",
              f"{len(round_pairs)} complete assay/round policy pairs. Per-assay best observed activity and held-out changes are stored separately in metrics.json. Raw activities are not averaged across incompatible assays.",
              "", "## Interpretation and evidence", "",
              "One seed; this is a purposively stratified panel. Related RNA families, repeated studies and previous development backgrounds are annotated in every row. No claim of training-data exclusion. Missing/incomplete outputs are never zero-filled; undefined correlations remain null.",
              "Static random32 and random-policy round32 share measured supports and cache-identical predictions. Both policies share the initial four labels and initial predictions. The permanent 250-test groups never change. Acquisition uses only complete pool predictions and previously revealed real labels.",
              "Nearest-neighbor and alpha=1 one-hot Ridge use the same observed supports. No global label normalization. See tasks/*/inputs.json for source-to-support mapping, inputs/*/oracle.json for measured acquisition labels, and separate evaluation.json for hidden test labels.",
              "Provider-returned thought summaries and final answers remain in attempts/*/provider/request_ledger.sqlite and predictions/*/result.json. They are observable summaries, not a complete or causal account of internal reasoning. No additional label-mechanism calls are run.",
              "", "## Multi-round versus previous campaign", "",
              "Both campaigns reveal real experimental measurements. This run changes to seed 1, 250 disjoint test plus 250 acquisition candidates, fixed independent test chunks, support-anchored scoring, and the expanded assay panel. Absolute differences from the old campaign are not a controlled comparison of a single algorithmic factor."]
    temp = output / "REPORT.md.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(output / "REPORT.md")
    return {"tasks": dict(statuses), "failures": dict(failures), "metric_rows": len(records),
            "expected_tasks": expected_tasks, "prediction_status": dict(prediction_status),
            "valid_rate_completed_requests": prediction_status["ok"] / sum(prediction_status.values()) if prediction_status else None}
