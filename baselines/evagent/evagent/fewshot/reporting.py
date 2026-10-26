"""Offline numerical baselines and evaluator-only reports; no provider calls."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .evaluation import evaluate_predictions, ridge_predict, sequence_nearest_neighbor
from .protocol import CONDITIONS, SHOTS, all_requests, digest, validate_inputs, resolve_conditions
from .runner import episodes, load, save, sha

BASELINES = {"nearest_neighbor": sequence_nearest_neighbor, "ridge": ridge_predict}


def run_baselines(episodes_root, output_root):
    count = 0
    for _folder, episode, support in episodes(episodes_root):
        validate_inputs(episode, support)
        for method, predict in BASELINES.items():
            for k in SHOTS[1:]:
                references = support["examples"][:k]
                condition = f"{method}_{k}"
                identity = digest([episode, references, method, "alpha=1.0", sha(Path(__file__).with_name("evaluation.py"))])
                target = Path(output_root) / episode["episode_id"] / (condition + ".json")
                if target.exists():
                    if load(target).get("input_hash") != identity:
                        raise ValueError("Baseline inputs or implementation changed; use a new output directory")
                    continue
                scores = predict(references, episode["query"], episode["higher_is_better"])
                save(target, {"status": "ok", "method": method, "shots": k,
                              "condition": condition, "episode_id": episode["episode_id"],
                              "input_hash": identity, "scores": scores, "model_calls": 0,
                              "reported_tokens": 0, "currency_cost": 0.0})
                count += 1
    return {"new_baseline_cells": count, "model_calls": 0}


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def report(episodes_root, run_dir, baselines_dir, output_root, *, plot=True):
    rows, cases = [], []
    root = Path(run_dir)
    run_plan = load(root / "plan.json") if (root / "plan.json").exists() else None
    condition_ids = [c['name'] for c in run_plan['conditions']] if run_plan else None
    active_conditions = resolve_conditions(condition_ids)
    execution = load(root / "execution_config.json") if (root / "execution_config.json").exists() else None
    if run_plan and run_plan.get("fingerprint") != digest({k: v for k, v in run_plan.items() if k != "fingerprint"}):
        raise ValueError("Run plan fingerprint mismatch")
    frozen_episodes = list(episodes(episodes_root))
    if run_plan and set(run_plan["episode_input_hashes"]) != {e["episode_id"] for _, e, _ in frozen_episodes}:
        raise ValueError("Run plan episode cohort mismatch")
    for folder, episode, support in frozen_episodes:
        if run_plan:
            expected_inputs = run_plan["episode_input_hashes"].get(episode["episode_id"])
            if expected_inputs != {name: sha(folder / name) for name in ("episode.json", "support.json")}:
                raise ValueError("Run plan episode input mismatch")
        # Sole evaluator-only access. Hash-check frozen label provenance before scoring.
        label_path = folder / "evaluation_labels.json"
        manifest = load(folder / "manifest.json")
        for filename, expected in manifest["files"].items():
            if sha(folder / filename) != expected:
                raise ValueError("Frozen evaluation inputs changed: " + filename)
        target = load(label_path)
        if target["episode_id"] != episode["episode_id"] or target["higher_is_better"] != episode["higher_is_better"]:
            raise ValueError("Evaluation episode mismatch")
        labels = target["labels"]
        by_condition = defaultdict(list)
        for request in all_requests(episode, support, condition_ids):
            p = root / "predictions" / request["request_hash"] / "result.json"
            result = load(p) if p.exists() else None
            if result and (not run_plan or not execution):
                raise ValueError("Prediction has no frozen run provenance")
            if result and result.get("request_hash") != request["request_hash"]:
                raise ValueError("Prediction input identity mismatch")
            by_condition[request["condition"]["name"]].append(result)
        score_maps = {}
        condition_names = [(c.name, c.shots, "cr_" + c.representation) for c in active_conditions]
        condition_names += [(f"{m}_{k}", k, m) for m in BASELINES for k in SHOTS[1:]]
        for name, k, method in condition_names:
            row = {"episode_id": episode["episode_id"], "assay_id": episode["assay_id"],
                   "domain": episode["domain"], "split": episode["split"], "seed": episode["seed"],
                   "condition": name, "method": method, "shots": k, "status": "not_run",
                   "n_expected": len(episode["query"]), "n_predicted": 0,
                   "coverage": 0.0, "reported_tokens": 0, "currency_cost": None}
            scores = None
            if method in BASELINES:
                path = Path(baselines_dir) / episode["episode_id"] / (name + ".json")
                if path.exists():
                    result = load(path)
                    expected = digest([episode, support["examples"][:k], method, "alpha=1.0",
                                       sha(Path(__file__).with_name("evaluation.py"))])
                    if result.get("input_hash") != expected:
                        raise ValueError("Baseline provenance mismatch")
                    row["status"] = result["status"]
                    scores = result.get("scores")
                    row["currency_cost"] = 0.0
            else:
                units = by_condition[name]
                present = [u for u in units if u is not None]
                row["planned_units"] = len(units)
                row["completed_units"] = len(present)
                row["reported_tokens"] = sum((u.get("usage_summary") or {}).get("total_tokens", 0) for u in present)
                if present and (execution.get("mode") != "live" or any(u.get("execution_mode") != "live" for u in present)):
                    row["status"] = "offline_test_excluded"
                elif present:
                    statuses = sorted(set(u["status"] for u in present))
                    row["status"] = "ok" if len(present) == len(units) and statuses == ["ok"] else "incomplete_or_invalid"
                    row["unit_statuses"] = ",".join(statuses)
                    # Count partial valid candidates but never compute subset performance.
                    row["n_predicted"] = sum(len(u.get("value") or {}) for u in present if u["status"] == "ok")
                    row["coverage"] = row["n_predicted"] / row["n_expected"]
                    if row["status"] == "ok":
                        scores = {}
                        for unit in present:
                            if set(scores) & set(unit["value"]):
                                raise ValueError("Duplicate candidate across CR chunks")
                            scores.update(unit["value"])
            if scores is not None and row["status"] == "ok":
                row.update(evaluate_predictions(scores, labels, episode["higher_is_better"],
                                               query_ids=[q["variant_id"] for q in episode["query"]]))
                score_maps[name] = scores
            rows.append(row)
        # Mechanism claims are not inferred from aggregate gains; retain inspectable rank changes.
        if "numeric_0" in score_maps:
            sign = 1 if episode["higher_is_better"] else -1
            true_order = sorted(labels, key=lambda v: (-sign * labels[v], v))
            truth = {v: i for i, v in enumerate(true_order, 1)}
            old_order = sorted(score_maps["numeric_0"], key=lambda v: (-score_maps["numeric_0"][v], v))
            old_rank = {v: i for i, v in enumerate(old_order, 1)}
            for name, scores in score_maps.items():
                if name not in {"numeric_16", "numeric_32"}:
                    continue
                new_order = sorted(scores, key=lambda v: (-scores[v], v))
                candidates = [{"variant_id": v, "true_rank": truth[v], "zero_shot_rank": old_rank[v],
                               "fewshot_rank": i, "rank_error_reduction": abs(old_rank[v]-truth[v])-abs(i-truth[v])}
                              for i, v in enumerate(new_order, 1)]
                selected = sorted(candidates, key=lambda r: (r["rank_error_reduction"], r["variant_id"]))
                for kind, subset in (("largest_degradation", selected[:3]), ("largest_improvement", selected[-3:])):
                    cases.extend(dict(episode_id=episode["episode_id"], condition=name, selection=kind, **r) for r in subset)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["assay_id"], row["split"], row["condition"], row["method"], row["shots"])].append(row)
    summary = []
    for (assay, split, condition, method, shots), members in sorted(groups.items()):
        complete = [r for r in members if r["status"] == "ok"]
        correlations = [r["spearman"] for r in complete if r["spearman"] is not None]
        precision = [r["top_k_overlap_precision"] for r in complete]
        summary.append({"assay_id": assay, "split": split, "condition": condition, "method": method,
                        "shots": shots, "planned_seeds": len(members), "complete_seeds": len(complete),
                        "defined_spearman_seeds": len(correlations),
                        "mean_spearman": sum(correlations)/len(correlations) if correlations else None,
                        "mean_top25_precision": sum(precision)/len(precision) if precision else None,
                        "reported_tokens": sum(r["reported_tokens"] for r in members)})
    lookup = {(r["episode_id"], r["condition"]): r for r in rows}
    paired = []
    for row in rows:
        base = lookup[(row["episode_id"], "numeric_0")]
        if row["condition"] != "numeric_0" and row["status"] == base["status"] == "ok":
            paired.append({"episode_id": row["episode_id"], "condition": row["condition"],
                           "delta_spearman": row["spearman"]-base["spearman"] if row["spearman"] is not None and base["spearman"] is not None else None,
                           "delta_top25_precision": row["top_k_overlap_precision"]-base["top_k_overlap_precision"]})
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    status = load(root / "status.json") if (root / "status.json").exists() else {"status": "not_started"}
    save(out / "results.json", {"rows": rows, "summary": summary, "paired_vs_zero_shot": paired,
                                "run_status": status,
                                "run_fingerprint": run_plan["fingerprint"] if run_plan else None,
                                "model_config": run_plan["model_config"] if run_plan else None,
                                "execution": execution,
                                "currency_cost_basis": "Unknown unless supplied by provider billing; tokens are not dollars"})
    save(out / "cases.json", {"selection": "post-hoc largest rank-error changes; not mechanistic proof", "cases": cases})
    _write_csv(out / "results.csv", rows)
    _write_csv(out / "learning_curves.csv", summary)
    _write_csv(out / "paired_vs_zero_shot.csv", paired)
    lines = ["# Fixed-harness few-shot pilot", "", f"LLM run status: **{status['status']}**.", "",
             "GB1 and Peri are an engineering pilot. No cross-assay generalization claim follows from these two tasks.",
             "Shots count every measured example supplied in every request. All arms share nested support and the same 250-query set per episode.",
             "Numerical baselines use support labels only; this is not evidence that LLM few-shot works.",
             "", "| Assay | Split | Condition | Complete/planned seeds | Spearman | Top-25 overlap |",
             "|---|---|---|---:|---:|---:|"]
    for row in summary:
        rho = "—" if row["mean_spearman"] is None else f"{row['mean_spearman']:.4f}"
        precision = "—" if row["mean_top25_precision"] is None else f"{row['mean_top25_precision']:.4f}"
        lines.append(f"| {row['assay_id']} | {row['split']} | {row['condition']} | {row['complete_seeds']}/{row['planned_seeds']} | {rho} | {precision} |")
    lines += ["", "Metrics require complete predictions; invalid or missing outputs are not imputed.",
              "Spearman uses average-rank ties; top-25 boundary ties break by frozen candidate ID. Constant-label or constant-prediction Spearman is undefined.",
              "Five seeds describe sampling variability within one landscape, not five independent biological studies.",
              "Numeric labels retain source units/direction; no test-label normalization. WT labels are not assumed.",
              "No adaptive harness, model finetuning, example retrieval or active acquisition is implemented."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    if plot:
        plot_curves(summary, out)
    return {"cells": len(rows), "complete_cells": sum(r["status"] == "ok" for r in rows),
            "llm_complete_cells": sum(r["status"] == "ok" and r["method"].startswith("cr_") for r in rows),
            "report": str(out / "REPORT.md")}


def plot_curves(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted({(r["assay_id"], r["split"]) for r in summary})
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3 * len(panels)), squeeze=False)
    for i, (assay, split) in enumerate(panels):
        for j, (metric, label) in enumerate((("mean_spearman", "Spearman"), ("mean_top25_precision", "Top-25 overlap precision"))):
            ax = axes[i][j]
            for method in ("cr_numeric", "nearest_neighbor", "ridge"):
                rs = sorted([r for r in summary if r["assay_id"] == assay and r["split"] == split
                             and r["method"] == method and r[metric] is not None], key=lambda r: r["shots"])
                if rs:
                    ax.plot([r["shots"] for r in rs], [r[metric] for r in rs], marker="o", label=method)
            ax.set(title=assay + " / " + split, xlabel="Measured support examples", ylabel=label)
            ax.set_xticks(SHOTS if any(r["method"] == "cr_numeric" and r[metric] is not None
                                       for r in summary) else SHOTS[1:])
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend()
    fig.suptitle("Fixed-harness few-shot pilot — available results only; missing LLM runs are not zero", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / "learning_curves.png", dpi=160)
    fig.savefig(out / "learning_curves.pdf")
    plt.close(fig)
