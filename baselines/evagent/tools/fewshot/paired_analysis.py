"""Read-only analysis of frozen few-shot results; write a NEW report snapshot only.

This standalone reporter is outside the frozen harness. No provider imports or
model calls. Numeric comparisons use the same completed seeds across all shots;
16-shot controls use matched episodes. Requires only Python for JSON/CSV/Markdown;
matplotlib is optional for PNG/PDF. Usage:

  code/.venv/bin/python -I experiments/fewshot_v1/paired_analysis.py \
      --experiment experiments/fewshot_v1 --output /path/to/new/snapshot \
      --analysis /path/to/progress_snapshot/analysis/results.json \
      --ledger /path/to/progress_snapshot/request_ledger.sqlite
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

SHOTS = (0, 4, 8, 16, 32)
NUMERIC = tuple(f"numeric_{k}" for k in SHOTS)
CONTROLS = ("numeric_16", "ordinal_16", "unlabeled_16", "shuffled_16")
METRICS = ("spearman", "top_k_overlap_precision")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def complete(row):
    return (row.get("status") == "ok" and row.get("coverage") == 1.0
            and row.get("n_valid") == row.get("n_expected")
            and row.get("n_expected", 0) > 0)


def mean(values):
    return sum(values) / len(values) if values else None


def index_rows(rows):
    panels = defaultdict(dict)
    for row in rows:
        key = row["episode_id"], row["condition"]
        panel = panels[row["assay_id"], row["split"]]
        if key in panel:
            raise ValueError(f"Duplicate analysis row: {key}")
        panel[key] = row
    return panels


def shared_episodes(panel, conditions, metric=None):
    ids = {episode_id for episode_id, _ in panel}
    return sorted(episode_id for episode_id in ids if all(
        (episode_id, condition) in panel and complete(panel[episode_id, condition])
        and (metric is None or finite(panel[episode_id, condition].get(metric)))
        for condition in conditions))


def common_curves(rows):
    """A fixed seed set per assay/split/metric across every numeric shot.

    Each baseline point must cover that exact same set with a defined metric.
    Missing/undefined baseline values suppress the point rather than shrink N.
    """
    output = []
    for (assay, split), panel in sorted(index_rows(rows).items()):
        planned = len({episode_id for episode_id, _ in panel})
        common_complete = shared_episodes(panel, NUMERIC)
        for metric in METRICS:
            common = shared_episodes(panel, NUMERIC, metric)
            for method in ("numeric", "nearest_neighbor", "ridge"):
                for shots in SHOTS if method == "numeric" else SHOTS[1:]:
                    condition = f"{method}_{shots}"
                    matched = [panel.get((episode_id, condition)) for episode_id in common]
                    defined = [row[metric] for row in matched
                               if row and complete(row) and finite(row.get(metric))]
                    ok = bool(common) and len(defined) == len(common)
                    output.append({
                        "assay_id": assay, "split": split, "metric": metric,
                        "method": method, "shots": shots, "condition": condition,
                        "planned_seeds": planned, "common_complete_seeds": len(common_complete),
                        "common_metric_seeds": len(common), "point_defined_seeds": len(defined),
                        "episode_ids": common, "mean": mean(defined) if ok else None,
                        "values": defined if ok else [],
                        "status": "matched" if ok else "no_common_seeds" if not common else "undefined_or_missing_on_common_seeds",
                    })
    return output


def available_table(rows):
    """Descriptive only: independent available-seed counts are NOT paired curves."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["assay_id"], row["split"], row["condition"]].append(row)
    result = []
    for (assay, split, condition), members in sorted(groups.items()):
        good = [row for row in members if complete(row)]
        item = {"assay_id": assay, "split": split, "condition": condition,
                "planned_seeds": len(members), "complete_seeds": len(good),
                "available_episode_ids": [row["episode_id"] for row in good],
                "status_counts": dict(Counter(row["status"] for row in members)),
                "archived_result_reported_tokens": sum(row.get("reported_tokens", 0) or 0 for row in members)}
        for metric in METRICS:
            values = [row[metric] for row in good if finite(row.get(metric))]
            item[f"defined_{metric}_seeds"] = len(values)
            item[f"mean_{metric}"] = mean(values)
        result.append(item)
    return result


def control_comparisons(rows):
    """Both all-four-common summaries and pairwise matched deltas vs numeric16."""
    points, differences, summary = [], [], []
    for (assay, split), panel in sorted(index_rows(rows).items()):
        for metric in METRICS:
            common = shared_episodes(panel, CONTROLS, metric)
            for condition in CONTROLS:
                values = [panel[episode_id, condition][metric] for episode_id in common]
                summary.append({"assay_id": assay, "split": split, "metric": metric,
                                "condition": condition, "common_seeds": len(common),
                                "episode_ids": common, "mean": mean(values)})
                for episode_id, value in zip(common, values):
                    points.append({"assay_id": assay, "split": split, "metric": metric,
                                   "episode_id": episode_id, "condition": condition, "value": value,
                                   "comparison_set": "all_four_controls_common"})
            for condition in CONTROLS[1:]:
                pairs = shared_episodes(panel, ("numeric_16", condition), metric)
                for episode_id in pairs:
                    numeric = panel[episode_id, "numeric_16"][metric]
                    comparator = panel[episode_id, condition][metric]
                    differences.append({"assay_id": assay, "split": split, "metric": metric,
                                        "episode_id": episode_id, "condition": condition,
                                        "numeric_16": numeric, "control": comparator,
                                        "delta_control_minus_numeric": comparator - numeric,
                                        "pairwise_complete_seeds": len(pairs)})
    return {"common_summary": summary, "common_points": points, "paired_deltas": differences,
            "delta_direction": "control minus numeric16; positive means the control performed better"}


def average_ranks(scores):
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    ranks, start = {}, 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[start]]:
            end += 1
        for key in ordered[start:end]:
            ranks[key] = (start + 1 + end) / 2
        start = end
    return ranks


def inspect_predictions(run_dir):
    """Read terminal atomic result files; never retry, repair or call a provider."""
    groups = defaultdict(Counter)
    score_maps = defaultdict(dict)
    files = []
    for path in sorted((Path(run_dir) / "predictions").glob("*/result.json")):
        value = read(path)
        if value.get("execution_mode") != "live":
            continue
        condition = value["condition"]["name"]
        key = value["episode_id"], condition
        stats = groups[condition]
        stats["archived_units"] += 1
        attempts = value.get("attempts") or []
        stats["units_with_archived_first_attempt"] += bool(attempts)
        stats["first_attempt_valid_units"] += bool(attempts and (attempts[0].get("validation") or {}).get("valid"))
        stats["final_valid_units"] += value.get("status") == "ok"
        stats["units_with_repairs"] += len(attempts) > 1
        stats["archived_generation_attempts"] += len(attempts)
        stats["archived_result_reported_tokens"] += (value.get("usage_summary") or {}).get("total_tokens", 0) or 0
        stats["terminal_" + value["status"]] += 1
        if value.get("status") == "ok":
            scores = value.get("value") or {}
            if (set(scores) != set(value["expected_ids"]) or not all(finite(score) for score in scores.values())
                    or set(scores) & set(score_maps[key])):
                raise ValueError(f"Invalid/duplicate saved live prediction: {path}")
            score_maps[key].update(scores)
        files.append({"path": str(path), "sha256": sha(path)})
    totals = Counter()
    for values in groups.values():
        totals.update(values)
    def with_rates(counts):
        value = dict(counts)
        value["first_attempt_valid_fraction"] = (counts["first_attempt_valid_units"] / counts["units_with_archived_first_attempt"]
                                                 if counts["units_with_archived_first_attempt"] else None)
        value["final_valid_fraction_of_archived_units"] = (counts["final_valid_units"] / counts["archived_units"]
                                                          if counts["archived_units"] else None)
        return value
    return {"totals": with_rates(totals), "by_condition": {key: with_rates(value) for key, value in sorted(groups.items())},
            "denominator_caveat": "Validity is among archived terminal result units; inflight/unarchived attempts are accounted separately by the ledger.",
            "files": files}, score_maps


def load_frozen_episodes(experiment):
    values = {}
    for path in sorted((experiment / "episodes").glob("*/episode.json")):
        manifest = read(path.with_name("manifest.json"))
        for filename, expected in manifest["files"].items():
            if sha(path.with_name(filename)) != expected:
                raise ValueError(f"Frozen input hash mismatch: {path.parent / filename}")
        episode, labels = read(path), read(path.with_name("evaluation_labels.json"))
        if episode["episode_id"] != labels["episode_id"] or episode["higher_is_better"] != labels["higher_is_better"]:
            raise ValueError("Episode/label identity mismatch")
        values[episode["episode_id"]] = (episode, labels["labels"])
    return values


def tied_cases(rows, episodes, score_maps):
    cases, availability = [], []
    accepted = {(row["episode_id"], row["condition"]) for row in rows if complete(row)}
    for episode_id, (episode, labels) in sorted(episodes.items()):
        zero_key = episode_id, "numeric_0"
        if zero_key not in accepted:
            continue
        zero = score_maps.get(zero_key, {})
        if set(zero) != set(labels):
            raise ValueError("Analysis marks complete zero-shot but matching prediction files are absent")
        sign = 1 if episode["higher_is_better"] else -1
        truth = average_ranks({key: sign * value for key, value in labels.items()})
        old = average_ranks(zero)
        query = {row["variant_id"]: row for row in episode["query"]}
        for condition in ("numeric_16", "numeric_32"):
            key = episode_id, condition
            if key not in accepted:
                continue
            scores = score_maps.get(key, {})
            if set(scores) != set(labels):
                raise ValueError("Analysis marks complete few-shot but matching prediction files are absent")
            new = average_ranks(scores)
            candidates = []
            for variant_id in sorted(labels):
                delta = abs(old[variant_id] - truth[variant_id]) - abs(new[variant_id] - truth[variant_id])
                candidates.append({"episode_id": episode_id, "assay_id": episode["assay_id"],
                                   "split": episode["split"], "condition": condition, "variant_id": variant_id,
                                   "sequence": query[variant_id]["sequence"], "mutation": query[variant_id]["mutation"],
                                   "measured_label": labels[variant_id], "higher_is_better": episode["higher_is_better"],
                                   "zero_shot_score": zero[variant_id], "fewshot_score": scores[variant_id],
                                   "true_average_rank": truth[variant_id], "zero_shot_average_rank": old[variant_id],
                                   "fewshot_average_rank": new[variant_id], "rank_error_reduction": delta})
            improving = sorted((row for row in candidates if row["rank_error_reduction"] > 0),
                               key=lambda row: (-row["rank_error_reduction"], row["variant_id"]))
            degrading = sorted((row for row in candidates if row["rank_error_reduction"] < 0),
                               key=lambda row: (row["rank_error_reduction"], row["variant_id"]))
            for kind, subset in (("improved", improving[:3]), ("degraded", degrading[:3])):
                cases.extend(dict(row, selection=kind) for row in subset)
            availability.append({"episode_id": episode_id, "condition": condition,
                                 "improved_variants": len(improving), "degraded_variants": len(degrading),
                                 "unchanged_variants": len(candidates) - len(improving) - len(degrading)})
    return {"selection": "Post-hoc largest signed average-rank-error changes; true improvement requires delta>0, degradation delta<0; no mechanistic or prospective claim.",
            "availability": availability, "cases": cases}


def ledger_snapshot(path, status):
    path = Path(path)
    if not path.exists():
        ledger = status.get("ledger")
        return dict(ledger or {}, basis="status.json ledger snapshot; database unavailable" if ledger else "unavailable",
                    reported_tokens=ledger.get("reported_tokens") if ledger else None,
                    currency_cost=None, currency_cost_basis="No billing statement; tokens are not currency")
    connection = sqlite3.connect("file:" + quote(str(path.resolve()), safe="/") + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        state = dict(connection.execute("SELECT key,value FROM run_state"))
        rows = connection.execute("SELECT status,tokens,request_json FROM attempts WHERE status!='not_submitted'").fetchall()
    finally:
        connection.close()
    previous = json.loads(state.get("prior_usage", "{}"))
    statuses = Counter(previous.get("requests_by_status", {}))
    statuses.update(row[0] for row in rows)
    inflight = sum(status == "inflight" for status, _, _ in rows)
    unknown = previous.get("unknown_usage_attempts", 0) + sum(
        "unknown" in status or status == "inflight" or tokens is None for status, tokens, _ in rows)
    by_condition = defaultdict(Counter)
    for terminal, tokens, request_json in rows:
        metadata = json.loads(request_json).get("metadata") or {}
        condition = metadata.get("condition", "unattributed")
        by_condition[condition]["provider_attempts"] += 1
        by_condition[condition]["reported_tokens"] += tokens or 0
        by_condition[condition][terminal] += 1
    return {"basis": "SQLite mode=ro, one read transaction", "path": str(path.resolve()),
            "run_fingerprint": state.get("fingerprint"),
            "captured_utc": utc(), "provider_attempts": len(rows) + previous.get("provider_attempts", 0),
            "reported_tokens": sum(tokens or 0 for _, tokens, _ in rows) + previous.get("reported_tokens", 0),
            "prior_reported_tokens": previous.get("reported_tokens", 0), "inflight_attempts": inflight,
            "unknown_usage_attempts": unknown, "usage_is_lower_bound": bool(unknown),
            "requests_by_status": dict(statuses), "pause_reason": state.get("pause_reason"),
            "token_budget": state.get("token_budget"),
            "by_condition_current_ledger": {key: dict(value) for key, value in sorted(by_condition.items())},
            "currency_cost": None, "currency_cost_basis": "No billing statement; tokens are not currency"}


def verify_campaign_identity(source, run_dir, ledger_info, *, has_llm_results):
    """Reject same-cohort results from another model/config/campaign.

    Baseline-only local copies may not yet contain a live plan or ledger. They
    remain explicitly unverified and cannot supply scientific LLM points.
    """
    plan_path = Path(run_dir) / "plan.json"
    ledger_present = ledger_info.get("basis") == "SQLite mode=ro, one read transaction"
    if not plan_path.exists():
        if has_llm_results or ledger_present:
            raise ValueError("LLM results/ledger require the matching frozen run plan")
        return {"status": "baseline_only_unverified", "reason": "No live run plan/ledger; no LLM comparison is available",
                "analysis_run_fingerprint": source.get("run_fingerprint")}
    plan = read(plan_path)
    fingerprint = plan.get("fingerprint")
    if not fingerprint or fingerprint != digest({key: value for key, value in plan.items() if key != "fingerprint"}):
        raise ValueError("Frozen run plan digest mismatch")
    if source.get("run_fingerprint") != fingerprint:
        raise ValueError("Analysis/run-plan campaign fingerprint mismatch")
    if ledger_present and ledger_info.get("run_fingerprint") != fingerprint:
        raise ValueError("Ledger/run-plan campaign fingerprint mismatch")
    if has_llm_results and not ledger_present:
        raise ValueError("Scientific LLM results require a matching read-only ledger or consistent backup")
    return {"status": "verified" if ledger_present else "planned_without_ledger",
            "run_fingerprint": fingerprint, "plan_path": str(plan_path),
            "plan_sha256": sha(plan_path), "ledger_identity_verified": ledger_present}


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def display(value):
    return "—" if value is None else f"{value:.4f}"


def markdown(result):
    tokens = result["ledger"].get("reported_tokens")
    validity = result["prediction_validity"]["totals"]
    lines = ["# Fixed-harness few-shot: matched snapshot", "",
             f"Snapshot: {result['snapshot_utc']}. Run status: **{result['run_status'].get('status', 'unavailable')}**.",
             f"Analysis input: `{result['provenance']['analysis_path']}`.",
             "", "Numeric curves use the same seeds at every shot; baseline points use exactly that same cohort. Missing groups are unavailable, never zero.",
             "Spearman curves require defined Spearman at every numeric shot. Top-25 uses its own common complete cohort. Five seeds are sampling repeats, not independent biological studies.",
             "", f"Ledger-reported tokens: **{tokens if tokens is not None else 'unavailable'}**; unknown-usage attempts: **{result['ledger'].get('unknown_usage_attempts', 'unavailable')}**; lower bound: **{result['ledger'].get('usage_is_lower_bound', 'unavailable')}**.",
             "Currency cost is unknown. Archived-result token sums are attribution subsets, not the ledger total.",
             f"First-attempt validity among archived attempts: {display(validity.get('first_attempt_valid_fraction'))}; final validity among archived terminal units: {display(validity.get('final_valid_fraction_of_archived_units'))}.",
             "Inflight or unarchived failures are represented by ledger statuses, not silently counted as valid predictions.",
             "", "## Matched numeric cohort", "", "| Assay | Split | Metric | Common complete | Defined common | Planned seeds |", "|---|---|---|---:|---:|---:|"]
    for row in result["common_curves"]:
        if row["condition"] == "numeric_0":
            lines.append(f"| {row['assay_id']} | {row['split']} | {row['metric']} | {row['common_complete_seeds']} | {row['common_metric_seeds']} | {row['planned_seeds']} |")
    lines += ["", "## 16-shot controls (all four share a cohort)", "", "| Assay | Split | Metric | Condition | Paired seeds | Mean |", "|---|---|---|---|---:|---:|"]
    for row in result["controls"]["common_summary"]:
        lines.append(f"| {row['assay_id']} | {row['split']} | {row['metric']} | {row['condition']} | {row['common_seeds']} | {display(row['mean'])} |")
    lines += ["", "Pairwise differences in `control_paired_deltas.csv` are **control minus numeric16**; positive means the control performed better. Each pair explicitly records its own matched-seed denominator.",
              "", "## All available results (descriptive; not a paired comparison)", "", "| Assay | Split | Condition | Complete/planned | Defined rho | Spearman | Top-25 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in result["all_available"]:
        lines.append(f"| {row['assay_id']} | {row['split']} | {row['condition']} | {row['complete_seeds']}/{row['planned_seeds']} | {row['defined_spearman_seeds']} | {display(row['mean_spearman'])} | {display(row['mean_top_k_overlap_precision'])} |")
    lines += ["", "## Cases and limitations", "",
              "`tied_cases.csv` includes sequence, mutation, measured label, scores and average ranks. Positive/negative error changes alone qualify as improvement/degradation; unchanged or tied variants are not relabeled as successes.",
              "Cases are selected after viewing outcomes and are not mechanistic evidence. Numeric labels retain the dataset's supplied DMS_score units; they are not raw instrument counts. The Peri Mg condition remains unspecified in its source metadata.",
              "All report inputs remain unchanged. The ledger transaction and atomic result files may represent slightly different moments during a live run; timestamps and hashes are retained in snapshot.json."]
    return "\n".join(lines) + "\n"


def plot(result, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib is not installed; JSON/CSV/Markdown complete"}
    curves = result["common_curves"]
    panels = sorted({(row["assay_id"], row["split"]) for row in curves})
    if not panels:
        return {"status": "unavailable", "reason": "no assay/split groups"}
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, 3.5 * len(panels)), squeeze=False)
    for index, (assay, split) in enumerate(panels):
        for column, metric in enumerate(METRICS):
            ax = axes[index][column]
            rows = [row for row in curves if (row["assay_id"], row["split"], row["metric"]) == (assay, split, metric)]
            n = rows[0]["common_metric_seeds"]
            for method in ("numeric", "nearest_neighbor", "ridge"):
                points = sorted((row for row in rows if row["method"] == method), key=lambda row: row["shots"])
                if any(row["mean"] is not None for row in points):
                    ax.plot([row["shots"] for row in points], [row["mean"] if row["mean"] is not None else math.nan for row in points], marker="o", label=f"{method} (same n={n})")
                    for row in points:
                        if row["mean"] is not None:
                            ax.scatter([row["shots"]] * len(row["values"]), row["values"], alpha=.25, s=12)
            if not n:
                ax.text(.5, .5, "Unavailable: no common complete seeds", ha="center", va="center", transform=ax.transAxes)
            ax.set(title=f"{assay} / {split}\ncommon n={n}/{rows[0]['planned_seeds']}", xlabel="Measured examples", ylabel=metric, xticks=SHOTS)
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8)
    fig.suptitle("Matched-seed learning curves; points = seeds, lines = means; missing is not zero", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .975))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"common_seed_learning_curves.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, 3.5 * len(panels)), squeeze=False)
    for index, (assay, split) in enumerate(panels):
        for column, metric in enumerate(METRICS):
            ax = axes[index][column]
            points = [row for row in result["controls"]["common_points"] if (row["assay_id"], row["split"], row["metric"]) == (assay, split, metric)]
            ids = sorted({row["episode_id"] for row in points})
            for episode_id in ids:
                values = {row["condition"]: row["value"] for row in points if row["episode_id"] == episode_id}
                ax.plot(range(4), [values[condition] for condition in CONTROLS], marker="o", alpha=.65, label=episode_id.rsplit("__", 1)[-1])
            if not ids:
                ax.text(.5, .5, "Unavailable: no complete four-control pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set(title=f"{assay} / {split}\npaired n={len(ids)}", ylabel=metric, xticks=range(4), xticklabels=[name.removesuffix("_16") for name in CONTROLS])
            ax.grid(alpha=.2)
            if ids:
                ax.legend(fontsize=8)
    fig.suptitle("16-shot matched controls; each line is one seed", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .975))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"paired_16shot_controls.{suffix}", dpi=160)
    plt.close(fig)
    return {"status": "complete", "files": ["common_seed_learning_curves.png", "common_seed_learning_curves.pdf", "paired_16shot_controls.png", "paired_16shot_controls.pdf"]}


def build_snapshot(experiment, output, *, analysis=None, run_dir=None, ledger=None, plots=True):
    started = utc()
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    analysis = Path(analysis).resolve() if analysis else experiment / "analysis/results.json"
    run_dir = Path(run_dir).resolve() if run_dir else experiment / "llm_full"
    ledger = Path(ledger).resolve() if ledger else run_dir / "provider/request_ledger.sqlite"
    if output.exists():
        raise FileExistsError("Output must be a new snapshot directory; no report is overwritten")
    for protected in (experiment / "episodes", experiment / "sources", experiment / "offline_baselines", run_dir, experiment / "analysis"):
        if output.is_relative_to(protected):
            raise ValueError("Output cannot be inside frozen inputs, predictions, baselines or original analysis")
    analysis_bytes = analysis.read_bytes()
    source = json.loads(analysis_bytes)
    rows = source["rows"]
    episodes = load_frozen_episodes(experiment)
    if {row["episode_id"] for row in rows} != set(episodes):
        raise ValueError("Analysis and frozen episode cohorts differ")
    status = read(run_dir / "status.json") if (run_dir / "status.json").exists() else source.get("run_status", {"status": "unavailable"})
    ledger_info = ledger_snapshot(ledger, status)
    validity, scores = inspect_predictions(run_dir)
    has_llm_results = bool(validity["files"]) or any(
        complete(row) and row["condition"] in NUMERIC + CONTROLS for row in rows)
    campaign = verify_campaign_identity(source, run_dir, ledger_info, has_llm_results=has_llm_results)
    result = {"schema": "fewshot-paired-report-v1", "snapshot_utc": utc(), "read_started_utc": started,
              "provenance": {"analysis_path": str(analysis), "analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
                             "reporter_sha256": sha(Path(__file__)), "experiment": str(experiment),
                             "run_dir": str(run_dir), "run_fingerprint": source.get("run_fingerprint"),
                             "campaign_identity": campaign,
                             "input_mode": "read-only; outputs written to a new directory only",
                             "consistency": "Analysis is one frozen JSON file; ledger is a single SQLite read transaction; atomic prediction files/status can advance independently during a live run."},
              "run_status": status, "ledger": ledger_info, "prediction_validity": validity,
              "common_curves": common_curves(rows), "all_available": available_table(rows),
              "controls": control_comparisons(rows), "cases": tied_cases(rows, episodes, scores)}
    result["ledger"]["archived_result_reported_tokens"] = validity["totals"].get("archived_result_reported_tokens", 0)
    result["ledger"]["attribution_caveat"] = "Ledger total includes attempts absent from terminal result files. During live reads, ledger and result timestamps can differ; do not subtract them as an exact unassigned-cost measure."
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "common_seed_curves.csv", result["common_curves"])
    write_csv(output / "all_available.csv", result["all_available"])
    write_csv(output / "control_common_summary.csv", result["controls"]["common_summary"])
    write_csv(output / "control_paired_deltas.csv", result["controls"]["paired_deltas"])
    write_csv(output / "tied_cases.csv", result["cases"]["cases"])
    result["plots"] = plot(result, output) if plots else {"status": "disabled"}
    (output / "snapshot.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "REPORT.md").write_text(markdown(result))
    return {"snapshot": str(output), "analysis_rows": len(rows), "common_numeric_cohorts": [
        {key: row[key] for key in ("assay_id", "split", "metric", "common_metric_seeds")}
        for row in result["common_curves"] if row["condition"] == "numeric_0"],
        "plots": result["plots"], "model_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analysis", type=Path, help="Existing analysis/results.json (e.g. a progress snapshot)")
    parser.add_argument("--run-dir", type=Path, help="Default: EXPERIMENT/llm_full")
    parser.add_argument("--ledger", type=Path, help="Read-only SQLite ledger or consistent backup")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_snapshot(args.experiment, args.output, analysis=args.analysis,
                                   run_dir=args.run_dir, ledger=args.ledger, plots=not args.no_plots)))
