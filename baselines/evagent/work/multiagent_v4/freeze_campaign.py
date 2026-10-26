"""Freeze a new cohort comparison, never replaying earlier predictions or unknown calls."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import importlib.metadata
import json
import platform
import sqlite3
from pathlib import Path

from evagent.integrated.research.campaign import run_campaign
from evagent.integrated.research.contracts import save
from evagent.integrated.research.framework_runner import DESIGN
from evagent.integrated.tasks import file_hash

AUTHORIZATION = (
    "User on 2026-09-13: 好的，你确定现在的agent和multiagent实验没问题的话，就跑上完整实验吧，有问题的话你自己改进下. "
    "This authorizes the reviewed new framework experiment and one operator-directed recovery. "
    "Retain every prior unknown-use attempt as unresolved lower-bound cost; do not replay old predictions. "
    "The standing connection policy remains at most three transport attempts per logical request; "
    "exhaustion or other unknown responses pause, without an unattended restart loop."
)


def debit(ledger, *, authorize_unknown):
    if Path(str(ledger) + "-wal").exists() and Path(str(ledger) + "-wal").stat().st_size:
        raise ValueError("Source ledger must be closed and checkpointed")
    with sqlite3.connect(ledger.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT status,tokens,request_sha256 FROM attempts WHERE status!='not_submitted'").fetchall()
    if any(status == "inflight" for status, _, _ in rows):
        raise ValueError("Unsettled source requests cannot be carried forward")
    unknown = sorted(h for status, tokens, h in rows
                     if status not in {"ok", "blocked", "truncated", "terminal_failure"}
                     or type(tokens) is not int)
    value = {"path": str(ledger), "sha256": file_hash(ledger)}
    if unknown:
        if not authorize_unknown:
            raise ValueError("New smoke has unknown usage; pause rather than freeze a live campaign")
        value["manual_recovery"] = {
            "request_sha256s": unknown, "authorization": AUTHORIZATION,
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "usage_status": "unresolved_lower_bound",
        }
    return value


def freeze(source, preparation, destination, smoke=None, diagnostic_ledgers=()):
    source, preparation, destination = (Path(p).resolve() for p in (source, preparation, destination))
    if destination.exists():
        raise ValueError("Never overwrite a frozen configuration")
    raw = json.loads(source.read_text())
    old = Path(raw["output_dir"])
    with (old / "campaign.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = json.loads((old / "report.json").read_text())
        if report["status"] not in {"paused", "complete", "interrupted"}:
            raise ValueError("Source campaign is still active")
        tasks = json.loads((preparation / "tasks.json").read_text())
        if {t["assay"] for t in tasks} != {t["assay"] for t in raw["tasks"]} or len(tasks) != 24:
            raise ValueError("The admitted cohort changed")
        for previous in raw["tasks"]:
            current = next(t for t in tasks if t["assay"] == previous["assay"])
            if any(file_hash(Path(previous[k])) != file_hash(Path(current[k])) for k in ("sample_file", "context_file")):
                raise ValueError("Frozen sample or original assay metadata changed")
        prior = [*raw.get("prior_ledgers", []), debit(old / "provider/request_ledger.sqlite", authorize_unknown=True)]
        for diagnostic in diagnostic_ledgers:
            diagnostic = Path(diagnostic).resolve()
            with sqlite3.connect(diagnostic.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
                rows = db.execute("SELECT status,request_json,response_json FROM attempts").fetchall()
            if not rows or any(
                status != "failed_unknown_cost"
                or json.loads(request)["metadata"].get("role") != "schema_probe"
                or json.loads(response or "{}").get("response_metadata", {}).get("transport_error", {}).get("status_code") != 400
                for status, request, response in rows
            ):
                raise ValueError("Only the explicitly investigated schema-only HTTP400 diagnostic is admissible here")
            prior.append(debit(diagnostic, authorize_unknown=True))
        if smoke:
            smoke = Path(smoke).resolve()
            receipt = json.loads((smoke / "receipt.json").read_text())
            if receipt["status"] != "passed":
                raise ValueError("Native framework API check did not pass")
            prior.append(debit(smoke / "request_ledger.sqlite", authorize_unknown=False))
        environment = destination.parent / "environment-framework.json"
        installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        value = {"python": platform.python_version(), "packages": dict(sorted(installed.items())),
                 "lockfile_sha256": file_hash(Path(__file__).resolve().parents[2] / "requirements-framework.lock.txt")}
        if environment.exists() and json.loads(environment.read_text()) != value:
            raise ValueError("Previously frozen framework environment changed")
        if not environment.exists():
            save(environment, value)
        raw.pop("execution", None)
        raw.pop("resume_from", None)
        raw.update(output_dir=str(destination.with_suffix("")), repeats=1,
                   analysis_profile="framework_agents_v4", analysis=DESIGN, tasks=tasks,
                   coverage_file=str(preparation / "coverage_31.json"),
                   environment_file=str(environment), prior_ledgers=prior)
        raw["model_config"]["research_tool_protocol"] = "native_framework_v4"
        save(destination, raw)
        dry = run_campaign(destination, dry_run=True)
        if (dry["tasks"], dry["initial_CR_units"], dry["method_results"]) != (24, 696, 141):
            raise ValueError("Unexpected single-seed full cohort matrix")
        root = destination.with_suffix("")
        save(root / "dry_run.json", dry)
        save(root / "protocol.json", {
            "design": DESIGN, "operator_authorization": AUTHORIZATION,
            "previous_config_sha256": file_hash(source),
            "previous_report_sha256": file_hash(old / "report.json"),
            "preparation_manifest_sha256": file_hash(preparation / "manifest.json"),
            "selection": "All 24 source-admitted tasks from full70/31 ncRNA coverage; Pitt N50 only.",
            "sampling": "One sample/generation seed1, one provider repeat. Nested N50/250; no new samples.",
            "calibration": "First sorted task, first two groups, all three conditions. Format/tools/accounting only.",
            "msa": "MSA v2 deterministic preparation shared by all conditions; no prediction-side MSA LLM or conservation prompt.",
            "analysis": "Single gets the union of the two independent literature and structure experts' tools and inputs. Multi roles cannot see each other's messages. Provider requests serialized.",
            "fusion": "Fresh independent CR integrations; no prior draft or shrinkage. Within each N, fixed 50:50 CR/MSA ranks; CR-only if no credible MSA.",
            "primary": "N250 multi-minus-single paired Spearman, task then RNA-family macro. Also report each versus reference, Kendall, N50, validity, costs and abstention.",
            "interpretation": "Exploratory one-seed protocol on previously inspected tasks. Framework, evidence/context, and integration changed together; do not attribute gains solely to framework migration.",
            "failure": "No output substitution; unknown responses/cost pause. Prior unknown costs remain lower bounds.",
            "reselection": "Do not tune prompts, fusion weights, evidence admission or task order using calibration or partial prediction accuracy.",
            "native_api_check": str(smoke) if smoke else "pending; this is preflight only",
        })
        return dry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("preparation")
    parser.add_argument("destination")
    parser.add_argument("--smoke")
    parser.add_argument("--diagnostic-ledger", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.preparation, args.destination, args.smoke, args.diagnostic_ledger), indent=2))
