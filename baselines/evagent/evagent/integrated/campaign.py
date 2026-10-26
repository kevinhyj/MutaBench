"""Frozen serial experiment matrix with one authoritative provider ledger."""
from __future__ import annotations

import fcntl
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .config import ExperimentConfig
from .experiment import _save, prepare, run_experiment
from .gateway import AdmissionPaused, RequestGateway, digest
from .tasks import file_hash


def resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def prepare_campaign(path: Path):
    raw = json.loads(path.read_text())
    if set(raw) - {"schema", "output_dir", "token_budget", "model_config", "cells", "calibration_assay", "prior_ledgers"}:
        raise ValueError("Unknown campaign option")
    if raw.get("schema") != "evagent-campaign-v1":
        raise ValueError("Invalid campaign schema")
    if type(raw.get("token_budget")) is not int or raw["token_budget"] <= 0:
        raise ValueError("Campaign requires an explicit positive total token budget")
    from .gateway import public_model_config
    model = public_model_config(raw["model_config"])
    if model["max_output_tokens"] is None:
        raise ValueError("Campaign requires explicit max_tokens for budget admission")
    cells, seen = [], set()
    for cell in raw["cells"]:
        key = (cell["assay"], cell["n"], cell["condition"])
        if key in seen:
            raise ValueError("Duplicate campaign cell")
        seen.add(key)
        row = dict(cell)
        if row["status"] == "eligible":
            if row.get("source_status") != "admitted":
                raise ValueError("Prediction requires an admitted source audit")
            config = ExperimentConfig.load(resolve(path.parent, cell["config"]))
            if config.values["backend"] != "direct" or config.values["memory_mode"] != "none":
                raise ValueError("Stage B requires direct backend and memory disabled")
            if public_model_config(config.model_config()) != model:
                raise ValueError("Campaign cell silently changes model settings")
            if config.values["token_budget"] != raw["token_budget"]:
                raise ValueError("Cell and campaign token budgets differ")
            task, manifest, _ = prepare(config)
            if task.assay != row["assay"] or len(task.case.ids) != row["n"]:
                raise ValueError("Campaign cell/task mismatch")
            row.update(config=str(config.source), fingerprint=manifest["fingerprint"],
                       config_sha256=file_hash(config.source),
                       initial_requests=len(manifest["initial_requests"]))
        elif row["status"] not in {"source_excluded", "source_unresolved", "insufficient_samples", "evidence_unavailable"}:
            raise ValueError("Invalid cell eligibility")
        elif not row.get("reason"):
            raise ValueError("Skipped cells require an explicit reason")
        cells.append(row)
    outputs = [str(ExperimentConfig.load(Path(c["config"])).values["output_dir"])
               for c in cells if c["status"] == "eligible"]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Campaign cells share an output directory")
    manifest = {**raw, "cells": cells, "model_config": model}
    manifest['prior_usage'] = read_prior_ledgers(raw.get('prior_ledgers', []), path.parent)
    manifest["fingerprint"] = digest(manifest)
    return manifest, resolve(path.parent, raw["output_dir"])


def read_prior_ledgers(records: list[dict], base: Path, *, allow_manual_recovery=False) -> dict:
    """Carry forward known costs from immutable, closed calibration ledgers."""
    usage = {'reported_tokens': 0, 'provider_attempts': 0, 'provider_elapsed_s': 0,
             'requests_by_status': {}, 'ledgers': []}
    seen = set()
    referenced = {str(resolve(base, r['path'])) for r in records}
    for record in records:
        p = resolve(base, record['path'])
        if str(p) in seen or file_hash(p) != record['sha256']:
            raise ValueError('Prior ledger duplicate or checksum mismatch')
        seen.add(str(p))
        if Path(str(p)+'-wal').exists() and Path(str(p)+'-wal').stat().st_size:
            raise ValueError('Prior ledger must be closed and checkpointed')
        with closing(sqlite3.connect(f'file:{p}?mode=ro&immutable=1', uri=True)) as db:
            inherited = db.execute("SELECT value FROM run_state WHERE key='prior_usage'").fetchone()
            if inherited:
                ancestors = json.loads(inherited[0]).get('ledgers', [])
                if any(str(resolve(base, a['path'])) not in referenced for a in ancestors):
                    raise ValueError('Include the complete prior ledger lineage without omitting costs')
            rows = db.execute("SELECT status,tokens,elapsed_s,request_sha256 FROM attempts WHERE status!='not_submitted'").fetchall()
            unresolved = sorted(h for status, tokens, _, h in rows
                                if status not in {'ok', 'blocked', 'truncated', 'terminal_failure'}
                                or type(tokens) is not int)
            acknowledgment = record.get('manual_recovery')
            if acknowledgment is not None:
                if (not allow_manual_recovery or not unresolved
                        or set(acknowledgment) != {'request_sha256s', 'authorization', 'recorded_at', 'usage_status'}
                        or acknowledgment['request_sha256s'] != unresolved
                        or acknowledgment['usage_status'] != 'unresolved_lower_bound'
                        or not isinstance(acknowledgment['authorization'], str)
                        or not acknowledgment['authorization'].strip()
                        or not isinstance(acknowledgment['recorded_at'], str)
                        or not acknowledgment['recorded_at'].strip()
                        or any(status == 'inflight' for status, _, _, _ in rows)):
                    raise ValueError('Invalid scoped manual recovery acknowledgment')
            for status, tokens, elapsed, request_hash in rows:
                if status not in {'ok', 'blocked', 'truncated', 'terminal_failure'} or type(tokens) is not int:
                    if acknowledgment is None:
                        raise ValueError('Unknown prior response/cost cannot be resubmitted in another campaign')
                    usage['unknown_usage_attempts'] = usage.get('unknown_usage_attempts', 0) + 1
                    usage['usage_is_lower_bound'] = True
                if tokens is not None and (type(tokens) is not int or tokens < 0):
                    raise ValueError('Invalid prior token value')
                usage['reported_tokens'] += tokens or 0
                usage['provider_attempts'] += 1
                usage['provider_elapsed_s'] += elapsed or 0
                usage['requests_by_status'][status] = usage['requests_by_status'].get(status, 0)+1
        usage['ledgers'].append({'path': str(p), 'sha256': record['sha256'],
                                **({'manual_recovery': acknowledgment} if acknowledgment else {})})
    return usage


class CampaignTransport:
    def __init__(self, gateway: RequestGateway):
        self.gateway = gateway

    def __call__(self, request: dict) -> dict:
        if self.gateway.summary()["paused"]:
            raise AdmissionPaused("Campaign budget exhausted or unresolved provider response")
        if request["model_config"] != self.gateway.model_config:
            raise ValueError("Campaign transport model mismatch")
        return self.gateway.invoke_messages(
            messages=request["messages"], run_name="campaign-cell",
            metadata={**request["metadata"], "cell_fingerprint": request["fingerprint"]},
        )


def summarize(rows: list[dict], ledger: dict) -> dict:
    from collections import Counter
    statuses = dict(Counter(r["status"] for r in rows))
    comparisons = []
    for n in sorted({r["n"] for r in rows}):
        baselines = {r["assay"]: r for r in rows if r["n"] == n and r["condition"] == "cr" and r["status"] == "ok"}
        for condition in ("msa", "structure", "literature", "all"):
            paired = []
            for r in rows:
                if r["condition"] != condition or r["n"] != n or r["status"] != "ok":
                    continue
                b = baselines.get(r["assay"])
                if not b:
                    continue
                a, z = r.get("metrics", {}).get("spearman"), b.get("metrics", {}).get("spearman")
                if a is not None and z is not None:
                    paired.append({"assay": r["assay"], "family": r["family"], "delta_spearman": a-z,
                                   "delta_kendall": r["metrics"]["kendall_tau"]-b["metrics"]["kendall_tau"]})
            interval = None
            if paired:
                import numpy as np
                families = sorted({p["family"] for p in paired})
                means = [float(np.mean([p["delta_spearman"] for p in paired if p["family"] == f])) for f in families]
                if len(means) >= 2:
                    draws = np.random.default_rng(1).choice(means, (2000, len(means)), replace=True).mean(axis=1)
                    interval = [float(x) for x in np.quantile(draws, [.025, .975])]
                comparisons.append({"n": n, "condition": condition, "paired_assays": len(paired),
                                    "families": len(families), "family_macro_delta_spearman": float(np.mean(means)),
                                    "family_macro_delta_kendall": float(np.mean([np.mean([p["delta_kendall"] for p in paired if p["family"] == f]) for f in families])),
                                    "family_bootstrap_95pct": interval, "pairs": paired})
    eligible = sum(r["status"] not in {"source_excluded", "source_unresolved", "insufficient_samples", "evidence_unavailable"} for r in rows)
    return {"statuses": statuses, "planned_cells": len(rows), "eligible_cells": eligible,
            "completion_fraction_of_eligible": statuses.get("ok", 0)/eligible if eligible else None,
            "complete_cells": statuses.get("ok", 0),
            "ledger": ledger, "comparisons": comparisons, "rows": rows}


def run_campaign(path: Path, *, dry_run: bool = False, transport=None) -> dict:
    manifest, root = prepare_campaign(path)
    if dry_run:
        return {"status": "dry_run", "fingerprint": manifest["fingerprint"],
                "cells": len(manifest["cells"]), "eligible": sum(c["status"] == "eligible" for c in manifest["cells"]),
                "initial_requests": sum(c.get("initial_requests", 0) for c in manifest["cells"]),
                "token_budget": manifest["token_budget"], "api_calls": 0, "writes": 0}
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mpath = root / "campaign_manifest.json"
        if mpath.exists() and json.loads(mpath.read_text())["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Campaign inputs changed; use a new campaign directory")
        _save(mpath, manifest)
        gateway = RequestGateway(model_config=manifest["model_config"], run_dir=root / "provider",
                                 fingerprint=manifest["fingerprint"], token_budget=manifest["token_budget"], transport=transport,
                                 reserve_budget=True, prior_usage=manifest['prior_usage'])
        results = []
        state = "complete"
        try:
            # This first real request also shares the total budget and persistent pause state.
            smoke = gateway.generate(system="API connectivity check.", user="Reply with exactly OK.", run_name="connectivity")
            if smoke["status"] != "ok" or smoke["final_text"].strip() != "OK":
                gateway.pause("Connectivity preflight failed")
            calibration = manifest.get("calibration_assay")
            ordered = sorted(manifest["cells"], key=lambda c: (not (c["assay"] == calibration and c["n"] == 50), c["assay"], c["n"], c["condition"]))
            calibration_checked = not calibration
            for cell in ordered:
                if not calibration_checked and not (cell["assay"] == calibration and cell["n"] == 50):
                    used = gateway.summary()
                    calibration_rows = [r for r in results if r['status'] == 'ok']
                    if len(calibration_rows) != 5:
                        gateway.pause("Calibration requires all five conditions to complete")
                    remaining = sum(c.get('initial_requests', 0) for c in ordered
                                    if c['status'] == 'eligible' and not (c['assay'] == calibration and c['n'] == 50))
                    current_attempts = used['provider_attempts'] - manifest['prior_usage']['provider_attempts']
                    per_call = used['current_run_reported_tokens']/max(1, current_attempts)
                    projection = used['reported_tokens'] + remaining*2*per_call
                    _save(root/'calibration_gate.json', {'status': 'passed' if projection <= manifest['token_budget'] else 'paused',
                        'reported_tokens': used['reported_tokens'], 'projection_with_one_repair_per_remaining_request': projection,
                        'decision_inputs': ['format completeness', 'accounting', 'token cost'],
                        'accuracy_used_for_decision': False})
                    if projection > manifest['token_budget']:
                        gateway.pause("Calibration cost projection exceeds frozen campaign budget")
                    calibration_checked = True
                row = dict(cell)
                if row["status"] == "eligible":
                    config = ExperimentConfig.load(Path(cell["config"]))
                    _, fresh, _ = prepare(config)
                    if file_hash(config.source) != cell["config_sha256"] or fresh["fingerprint"] != cell["fingerprint"]:
                        gateway.pause("Frozen campaign cell changed after preflight")
                        raise AdmissionPaused("Frozen campaign cell changed after preflight")
                    result = run_experiment(config, transport=CampaignTransport(gateway))
                    row.update(status=result["status"], metrics=result.get("metrics", {}), ledger=result.get("ledger", {}))
                    results.append(row)
                    processed = {(r["assay"], r["n"], r["condition"]) for r in results}
                    pending = [{**c, "status": "not_run" if c["status"] == "eligible" else c["status"]}
                               for c in manifest["cells"] if (c["assay"], c["n"], c["condition"]) not in processed]
                    _save(root / "progress.json", summarize(results + pending, gateway.summary()))
                    print(json.dumps({"assay": cell["assay"], "n": cell["n"], "condition": cell["condition"],
                                      "status": row["status"], "tokens": gateway.summary()["reported_tokens"]}), flush=True)
                    if row["status"] == "paused" or gateway.summary()["paused"]:
                        state = "paused"
                        break
                    if cell["assay"] == calibration and cell["n"] == 50 and row["status"] != "ok":
                        state = "calibration_failed"
                        break
                else:
                    results.append(row)
        except Exception as error:
            state = "paused"
            _save(root / "failure.json", {"error_type": type(error).__name__, "reason": str(error)})
        finally:
            ledger = gateway.summary()
            gateway.close()
        seen = {(r["assay"], r["n"], r["condition"]) for r in results}
        results += [{**c, "status": "not_run" if c["status"] == "eligible" else c["status"]}
                    for c in manifest["cells"] if (c["assay"], c["n"], c["condition"]) not in seen]
        report = {"status": state, **summarize(results, ledger)}
        _save(root / "report.json", report)
        return report


def command(path: Path, *, dry_run: bool) -> int:
    result = run_campaign(path, dry_run=dry_run)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "comparisons"}}, indent=2))
    return 0 if result["status"] in {"complete", "dry_run"} else 2
