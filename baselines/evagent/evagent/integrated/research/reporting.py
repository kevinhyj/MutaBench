"""Read-only accounting and evidence QA; unknown usage is never counted as zero."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .contracts import save


def audit_report(root, manifest, completed):
    path = root / "provider/request_ledger.sqlite"
    groups = defaultdict(
        lambda: {
            "provider_attempts": 0,
            "reported_tokens": 0,
            "unknown_usage_attempts": 0,
            "elapsed_s": 0.0,
        }
    )
    tasks = defaultdict(list)
    if path.exists():
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            for aid, request_hash, status, raw, tokens, elapsed in db.execute(
                'SELECT id,request_sha256,status,request_json,tokens,elapsed_s FROM attempts WHERE status != "not_submitted" ORDER BY id'
            ):
                meta = json.loads(raw).get("metadata", {})
                key = tuple(meta.get(k, "unspecified") for k in ("phase", "condition", "role"))
                group = groups[key]
                group["provider_attempts"] += 1
                group["unknown_usage_attempts"] += tokens is None or "unknown" in status
                group["reported_tokens"] += tokens or 0
                group["elapsed_s"] += elapsed or 0
                tasks[meta.get("assay", "unspecified")].append(
                    {
                        "shared_ledger": str(path),
                        "attempt_id": aid,
                        "request_sha256": request_hash,
                        "status": status,
                        "metadata": meta,
                        "reported_tokens": tokens,
                        "elapsed_s": elapsed,
                    }
                )
    for assay, rows in tasks.items():
        save(root / "task_ledgers" / (assay + ".json"), rows)
    costs = [
        {**dict(zip(("phase", "condition", "role"), key)), **value}
        for key, value in sorted(groups.items())
    ]
    quality = []
    for r in manifest["tasks"]:
        snap = json.loads(Path(r["snapshot_file"]).read_text())
        docs = snap.get("literature_documents", [])
        quality.append(
            {
                "assay": r["assay"],
                "family": r["family"],
                "msa_fusion_available": snap.get("msa_values") is not None,
                "reviewed_documents": len(docs),
                "document_relevance": dict(Counter(d["relevance"] for d in docs)),
                "fulltext_documents": sum(d["location"] != "abstract" for d in docs),
                "geometry_status": snap.get("geometry", {}).get("status", "unavailable"),
                "geometry_mapped_positions": len(snap.get("geometry", {}).get("positions", {})),
                "geometry_annotations": len(snap.get("geometry", {}).get("annotations", [])),
                "msa_site_support": dict(
                    Counter(
                        p.get("support", "unspecified")
                        for p in snap["sources"]["msa"].get("positions", {}).values()
                    )
                ),
            }
        )
    states = Counter()
    tools = Counter()
    claims = 0
    traceable = 0
    cached = 0
    for unit in completed.values():
        for call in unit.get("raw_tool_calls", []):
            tools[call["tool"] + ":" + call["status"]] += 1
        for report in unit.get("reports", {}).values():
            states[report["role"] + ":" + report["status"]] += 1
            for call in report.get("tool_calls", []):
                tools[call["tool"] + ":" + call["status"]] += 1
            cached += sum(bool(c.get("cache_hit")) for c in report.get("model_calls", []))
            ids = {
                e["id"]
                for out in report.get("observed_evidence", [])
                for e in out.get("evidence", [])
            }
            for claim in report.get("claims", []):
                claims += 1
                traceable += bool(claim["evidence_ids"]) and set(claim["evidence_ids"]) <= ids
    event_counts = Counter()
    for journal in (root / "tool_events").glob("*.jsonl"):
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            event_counts[event["status"]] += 1
    proposals = Counter()
    for journal in (root / "framework_tool_proposals").glob("*.jsonl"):
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            proposals["admitted" if event.get("admitted") else "not_admitted"] += 1
    return {
        "prior_ledger_cost_including_preparation_and_recovery": manifest["prior_usage"],
        "execution_cost_by_phase_condition_role": costs,
        "cost_note": "Actual provider attempts only; cache hits do not debit tokens. Unknown or partial usage is explicitly counted; reported token sums are lower bounds while any such attempt remains unresolved. Preparation is shared across conditions.",
        "evidence_quality": quality,
        "analyst_statuses": dict(states),
        "completed_unit_tool_calls": dict(tools),
        "durable_tool_events_including_interrupted": dict(event_counts),
        "native_tool_proposals_including_invalid_arguments": dict(proposals),
        "tool_accounting_note": "Native proposals, local executions, and inflight/settled event transitions are separate counts. Proposals include final structured-report calls; provider attempts are charged once in the ledger.",
        "cached_analysis_calls": cached,
        "claim_citation_traceability": {
            "claims": claims,
            "linked_to_observed_evidence": traceable,
            "rate": traceable / claims if claims else None,
            "semantic_support_rate": None,
            "semantic_support_status": "Not established by ID validation; requires passage-level review",
        },
        "no_msa_subgroup": [r["assay"] for r in quality if not r["msa_fusion_available"]],
    }
