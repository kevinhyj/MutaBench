"""Whitelist-only tools over an immutable snapshot; no shell, filesystem or live search."""

from __future__ import annotations

import json
import os
import time

from ..gateway import digest
from .literature import PaperEvidence


class FrozenTools:
    def __init__(self, task, snapshot, role, model):
        self.task, self.snapshot, self.role, self.model = task, snapshot, role, model
        self.sources = ["msa", "structure", "literature"] if role == "single" else [role]
        if snapshot.get("msa_usage") == "external_fusion":
            self.sources = [s for s in self.sources if s != "msa"]
            if role == "msa":
                raise ValueError("MSA expert builds evidence during preparation, not fusion-time prompting")
        self.records, self.results, self.observed = [], [], set()
        self.variants = {v["variant_id"]: v for v in task["variants"]}
        self.paperqa = PaperEvidence(snapshot.get("literature_documents", []),
                                     passage_policy=snapshot.get("literature_policy") == "passage_review_v2")
        self.descriptions = {
            "evidence": {
                "description": "Read frozen task, site and candidate evidence, including source limitations.",
                "arguments": {
                    "source": {"type": "string", "enum": self.sources,
                               "description": "Choose exactly one source string per request, never a list."},
                    "candidate_ids": "optional list of supplied IDs",
                    "positions": "optional 1-based WT positions",
                },
            }
        }
        if "literature" in self.sources:
            self.descriptions["literature_search"] = {
                "description": "PaperQA retrieval and relevance summaries from reviewed frozen papers only.",
                "arguments": {
                    "query": "background question, no candidate mutations or effect queries"
                },
            }
        if "structure" in self.sources:
            self.descriptions["structure_geometry"] = {
                "description": "Barnaba annotations and Gemmi geometry around WT positions; no mutant 3D inference.",
                "arguments": {"positions": "list of 1-based WT positions"},
            }

    def missing_required_tools(self):
        reads = {
            r.get("arguments", {}).get("source")
            for r in self.records
            if r["tool"] == "evidence" and r["status"] == "ok"
        }
        missing = ["evidence(source=" + s + ")" for s in self.sources if s not in reads]
        names = {r["tool"] for r in self.records if r["status"] == "ok"}
        if (
            "structure" in self.sources
            and self.snapshot.get("geometry", {}).get("status") == "available"
            and "structure_geometry" not in names
        ):
            missing.append("structure_geometry")
        if (
            "literature" in self.sources
            and self.snapshot.get("literature_documents")
            and "literature_search" not in names
        ):
            missing.append("literature_search")
        return missing

    def validate_report(self, report):
        if self.snapshot.get("literature_policy") != "passage_review_v2":
            return
        passages = {e["id"]: e for r in self.results for e in r.get("evidence", []) if "excerpt" in e}
        for claim in report["claims"]:
            if any(i.startswith("literature:") for i in claim["evidence_ids"]):
                raise ValueError("Literature findings require retrieved excerpts, not catalog metadata")
            cited = [passages[i] for i in claim["evidence_ids"] if i in passages]
            if cited and (claim["candidate_ids"] or claim["positions"]):
                raise ValueError("Reviewed literature supplies task context; candidate/coordinate claims require a separate audited mapping")

    def call(self, name, arguments):
        started = time.time()
        entry = {"tool": name, "arguments": arguments, "status": "inflight"}
        self.records.append(entry)
        journal = self.model.gateway.run_dir.parent / "tool_events"
        journal.mkdir(parents=True, exist_ok=True)
        journal_path = journal / (digest(self.model.metadata) + ".jsonl")

        def record_event(value):
            with journal_path.open("a") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        record_event(entry)
        try:
            if name not in self.descriptions or not isinstance(arguments, dict):
                raise ValueError("Unknown or unauthorized tool")
            if name == "literature_search":
                if set(arguments) != {"query"}:
                    raise ValueError("Expected query")
                result = self.paperqa.search(arguments["query"], self.model)
            else:
                allowed = (
                    {"positions"}
                    if name == "structure_geometry"
                    else {"source", "positions", "candidate_ids"}
                )
                if set(arguments) - allowed:
                    raise ValueError("Unknown tool argument")
                ids = arguments.get("candidate_ids", list(self.variants))
                if not isinstance(ids, list) or set(ids) - self.variants.keys():
                    raise ValueError("Candidate outside the current CR group")
                positions = arguments.get(
                    "positions",
                    sorted(
                        {
                            i + 1
                            for v in self.variants.values()
                            for i, (a, b) in enumerate(zip(self.task["wt_sequence"], v["sequence"]))
                            if a != b
                        }
                    ),
                )
                if not isinstance(positions, list) or any(
                    type(p) is not int or not 1 <= p <= len(self.task["wt_sequence"])
                    for p in positions
                ):
                    raise ValueError("Position outside WT coordinates")
                source = "structure" if name == "structure_geometry" else arguments.get("source")
                if source not in self.sources:
                    raise ValueError("Source unavailable to this role")
                if name == "structure_geometry":
                    raw = self.snapshot.get("geometry", {})
                    data = {
                        "status": raw.get("status", "unavailable"),
                        "reason": raw.get("reason"),
                        "annotations": [
                            r
                            for r in raw.get("annotations", [])
                            if set(r.get("positions", [])) & set(positions)
                        ],
                        "positions": {
                            str(p): raw.get("positions", {}).get(str(p), {"status": "unmapped"})
                            for p in positions
                        },
                        "provenance": raw.get("provenance", []),
                        "limitations": raw.get("limitations", []),
                    }
                else:
                    raw = self.snapshot["sources"][source]
                    data = {
                        "status": raw["status"],
                        "reason": raw.get("reason"),
                        "task": raw.get("task", {}),
                        "positions": {
                            str(p): raw.get("positions", {}).get(str(p), {"status": "unmapped"})
                            for p in positions
                        },
                        "candidates": {
                            i: raw.get("candidates", {}).get(
                                digest(self.variants[i]["sequence"]), {}
                            )
                            for i in ids
                        },
                        "provenance": raw.get("provenance", []),
                        "limitations": raw.get("limitations", []),
                    }
                result = {
                    "status": "ok",
                    "evidence": [{"id": source + ":" + digest(data)[:20], **data}],
                }
            self.observed.update(r["id"] for r in result.get("evidence", []))
            self.results.append(result)
            entry.update(status="ok", result_sha256=digest(result), elapsed_s=time.time() - started)
            record_event({**entry, "result": result})
            return result
        except BaseException as error:
            entry.update(
                status="error", error_type=type(error).__name__, elapsed_s=time.time() - started
            )
            record_event(entry)
            raise
