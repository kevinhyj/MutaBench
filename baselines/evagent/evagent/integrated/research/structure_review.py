"""Versioned tool-using structure reviewer with equal single/multi permissions."""

from __future__ import annotations

import json
import os
import time

from ..gateway import ExecutionPaused, assert_public, digest
from .contracts import save
from .review import REVIEW_DESIGN, ROLE_SOURCES, ReviewTools, _rounded
from .review import SYSTEM as REVIEW_SYSTEM
from .structure_analysis import TOOL_PARAMETERS, StructureAnalysis

DESIGN = {
    **REVIEW_DESIGN,
    "profile": "tool_review_v3",
    "structure_tools": ["rna_secondary", "rna_contacts", "rna_annotations"],
    "structure_tool_parameters": TOOL_PARAMETERS,
    "structure_analysis": "on_demand_local_computation_on_frozen_assets",
    "required_structure_investigations": "complete secondary analysis; contacts and annotations at usable WT sites",
}
SYSTEM = (
    REVIEW_SYSTEM.replace(
        "registered evidence function", "registered evidence and structure-analysis functions"
    )
    + """
Structure review must use calculations, not intuition about a dot-bracket string or a
generic statement that stability implies activity. rna_secondary computes full WT and
full mutant folding: use pairing to locate lost/gained partners, ensemble to compare
MFE/centroid/MEA and uncertainty, and joint_changes to test structural nonadditivity
against single-change counterfactuals for suspicious multi-substitution candidates.
Structural nonadditivity is not measured functional epistasis. Never replace a joint
mutant calculation with a sum of single mutants. rna_contacts computes actual WT
heavy-atom distances; rna_annotations runs Barnaba for base pairing, stacking, and
torsions. Request those analyses for candidates whose changed WT sites have usable
geometry. You may narrow candidate_ids or positions and follow up with other modes.
Batch independent tool calls to reserve model calls for interpreting results.
Every candidate assigned to structure needs a complete (default positions) pairing
or ensemble analysis. If its changed sites have trusted WT coordinates, investigate
both contacts and annotations. Missing/low-confidence data must remain explicit.
The initial tool context lists which candidates require these investigations.
Every final candidate check must cite an actual computed structure result (or its
explicit unavailable result); a single reviewer must also cite sequence evidence.
Use specific mechanisms and counterevidence; do not request revision merely because
a metric changed. A retain decision may be scientifically appropriate after analysis.
No external searches, new 3D predictions, temperature tuning, conservation inputs,
or extra model calls outside the gateway are available.
"""
)


def declarations(role):
    from .review import declarations as evidence_declarations

    result = evidence_declarations(role)
    if "structure" not in ROLE_SOURCES[role]:
        return result
    shared = {
        "candidate_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 25},
        "positions": {"type": "array", "items": {"type": "integer"}, "maxItems": 12},
    }
    for name, description, extra in [
        (
            "rna_secondary",
            "Calculate complete WT/mutant pairing, ensemble, or joint-change counterfactuals using ViennaRNA. Omit positions for all changed sites.",
            {"mode": {"type": "string", "enum": ["pairing", "ensemble", "joint_changes"]}},
        ),
        (
            "rna_contacts",
            "Compute minimum WT heavy-atom distances around selected sites and between mutated sites using frozen mapped coordinates. Omit positions for changed sites.",
            {"radius_A": {"type": "number", "minimum": 3, "maximum": 12}},
        ),
        (
            "rna_annotations",
            "Run Barnaba and query WT base-pair/stacking classifications and backbone torsions at selected sites. Omit positions for changed sites.",
            {},
        ),
    ]:
        result.append(
            {
                "name": name,
                "description": description,
                "parameters_json_schema": {
                    "type": "object",
                    "properties": {**shared, **extra},
                    "additionalProperties": False,
                },
            }
        )
    return result


def normalize_calls(calls, role):
    if role not in ROLE_SOURCES or not isinstance(calls, list) or not 1 <= len(calls) <= 24:
        return None
    allowed = {d["name"] for d in declarations(role)}
    if any(
        not isinstance(c, dict)
        or c.get("name") not in allowed
        or not isinstance(c.get("args"), dict)
        for c in calls
    ):
        return None
    return {"actions": [{"tool": c["name"], "arguments": c["args"]} for c in calls]}


class ActiveReviewTools(ReviewTools):
    def __init__(self, task, snapshot, role, model, *, snapshot_path):
        super().__init__(task, snapshot, role, model)
        self.task, self.snapshot = task, snapshot
        self.backend = (
            StructureAnalysis(
                task, snapshot, snapshot_path, model.gateway.run_dir.parent / "structure_cache"
            )
            if "structure" in self.sources
            else None
        )
        self.invocations = {
            name: set() for name in ("rna_secondary", "rna_contacts", "rna_annotations")
        }
        self.required_geometry = set()
        if self.backend:
            allowed = set(self.backend.coverage["eligible_geometry_positions"])
            from .structure_analysis import canonical, changed

            self.required_geometry = {
                v["variant_id"]
                for v in task["variants"]
                if set(changed(self.backend.wt, canonical(v["sequence"])) or []) & allowed
            }
            # Old summary cards are optional orientation only, not the required analysis.
            for card in self.cards["structure"].values():
                card.pop("positions", None)
        self.model_context = {
            "tools": declarations(role),
            "parameters": TOOL_PARAMETERS,
            "geometry_coverage": self.backend.coverage if self.backend else None,
            "candidates_requiring_contacts_and_annotations": sorted(self.required_geometry),
            "default_candidate_selection": "all candidates in this group",
            "runtime": "local calculations, no external retrieval or hidden model calls",
        }

    def validate_completion(self, value):
        if not self.backend:
            return
        ids = set(self.backend.variants)
        if self.invocations["rna_secondary"] != ids:
            raise ValueError("Run complete secondary pairing/ensemble analysis for every candidate")
        for tool in ("rna_contacts", "rna_annotations"):
            if not self.required_geometry <= self.invocations[tool]:
                raise ValueError(
                    "Missing required " + tool + " analysis for mapped trusted changed sites"
                )
        for row in value["checks"]:
            refs = [self.observed[e] for e in row["evidence_ids"]]
            if not any(r.get("tool") in self.invocations for r in refs):
                raise ValueError("Candidate checks must cite an actual computed structure result")
            if self.role == "review_single" and not any(
                r.get("source") == "sequence" for r in refs
            ):
                raise ValueError(
                    "Single-review candidate checks must cite both sequence and computed structure evidence"
                )

    def call(self, name, args):
        maximum = 24 if self.role == "review_single" else 12
        if len(self.records) >= maximum:
            raise ValueError("Tool allowance exhausted")
        start = time.monotonic()
        event = {"tool": name, "arguments": args, "status": "inflight"}
        self.records.append(event)
        root = self.model.gateway.run_dir.parent
        journal = root / "tool_events" / (digest(self.model.metadata) + ".jsonl")
        journal.parent.mkdir(parents=True, exist_ok=True)

        def record(value):
            if getattr(self.model, "external_tool_journal", False):
                return
            with journal.open("a") as handle:
                handle.write(json.dumps(value, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

        record(event)
        before = len(self.backend.cache.events) if self.backend else 0
        try:
            schema = next(
                (d["parameters_json_schema"] for d in declarations(self.role) if d["name"] == name),
                None,
            )
            if (
                schema is None
                or not isinstance(args, dict)
                or set(args) - set(schema["properties"])
            ):
                raise ValueError("Unknown tool or unsupported arguments")
            ids = args.get("candidate_ids", [v["variant_id"] for v in self.task["variants"]])
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 25
                or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids)
                or set(ids) - {v["variant_id"] for v in self.task["variants"]}
            ):
                raise ValueError("Invalid candidate selection")
            if name == "evidence":
                source = args.get("source")
                if source not in self.sources:
                    raise ValueError("Source not assigned to this role")
                cards = [self.cards[source][i] for i in ids]
                context = self.context[source]
            else:
                source = "structure"
                if name == "rna_secondary":
                    cards = self.backend.secondary(
                        ids, mode=args.get("mode", "pairing"), positions=args.get("positions")
                    )
                else:
                    cards = self.backend.tertiary(
                        ids,
                        mode="contacts" if name == "rna_contacts" else "annotations",
                        positions=args.get("positions"),
                        radius_A=args.get("radius_A", 5.0),
                    )
                context = {
                    "snapshot_sha256": self.snapshot["sha256"],
                    "parameters": TOOL_PARAMETERS,
                    "tool_versions": self.backend.cache.signature["versions"],
                    "representation": "text_and_numeric",
                }
                computations = self.backend.cache.events[before:]
                for card in cards:
                    card.update(source=source, tool=name)
                    card["id"] = (
                        "computed:"
                        + name
                        + ":"
                        + digest(
                            {
                                "card": card,
                                "arguments": args,
                                "snapshot": self.snapshot["sha256"],
                                "runtime": self.backend.cache.signature,
                            }
                        )
                    )
                    relevant = {
                        card.get("wt_sequence_sha256"),
                        card.get("candidate_sequence_sha256"),
                    }
                    relevant.update(
                        s["sequence_sha256"] for s in card.get("single_change_counterfactuals", [])
                    )
                    card["computation_keys"] = list(
                        dict.fromkeys(
                            c["computation_key"]
                            for c in computations
                            if c["input_sequence_sha256"] is None
                            or c["input_sequence_sha256"] in relevant
                        )
                    )
                # Narrow follow-up queries do not count as the required complete first pass.
                if args.get("positions") is None and (
                    name != "rna_secondary"
                    or args.get("mode", "pairing") in {"pairing", "ensemble"}
                ):
                    self.invocations[name].update(ids)
            result = _rounded(
                {
                    "status": "ok",
                    "source": source,
                    "tool": name,
                    "context": context,
                    "evidence": cards,
                }
            )
            assert_public(result)
            self.coverage[source].update(ids)
            self.observed.update({c["id"]: c for c in result["evidence"]})
            self.results.append(result)
            artifact = root / "structure_tool_results" / (digest(result) + ".json")
            save(artifact, {"result": result, "sha256": digest(result)})
            event.update(
                status="ok",
                result_sha256=digest(result),
                result_artifact=str(artifact.relative_to(root)),
                computations=self.backend.cache.events[before:] if self.backend else [],
            )
            return result
        except ExecutionPaused:
            event["status"] = "integrity_pause"
            raise
        except Exception as exc:
            event["status"] = type(exc).__name__
            # Unexpected scientific-tool failures are not repaired by another model request.
            if not isinstance(exc, (ValueError, TypeError, KeyError)):
                raise ExecutionPaused("Structure analysis failed: " + type(exc).__name__) from exc
            raise
        finally:
            event["elapsed_s"] = time.monotonic() - start
            record(event)


def factory(snapshot_path):
    def build(task, snapshot, role, model):
        return ActiveReviewTools(task, snapshot, role, model, snapshot_path=snapshot_path)

    return build
