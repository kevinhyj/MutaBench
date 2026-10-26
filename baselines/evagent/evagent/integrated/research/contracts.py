"""Immutable scientific evidence, bounded model budgets and checked claims."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from ..bundles import SpecialistInput
from ..evidence import EvidenceBundle
from ..gateway import assert_public, digest
from ..tasks import file_hash

SCHEMA = "evagent-research-snapshot-v1"
SOURCES = ("literature", "structure", "msa")
CONDITIONS = ("reference", "single", "multi")
ANALYSIS_CONFIG = {
    "backend": "direct",
    "modes": list(CONDITIONS),
    "tools": ["paperqa_frozen_search", "barnaba", "viennarna", "gemmi", "frozen_msa"],
    "specialist_model_calls": 6,
    "specialist_tool_calls": 12,
    "single_model_calls": 18,
    "single_tool_calls": 36,
    "live_retrieval_during_analysis": False,
    "embedding": "paperqa-local-sparse-256",
    "tool_protocol": "bounded-json-action-v1",
}


@dataclass(frozen=True)
class ResearchSpecialistInput(SpecialistInput):
    """Stage D extension of the original label-free specialist boundary."""

    metadata: dict = dataclass_field(default_factory=dict)


class ResearchEvidenceBundle(EvidenceBundle):
    """Original JSON evidence interface plus frozen tools, assets and scientific provenance."""


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def load_snapshot(path: Path, task=None):
    value = json.loads(path.read_text())
    assert_public(value)
    if value.get("schema") != SCHEMA or value.get("source_audit", {}).get("status") != "admitted":
        raise ValueError("Research evidence requires an admitted source audit")
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Research snapshot checksum mismatch")
    for asset in value.get("assets", []):
        p = (path.parent / asset["path"]).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_hash(p) != asset["sha256"]:
            raise ValueError("Research asset escaped root, changed or missing")
    from .literature import audit_document

    if value.get("literature_policy") not in {
        None, "passage_review_v2",
        "entity-title-match; benchmark/related DOI and title exclusion; conservative full-visible-text contamination filter",
    }:
        raise ValueError("Unknown frozen literature policy")
    for document in value.get("literature_documents", []):
        if value.get("literature_policy") == "passage_review_v2":
            from .literature_review import validate_passage
            validate_passage(document, value.get("denied_dois", []), value.get("denied_titles", []))
        elif audit_document(document, value.get("denied_dois", []), value.get("denied_titles", [])):
            raise ValueError("Frozen literature failed contamination or provenance validation")
    if value.get("msa_usage") is not None:
        from .msa_agent import validate_usage
        validate_usage(value["msa_usage"])
        if value["msa_usage"] == "prompt_only" and value.get("msa_values") is not None:
            raise ValueError("Prompt-only MSA must not also supply fusion scores")
    if set(value["sources"]) != set(SOURCES):
        raise ValueError("Snapshot must declare all evidence sources")
    if task and (
        value["assay"] != task.assay
        or value["wt_sha256"] != digest(task.wt_sequence)
        or value["variants_sha256"] != digest(task.variants)
    ):
        raise ValueError("Research snapshot/task mismatch")
    if task:
        wt = task.wt_sequence.replace("T", "U")
        for name, source in value["sources"].items():
            if source.get("status") not in {"available", "unavailable"}:
                raise ValueError("Invalid evidence availability")
            if source["status"] == "unavailable" and not source.get("reason"):
                raise ValueError("Absent evidence requires a reason")
            for pos, record in source.get("positions", {}).items():
                if not str(pos).isdigit() or not 1 <= int(pos) <= len(wt):
                    raise ValueError("Evidence position outside WT coordinates")
                if name == "msa" and record.get("wt", wt[int(pos) - 1]) != wt[int(pos) - 1]:
                    raise ValueError("MSA reference base differs from WT")
        geometry = value.get("geometry", {})
        mapped = {int(p) for p in geometry.get("positions", {})}
        if not mapped <= set(range(1, len(wt) + 1)):
            raise ValueError("Geometry position outside WT coordinates")
        for annotation in geometry.get("annotations", []):
            if not set(annotation["positions"]) <= mapped:
                raise ValueError("Geometry annotation cites an unmapped residue")
        for record in geometry.get("positions", {}).values():
            angles = record.get("backbone_angles_radians")
            if angles is not None and len(angles) != 7:
                raise ValueError("Expected seven Barnaba torsions including chi")
    return ResearchEvidenceBundle(value)


class CallLimit(Exception):
    """Local research allowance exhausted, not a provider failure."""


class BoundedModel:
    def __init__(self, gateway, metadata, maximum):
        self.gateway, self.metadata, self.maximum = gateway, metadata, maximum
        self.records = []

    def generate(self, system, user, phase="analysis"):
        if len(self.records) >= self.maximum:
            raise CallLimit("Model-call allowance exhausted")
        result = self.gateway.generate(
            system=system,
            user=user,
            run_name="research-" + phase,
            metadata={**self.metadata, "phase": phase, "model_turn": len(self.records),
                      **({'provider_attempt_limit': self.maximum - len(self.records)}
                         if getattr(self.gateway, 'connection_retry_policy', None) else {})},
        )
        self.records.extend(result.get('transport_attempt_records') or [
            {
                k: result.get(k)
                for k in ("request_sha256", "status", "cache_hit", "usage_summary", "elapsed_s")
            }
        ])
        if result["status"] != "ok":
            raise ResearchFailure(result["status"])
        return result["final_text"]


class ResearchFailure(Exception):
    """Known provider failure or invalid research output; never a semantic CR repair."""


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def validate_claims(value, evidence_ids, candidates, positions):
    if not isinstance(value, dict) or set(value) - {"claims", "limitations", "abstain"}:
        raise ValueError("Expected claims, limitations and abstain")
    if type(value.get("abstain")) is not bool or not isinstance(value.get("claims"), list):
        raise ValueError("Invalid specialist output")
    if not isinstance(value.get("limitations"), list) or not all(
        isinstance(x, str) for x in value["limitations"]
    ):
        raise ValueError("Explicit limitations required")
    for claim in value["claims"]:
        if set(claim) != {"text", "evidence_ids", "candidate_ids", "positions"}:
            raise ValueError("Invalid claim fields")
        if not isinstance(claim["text"], str) or not claim["text"].strip():
            raise ValueError("Empty claim")
        for field, allowed in (
            ("evidence_ids", evidence_ids),
            ("candidate_ids", candidates),
            ("positions", positions),
        ):
            if not isinstance(claim[field], list) or any(x not in allowed for x in claim[field]):
                raise ValueError("Unsupported claim citation or coordinate")
        if any(type(position) is not int for position in claim["positions"]):
            raise ValueError("Claim positions must be integer WT coordinates")
        if not claim["evidence_ids"]:
            raise ValueError("Every claim requires observed evidence")
    if not value["claims"] and not value["abstain"]:
        raise ValueError("Empty output must explicitly abstain")
    assert_public(value)
    return value


def fuse(ids, cr, msa):
    from scipy.stats import rankdata

    if set(cr) != set(ids) or any(not math.isfinite(cr[x]) for x in ids):
        raise ValueError("Fusion requires complete finite CR predictions")
    a = rankdata([cr[x] for x in ids], method="average")
    if msa is not None:
        if set(msa) != set(ids) or any(not math.isfinite(msa[x]) for x in ids):
            raise ValueError("Fusion requires complete finite MSA values")
        a = (a + rankdata([msa[x] for x in ids], method="average")) / 2
    return {x: float(v) for x, v in zip(ids, a)}
