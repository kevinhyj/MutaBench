"""Frozen, label-free multi-source evidence and future specialist contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .gateway import assert_public, digest


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

SOURCES = {"msa", "structure", "literature"}
SCHEMA = "evagent-frozen-evidence-v1"


@dataclass(frozen=True)
class SpecialistInput:
    assay: str
    source: str
    evidence: dict
    candidate_ids: tuple[str, ...]
    wt_sequence: str = ""
    variants: tuple[dict, ...] = ()

    def __post_init__(self):
        assert_public(asdict(self))


class Specialist(Protocol):
    def analyze(self, task: SpecialistInput) -> dict:
        """Return claims, evidence_ids, uncertainty and call_records; no labels."""
        ...


def validate_bundle(value: dict, *, root: Path | None = None) -> dict:
    if value.get("schema") != SCHEMA or not isinstance(value.get("assay"), str):
        raise ValueError("Invalid frozen evidence schema/assay")
    assert_public(value)
    json.dumps(value, allow_nan=False)
    payload = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != digest(payload):
        raise ValueError("Frozen evidence checksum mismatch")
    if value.get("source_status") != "admitted":
        raise ValueError("Evidence requires an admitted source audit")
    if set(value.get("sources", {})) != SOURCES:
        raise ValueError("Evidence must describe all three source categories")
    for name, source in value["sources"].items():
        if source.get("status") not in {"available", "unavailable"}:
            raise ValueError(f"Invalid {name} availability")
        if source["status"] == "unavailable" and not source.get("reason"):
            raise ValueError("Missing evidence needs an explicit reason")
        for field in ("task", "positions", "candidates"):
            if not isinstance(source.get(field), dict):
                raise ValueError(f"Invalid {name}.{field}")
        if source["status"] == "available" and not source.get("provenance"):
            raise ValueError("Available evidence requires provenance")
        for record in source.get("provenance", []):
            if not all(record.get(k) for k in ("id", "sha256", "tool", "version")):
                raise ValueError("Incomplete evidence provenance")
            if root is not None and record.get("artifact"):
                path = (root / record["artifact"]).resolve()
                if not path.is_relative_to(root.resolve()) or file_hash(path) != record["sha256"]:
                    raise ValueError("Evidence artifact missing, escaped root, or changed")
    return value


class FrozenEvidence:
    def __init__(self, path: Path, *, assay: str, wt_sequence: str,
                 variants: list[dict], sources: list[str]):
        if not sources or len(sources) != len(set(sources)) or set(sources) - SOURCES:
            raise ValueError("Select unique, supported evidence sources")
        self.bundle = validate_bundle(json.loads(path.read_text()), root=path.parent)
        if self.bundle["assay"] != assay or self.bundle["wt_sha256"] != digest(wt_sequence):
            raise ValueError("Evidence does not match the frozen WT/task")
        self.variants = {v["variant_id"]: v for v in variants}
        self.sources = sorted(sources)
        for source in self.bundle["sources"].values():
            for v in variants:
                key = digest(v["sequence"])
                if source["status"] == "available" and source.get("candidate_specific"):
                    if key not in source["candidates"]:
                        raise ValueError("Evidence lacks a frozen candidate sequence")

    @property
    def available(self) -> list[str]:
        return [s for s in self.sources if self.bundle["sources"][s]["status"] == "available"]

    def render(self, ids: list[str]) -> str:
        rows: dict[str, Any] = {}
        for name in self.sources:
            source = self.bundle["sources"][name]
            if source["status"] != "available":
                rows[name] = {"status": "unavailable", "reason": source["reason"]}
                continue
            positions = set()
            candidates = {}
            for vid in ids:
                variant = self.variants[vid]
                item = source["candidates"].get(digest(variant["sequence"]), {})
                candidates[vid] = item
                positions.update(str(p) for p in item.get("changed_positions", []))
            rows[name] = {
                "task": source["task"], "candidates": candidates,
                "positions": {p: source["positions"][p] for p in sorted(positions, key=int)
                              if p in source["positions"]},
                "provenance_ids": [p["id"] for p in source["provenance"]],
                "limitations": source.get("limitations", []),
            }
        return "Frozen multi-source evidence (text/numeric; observations, not fitness labels):\n" + json.dumps(
            rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def command(config_path: Path, action: str) -> int:
    from .evidence_tools import prepare_bundle

    config = json.loads(config_path.read_text())
    path = Path(config["bundle_file"])
    if not path.is_absolute():
        path = config_path.parent / path
    if action == "prepare":
        prepare_bundle(config, base=config_path.parent, output=path)
    bundle = validate_bundle(json.loads(path.read_text()), root=path.parent)
    print(json.dumps({"status": "validated", "assay": bundle["assay"],
                      "sha256": bundle["sha256"], "bundle_file": str(path)}))
    return 0
