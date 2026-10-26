"""CLI adapter for source-admitted MSA construction and reviewed literature."""
from __future__ import annotations

import json
from pathlib import Path

from ..bundles import file_hash
from ..evidence_tools import canonical
from ..gateway import assert_public, digest
from .literature_review import POLICY, screen_document, validate_passage
from .msa_agent import MSAPreparationAgent, validate_usage

SCHEMA = "evagent-evidence-agents-v2"


def load_prepared(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("source_audit", {}).get("status") != "admitted":
        raise ValueError("Invalid prepared agent evidence")
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Prepared evidence checksum mismatch")
    validate_usage(value["msa_usage"])
    if value.get("literature_policy") != POLICY:
        raise ValueError("Unknown literature review policy")
    for asset in value["assets"]:
        p = (path.parent / asset["path"]).resolve()
        if not p.is_relative_to(path.parent.resolve()) or file_hash(p) != asset["sha256"]:
            raise ValueError("Prepared asset escaped root, changed or missing")
    for record in value["literature_documents"]:
        validate_passage(record, value["denied_dois"], value["denied_titles"])
    raw = json.loads((path.parent / "quarantine/documents.json").read_text())
    admitted = [p for d in raw for p in screen_document(d, value["denied_dois"], value["denied_titles"])[0]]
    if admitted != value["literature_documents"]:
        raise ValueError("Admitted passages no longer match reviewed original text")
    assert_public(value)
    return value


def command(path, action):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    allowed = {"schema", "task_file", "output_dir", "source_audit", "msa_agent", "literature_file",
               "denied_dois", "denied_titles", "literature_discovery"}
    if config.get("schema") != SCHEMA or set(config) - allowed:
        raise ValueError("Invalid evidence-agent config")
    if config.get("source_audit", {}).get("status") != "admitted":
        raise ValueError("Only source-admitted tasks may prepare predictive evidence")
    root = (path.parent / config["output_dir"]).resolve()
    if action == "validate":
        result = load_prepared(root / "prepared.json")
        if result["config_sha256"] != digest(config):
            raise ValueError("Preparation config changed; use a new output directory")
    elif action == "prepare":
        task = json.loads((path.parent / config["task_file"]).read_text())
        assert_public(task)
        if set(task) - {"assay", "wt_sequence", "variants"}:
            raise ValueError("Evidence task accepts only assay, WT, and label-free variants")
        for variant in task.get("variants", []):
            if not {"variant_id", "sequence"} <= set(variant) or set(variant) - {"variant_id", "sequence", "mutation"}:
                raise ValueError("Only variant ID, sequence and optional mutation notation are allowed")
            if len(canonical(variant["sequence"])) != len(canonical(task["wt_sequence"])):
                raise ValueError("Only full substitution candidates are supported")
        root.mkdir(parents=True, exist_ok=False)
        quarantine = root / "quarantine"
        quarantine.mkdir()
        (root / "config.json").write_text(json.dumps(config, indent=2))
        (root / "task.json").write_text(json.dumps(task, indent=2))
        msa_config = config.get("msa_agent", {"mode": "precomputed", "usage": "external_fusion"})
        msa = MSAPreparationAgent(msa_config, path.parent, root / "msa").prepare(
            task["wt_sequence"], task.get("variants", []))
        raw = json.loads((path.parent / config["literature_file"]).read_text()) if config.get("literature_file") else []
        if config.get("literature_discovery"):
            from .prepare import collect_papers
            discovery = config["literature_discovery"]
            if set(discovery) - {"query", "proxy"}:
                raise ValueError("Discovery accepts only a background query and proxy")
            found, _ = collect_papers(discovery["query"], config.get("denied_dois", []),
                                     config.get("denied_titles", []), quarantine / "retrieval",
                                     discovery.get("proxy"), policy=POLICY)
            raw.extend(found)
        (quarantine / "documents.json").write_text(json.dumps(raw, indent=2))
        approved, audit = [], []
        for record in raw:
            passages, decisions = screen_document(record, config.get("denied_dois", []), config.get("denied_titles", []))
            approved.extend(passages)
            audit.append({"doi": record.get("doi"), "decisions": decisions})
        if len({p["id"] for p in approved}) != len(approved):
            raise ValueError("Duplicate reviewed passages; deduplicate source records before freezing")
        (quarantine / "review_audit.json").write_text(json.dumps(audit, indent=2))
        result = {"schema": SCHEMA, "source_audit": config["source_audit"],
                  "assay": task["assay"], "wt_sha256": digest(task["wt_sequence"]),
                  "variants_sha256": digest(task.get("variants", [])), "config_sha256": digest(config),
                  "msa_source": msa, "msa_usage": msa_config.get("usage", "external_fusion"),
                  "literature_policy": POLICY, "literature_documents": approved,
                  "denied_dois": config.get("denied_dois", []), "denied_titles": config.get("denied_titles", []),
                  "assets": [{"path": str(p.relative_to(root)), "sha256": file_hash(p)}
                             for p in sorted(root.rglob("*")) if p.is_file()]}
        assert_public(result)
        result["sha256"] = digest(result)
        (root / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
        load_prepared(root / "prepared.json")
    else:
        raise ValueError("Use prepare or validate")
    print(json.dumps({"status": "validated", "msa_status": result["msa_source"]["status"],
                      "approved_passages": len(result["literature_documents"]), "output": str(root / "prepared.json")}))
    return 0
