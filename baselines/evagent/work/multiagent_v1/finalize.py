"""Create a reviewed release from provisional preparation snapshots, without predictions."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import shutil
import sys
from pathlib import Path

from evagent.integrated.research.campaign import prepare
from evagent.integrated.research.contracts import ANALYSIS_CONFIG, digest, load_snapshot, save
from evagent.integrated.research.literature import audit_document
from evagent.integrated.tasks import file_hash, load_task

root = Path(sys.argv[1]).resolve()
config = json.loads((root / "campaign_final.json").read_text())
initial = json.loads((root / "prepare.json").read_text())
inputs = {r["assay"]: r for r in initial["tasks"]}
environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "embedding": "PaperQA SparseEmbeddingModel, local token-frequency 256-bin representation",
    "retrieval": "PaperQA local candidate selection and gateway relevance summaries; no live lookup during prediction",
}
environment_path = root / "environment.json"
if environment_path.exists():
    if json.loads(environment_path.read_text()) != environment:
        raise ValueError("Refusing to replace a frozen environment manifest")
else:
    save(environment_path, environment)
release_specs = []
coverage = []
for task in config["tasks"]:
    old = Path(task["snapshot_file"])
    snapshot = json.loads(old.read_text())
    if snapshot["sha256"] != digest({k: v for k, v in snapshot.items() if k != "sha256"}):
        raise ValueError("Provisional snapshot changed")
    previous = snapshot.pop("sha256")
    reviewed = []
    decisions = []
    titles = set()
    for document in snapshot["literature_documents"]:
        title = re.sub(r"\W", "", re.sub(r"<[^>]+>", "", document["title"]).lower())
        reason = audit_document(
            document, snapshot.get("denied_dois", []), snapshot.get("denied_titles", [])
        )
        if title in titles:
            reason = "duplicate_normalized_title"
        if not reason:
            reviewed.append(document)
            titles.add(title)
        decisions.append(
            {
                "doi": document["doi"],
                "decision": "exclude" if reason else "include",
                "reason": reason,
            }
        )
    snapshot["literature_documents"] = reviewed
    source = snapshot["sources"]["literature"]
    source["status"] = "available" if reviewed else "unavailable"
    source["reason"] = (
        None if reviewed else "No task-relevant document passed the final contamination review"
    )
    source["task"]["papers"] = [
        {k: d[k] for k in ("id", "doi", "title", "location", "relevance")} for d in reviewed
    ]
    source["provenance"].append(
        {"tool": "release-visible-text-review", "version": "v2", "sha256": digest(reviewed)}
    )
    snapshot["final_literature_review"] = decisions
    dst = root / "snapshots_release" / task["assay"]
    snapshot["parent_snapshot_sha256"] = previous
    snapshot["release_review"] = {
        "status": "reviewed",
        "method": "entity/title, visible-text contamination and provenance checks; exact-construct applicability not assumed",
        "metadata_corrections": [
            "Barnaba torsion order includes the seventh glycosidic chi angle",
            "Human TS-like structure-family analogy classified as background",
        ],
    }
    geometry = snapshot.get("geometry", {})
    geometry["limitations"] = [
        s.replace(
            "alpha,beta,gamma,delta,epsilon,zeta;", "alpha,beta,gamma,delta,epsilon,zeta,chi;"
        )
        for s in geometry.get("limitations", [])
    ]
    if geometry.get("status") == "available":
        geometry["torsion_order"] = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
    if task["family"] == "human_TS_like":
        for doc in snapshot["literature_documents"]:
            doc["relevance"] = "background"
        for doc in snapshot["sources"]["literature"]["task"].get("papers", []):
            doc["relevance"] = "background"
    # Explicitly freeze the environment description with the evidence release.
    snapshot["tool_environment_sha256"] = file_hash(root / "environment.json")
    snapshot["sha256"] = digest(snapshot)
    if dst.exists():
        if load_snapshot(dst / "snapshot.json")["sha256"] != snapshot["sha256"]:
            raise ValueError("Release already frozen differently")
    else:
        shutil.copytree(old.parent, dst)
        save(dst / "snapshot.json", snapshot)
    task["snapshot_file"] = str(dst / "snapshot.json")
    load_snapshot(dst / "snapshot.json", load_task(task))
    release_specs.append({**inputs[task["assay"]], "snapshot_file": str(dst / "snapshot.json")})
    coverage.append(
        {
            "assay": task["assay"],
            "family": task["family"],
            "snapshot_sha256": snapshot["sha256"],
            "maximum_n": len(load_task(task).variants),
            "geometry": geometry.get("status"),
            "msa_fusion": snapshot["msa_values"] is not None,
            "literature_documents": len(snapshot["literature_documents"]),
            "fulltext_documents": sum(
                d["location"] != "abstract" for d in snapshot["literature_documents"]
            ),
        }
    )
config["analysis"] = ANALYSIS_CONFIG
config["environment_file"] = str(root / "environment.json")
config_path = root / "campaign_release.json"
if config_path.exists() and json.loads(config_path.read_text()) != config:
    raise ValueError("Campaign release already frozen differently")
save(config_path, config)
save(
    root / "prepare_release.json",
    {"schema": "evagent-research-prepare-v1", "output_dir": str(root), "tasks": release_specs},
)
save(root / "evidence_coverage.json", coverage)
prepare(config_path)
print(
    json.dumps(
        {
            "status": "release_validated",
            "tasks": len(coverage),
            "literature_tasks": sum(bool(r["literature_documents"]) for r in coverage),
            "credible_msa_tasks": sum(r["msa_fusion"] for r in coverage),
            "geometry_tasks": sum(r["geometry"] == "available" for r in coverage),
        },
        indent=2,
    )
)
