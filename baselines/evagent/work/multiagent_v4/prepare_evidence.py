"""Freeze reviewed evidence for framework v4, without provider calls or label access.

Existing snapshots are read-only inputs. The only loaded candidate fields come
through load_task's public projection. Raw literature stays in quarantine; only
hash-bound, explicitly reviewed literal spans enter the new snapshots.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import shutil
from pathlib import Path

from evagent.integrated.evidence_tools import empty_source
from evagent.integrated.gateway import assert_public, digest
from evagent.integrated.tasks import file_hash, load_task
from evagent.integrated.research.contracts import load_snapshot, save
from evagent.integrated.research.literature_review import POLICY, screen_document
from evagent.integrated.research.msa_agent import MSAPreparationAgent


def admit(family, curation, denied_dois, denied_titles):
    spec = curation["families"][family]
    documents, raw, audit = [], [], []
    for item in spec["documents"]:
        record = copy.deepcopy(item["record"])
        record["passage_review"] = {
            "source_sha256": digest(record["text"]), "policy": POLICY,
            "reviewer": curation["reviewer"], "reviewed_at": curation["reviewed_at"],
            "spans": item["spans"],
        }
        approved, decisions = screen_document(record, denied_dois, denied_titles)
        if len(approved) != len(item["spans"]):
            raise ValueError(f"Reviewed source failed admission: {record['doi']}: {decisions}")
        documents.extend(approved)
        raw.append(record)
        audit.append({"doi": record["doi"], "decisions": decisions})
    return documents, raw, audit, spec.get("missing_reason")


def find_alignment(snapshot, snapshot_path, assay, expansion_root):
    """Resolve only audited pre-existing natural assets, never a candidate library."""
    source = snapshot["sources"]["msa"]
    if source["status"] != "available":
        return None
    reviews = [p.get("audit", p) for p in source.get("provenance", [])
               if p.get("audit", p).get("origin") == "natural_homologs"
               and p.get("audit", p).get("reviewed") is True]
    if not reviews:
        raise ValueError("Available historical MSA lacks reviewed natural origin")
    review = reviews[0]
    candidates = [(snapshot_path.parent / p["artifact"], p["sha256"])
                  for p in source["provenance"] if p.get("artifact") and
                  Path(p["artifact"]).suffix in {".sto", ".a2m", ".fasta"}]
    # V10 snapshots retained a feature hash but omitted a copied alignment asset.
    # Resolve exactly that task's v10 file and require its original recorded hash.
    if not candidates:
        expected = next((p.get("alignment_sha256") for p in source["provenance"]
                         if p.get("alignment_sha256")), None)
        candidate = expansion_root / assay / "raw/expanded_mapped.sto"
        if expected and candidate.exists():
            candidates = [(candidate, expected)]
    if len(candidates) != 1:
        raise ValueError(f"Cannot resolve exactly one pinned natural alignment for {assay}")
    path, expected = candidates[0]
    if file_hash(path) != expected:
        raise ValueError("Historical MSA alignment hash changed")
    return {"path": str(path), "sha256": expected, "format": "stockholm",
            "query_identifier": review.get("query_identifier", "WT_query"),
            "origin": "natural_homologs", "reviewed": True,
            "source": review.get("source_urls") or review.get("source") or
                      {"historical_snapshot": str(snapshot_path), "audit": review}}


def assay_context(task, family, documents):
    # Descriptive metadata and approved source claims only. The benchmark label
    # vector is neither loaded here nor used to write hypotheses/direction.
    description = task.metadata.get("ASSAY_DESCRIPTION", "RNA property in supplied assay metadata")
    hypotheses = []
    for document in documents:
        hypotheses.append(f"[{document['id']}; DOI {document['doi']}] "
                          f"{document['applicability']} Limitation: {document['limitations']}")
    if not hypotheses:
        hypotheses = ["No independently reviewed mechanism passage was admitted for this task; "
                      "do not infer a detailed catalytic mechanism from the family name alone."]
    return {"status": "complete", "data": {
        "assay_id": task.assay,
        "assay_summary": f"{description} Verified source family: {family}. "
                         "Use the supplied full WT and full candidate sequences. "
                         "Conditions not specified in the frozen assay metadata remain unknown.",
        "property_direction": "Higher assayed function is better in the RNAGym standardized direction. "
                              "Predict the supplied assay endpoint, not generic folding stability.",
        "functional_hypotheses": hypotheses,
        "sequence_features": ["Positions are one-based coordinates of the supplied full WT. "
                              "Literature residue numbers require an explicit mapping before use."]},
        "provenance": {"kind": "reviewed-task-context-not-long-term-memory", "family": family,
                       "input_context_sha256": task.input_hashes["context"],
                       "passage_ids": [d["id"] for d in documents]},
    }


def prepare(config_path, output, curation_path, expansion_root, cohort_path=None):
    config = json.loads(config_path.read_text())
    curation = json.loads(curation_path.read_text())
    cohort = json.loads(cohort_path.read_text()) if cohort_path else None
    if cohort:
        admitted = {t["assay"] for t in cohort["ncrna_tasks"] if t["audit"]["status"] == "admitted"}
        if (cohort["total_reference_tasks"] != 70 or len(cohort["ncrna_tasks"]) != 31 or
                admitted != {t["assay"] for t in config["tasks"]}):
            raise ValueError("Frozen 70/31 source coverage and predictive 24-task cohort disagree")
    if curation.get("schema") != "evagent-framework-evidence-curation-v1":
        raise ValueError("Invalid explicit curation manifest")
    if output.exists():
        raise FileExistsError("Use a new preparation directory; frozen evidence is never overwritten")
    output.mkdir(parents=True)
    save(output / "curation.json", curation)
    rows, tasks = [], []
    for spec in config["tasks"]:
        task = load_task(spec)
        assert_public(task.variants)
        original_path = Path(spec["snapshot_file"])
        old = load_snapshot(original_path, task)
        if old["source_audit"].get("family", spec["family"]) != spec["family"]:
            raise ValueError("Source admission family mismatch")
        destination = output / "tasks" / task.assay
        destination.mkdir(parents=True)
        current = copy.deepcopy(dict(old))
        current.pop("sha256", None)
        # Assets are copied, never linked outside the new frozen snapshot root.
        for asset in old.get("assets", []):
            p = destination / asset["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_path.parent / asset["path"], p)
        documents, raw, audit, missing = admit(spec["family"], curation,
                                              old.get("denied_dois", []), old.get("denied_titles", []))
        save(destination / "quarantine/documents.json", raw)
        save(destination / "quarantine/review_audit.json", audit)
        approved_dois = {d["doi"].lower() for d in documents}
        save(destination / "quarantine/previous_corpus_disposition.json", [{
            "doi": d["doi"], "title": d["title"], "source_sha256": digest(d["text"]),
            "decision": "reviewed_spans_only" if d["doi"].lower() in approved_dois else "withheld",
            "reason": ("Only the newly reviewed literal spans enter prediction; all remaining text is withheld."
                       if d["doi"].lower() in approved_dois else missing or
                       "Not selected by explicit mechanism/background review for framework v4."),
        } for d in old.get("literature_documents", [])])
        current["literature_policy"] = POLICY
        current["literature_documents"] = documents
        if documents:
            current["sources"]["literature"] = {
                "status": "available", "task": {"papers": [
                    {k: d[k] for k in ("id", "doi", "title", "location", "relevance")} for d in documents]},
                "positions": {}, "candidates": {}, "provenance": [],
                "limitations": ["Source and passage filtering cannot exclude model pretraining exposure."]}
        else:
            current["sources"]["literature"] = empty_source(missing)
        alignment = find_alignment(old, original_path, task.assay, expansion_root)
        msa_config = {"mode": "precomputed", "usage": "external_fusion"}
        if alignment:
            frozen = destination / "msa_input/original.sto"
            frozen.parent.mkdir()
            shutil.copy2(alignment["path"], frozen)
            alignment["original_path"] = alignment["path"]
            alignment["path"] = str(frozen)
            msa_config["alignment"] = alignment
        msa = MSAPreparationAgent(msa_config, destination, destination / "msa_v2").prepare(
            task.wt_sequence.replace("T", "U"), task.variants)
        current["sources"]["msa"] = msa
        current["msa_usage"] = "external_fusion"
        current["msa_values"] = None
        if msa["status"] == "available" and any(
                p.get("support") == "adequate" for p in msa["positions"].values()):
            current["msa_values"] = {
                v["variant_id"]: sum(msa["candidates"][digest(v["sequence"])]["site_log_frequency_ratios"].values())
                for v in task.variants}
        context = assay_context(task, spec["family"], documents)
        assert_public(context)
        save(destination / "assay_memory.json", context)
        current["preparation_v4"] = {
            "parent_snapshot_sha256": old["sha256"], "parent_snapshot_path": str(original_path),
            "curation_sha256": digest(curation), "source_code_sha256": file_hash(Path(__file__)),
            "msa_implementation_sha256": file_hash(Path(__file__).parents[2] /
                                                     "evagent/integrated/research/msa_agent.py"),
            "msa_mode": "v2_precomputed", "structure": "unchanged_frozen_assets_no_reprediction",
            "provider_calls": 0,
        }
        # Include all local source/receipt material in asset validation, including
        # quarantine. Only literature_documents is visible to prediction tools.
        current["assets"] = [{"path": str(p.relative_to(destination)), "sha256": file_hash(p)}
                             for p in sorted(destination.rglob("*")) if p.is_file()]
        current["sha256"] = digest(current)
        snapshot_path = destination / "snapshot.json"
        save(snapshot_path, current)
        new_spec = dict(spec, snapshot_file=str(snapshot_path),
                        assay_memory_file=str(destination / "assay_memory.json"))
        load_snapshot(snapshot_path, load_task(new_spec))
        tasks.append(new_spec)
        rows.append({"assay": task.assay, "family": spec["family"],
                     "source_status": old["source_audit"]["status"],
                     "snapshot_sha256": current["sha256"], "sample_sha256": task.input_hashes["sample"],
                     "context_sha256": file_hash(destination / "assay_memory.json"),
                     "approved_passages": len(documents), "dois": [d["doi"] for d in documents],
                     "literature_missing_reason": missing, "msa_status": msa["status"],
                     "msa_mode": "v2_precomputed", "msa_fusion_available": current["msa_values"] is not None,
                     "msa_neff": msa["task"].get("neff"),
                     "previous_msa_fusion_available": old.get("msa_values") is not None,
                     "msa_values_identical_to_previous": current["msa_values"] == old.get("msa_values"),
                     "structure_candidates": len(current["sources"]["structure"]["candidates"]),
                     "provider_calls": 0})
        save(output / "coverage.partial.json", rows)
        print(json.dumps(rows[-1]), flush=True)
    if len(tasks) != 24 or [t["assay"] for t in tasks] != [t["assay"] for t in config["tasks"]]:
        raise ValueError("Expected unchanged frozen 24-task source-admitted cohort")
    save(output / "tasks.json", tasks)
    save(output / "coverage.json", rows)
    if cohort:
        cohort = copy.deepcopy(cohort)
        prepared = {r["assay"]: r for r in rows}
        for item in cohort["ncrna_tasks"]:
            item["framework_preparation"] = prepared.get(item["assay"], {
                "prediction": "not_admitted", "evidence_preparation": "not_performed",
                "reason": item["audit"]["reason"]})
        cohort["parent_source_manifest"] = {"path": str(cohort_path), "sha256": file_hash(cohort_path)}
        save(output / "coverage_31.json", cohort)
    save(output / "manifest.json", {"schema": "evagent-framework-preparation-v4",
         "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "input_config": str(config_path), "input_config_sha256": file_hash(config_path),
         "curation_sha256": digest(curation), "tasks_sha256": file_hash(output / "tasks.json"),
         "coverage_sha256": file_hash(output / "coverage.json"), "tasks": len(rows),
         "msa_available": sum(r["msa_fusion_available"] for r in rows),
         "literature_available": sum(r["approved_passages"] > 0 for r in rows),
         "provider_calls": 0, "reported_tokens": 0})
    return tasks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--curation", type=Path, default=Path(__file__).parent / "curation/reviewed_passages.json")
    p.add_argument("--msa-expansion-root", type=Path, default=Path(
        "/data/yanjie_huang/EVAgent/experiments/msa_coverage_expansion_v10/assets/evidence"))
    p.add_argument("--cohort", type=Path, default=Path(
        "/data/yanjie_huang/EVAgent/experiments/ncrna_multiagent_v1/coverage.json"))
    args = p.parse_args()
    prepare(args.config, args.output, args.curation, args.msa_expansion_root, args.cohort)
