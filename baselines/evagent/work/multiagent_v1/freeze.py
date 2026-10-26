"""Build a fresh complete ncRNA research cohort from audited Stage B assets."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from evagent.integrated.research.contracts import ANALYSIS_CONFIG, digest, save
from evagent.integrated.research.literature import audit_document, relevance_gate
from evagent.integrated.research.prepare import FAMILY_TERMS, build_snapshot, collect_papers
from evagent.integrated.tasks import file_hash

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--collect", action="store_true")
args = parser.parse_args()
P = args.project
OUT = args.output
B = P / "experiments/cr_ncrna_evidence_b_v3/assets"
X = P / "experiments/msa_coverage_expansion_v10/assets"
registry = json.loads((B / "registry.json").read_text())
assert registry["total_reference_tasks"] == 70 and len(registry["ncrna_tasks"]) == 31
refs = {
    r["DMS_ID"]: r
    for r in csv.DictReader(
        (P.parent / "RG-LLM/data/RNAGym/fitness/reference_sheet_final.csv").open(
            encoding="utf-8-sig"
        )
    )
}
review = json.loads((P / "work/stage_b/literature_cards_v3.json").read_text())
denied_dois = [r["doi"] for r in review["exclusions"] if r.get("doi")]
denied_titles = [r["TITLE"] for r in refs.values()] + [
    r["title"] for r in review["exclusions"] if r.get("title")
]
prior = {
    r["assay"]: r
    for r in json.loads((P / "experiments/cr_ncrna_evidence_b_repro_v4/tasks.json").read_text())[
        "tasks"
    ]
}
coverage = []
specs = []
for row in registry["ncrna_tasks"]:
    assay = row["assay"]
    audit = dict(row["audit"])
    extra = X / "contexts" / (assay + ".json")
    if extra.exists():
        audit = json.loads(extra.read_text())["source_audit"]
    # A later source review excludes the viral hairpin explicitly.
    if assay == "Roberts_2023_hp_ribozyme":
        audit = {
            **audit,
            "status": "source_excluded",
            "reason": "Primary source review identifies tobacco ringspot viral satellite origin",
        }
    coverage.append({"assay": assay, "rna_type": row["rna_type"], "audit": audit})
    if audit["status"] != "admitted":
        continue
    target = OUT / "inputs" / assay
    target.mkdir(parents=True, exist_ok=True)
    if assay in prior:
        n = prior[assay]["maximum_n"]
        src = Path(prior[assay]["samples"][str(n)]["path"])
        context = json.loads((B / "contexts" / (assay + ".json")).read_text())
    else:
        context_extra = json.loads(extra.read_text())
        src = Path(context_extra["samples"]["250"]["path"])
        n = 250
        task = json.loads((X / "tasks" / (assay + ".json")).read_text())
        context = {
            "assay": assay,
            "wt_sequence": task["wt_sequence"],
            "metadata": context_extra["metadata"],
        }
    sample = target / "sample.json"
    if sample.exists() and file_hash(sample) != file_hash(src):
        raise ValueError("Frozen sample changed")
    shutil.copy2(src, sample)
    save(target / "context.json", context)
    spec = {
        "assay": assay,
        "family": audit["family"],
        "source_audit": audit,
        "task_reviewed": True,
        "sample_file": str(sample),
        "context_file": str(target / "context.json"),
    }
    old = B / "evidence" / assay / "bundle_v3.json"
    if old.exists():
        assert json.loads(old.read_text())["sha256"] == prior[assay]["evidence_sha256"], (
            "Use the actual V4 frozen bundle"
        )
        spec["legacy_bundle"] = str(old)
    msa = X / "evidence" / assay / "msa_features.json"
    if msa.exists() and assay != "Soo_2021_ribozyme":
        spec.update(msa_features=str(msa), msa_reviewed=True)
    if assay == "Soo_2021_ribozyme":
        spec["msa_review_reason"] = (
            "Family-local applicability remains exploratory; no MSA fusion in the primary campaign"
        )
    query = FAMILY_TERMS.get(audit["family"])
    if "GFP" in assay:
        query = "GFP binding RNA aptamer structural recognition"
    if "NELFE" in assay:
        query = "NELF E RNA recognition binding domain"
    if query is None:
        raise ValueError("Missing reviewed task query: " + assay)
    spec["literature_query"] = query
    spec["literature_file"] = str(OUT / "literature" / assay / "documents.json")
    specs.append(spec)
save(
    OUT / "coverage.json",
    {
        "total_reference_tasks": 70,
        "ncrna_tasks": coverage,
        "splicing_tasks_separate": registry["splicing_tasks_separate"],
        "admitted_count": len(specs),
    },
)
save(
    OUT / "prepare.json",
    {"schema": "evagent-research-prepare-v1", "output_dir": str(OUT), "tasks": specs},
)
print("source-admitted", len(specs), flush=True)
for spec in specs:
    lp = Path(spec["literature_file"])
    if args.collect and not lp.exists():
        shared = OUT / "collection_cache" / digest(spec["literature_query"])
        if (shared / "documents.json").exists():
            documents = json.loads((shared / "documents.json").read_text())
        else:
            documents, receipts = collect_papers(
                spec["literature_query"],
                denied_dois,
                denied_titles,
                shared / "quarantine",
                proxy="http://127.0.0.1:17892",
            )
            save(shared / "documents.json", documents)
        save(lp, documents)
    if not lp.exists():
        raise ValueError("Collect and review literature before freezing")
    documents = json.loads(lp.read_text())
    filtered = []
    decisions = []
    for d in documents:
        reason = relevance_gate(spec["literature_query"], d) or audit_document(
            d, denied_dois, denied_titles
        )
        decisions.append(
            {"doi": d["doi"], "decision": "exclude" if reason else "include", "reason": reason}
        )
        if not reason:
            if any(s in spec["family"] for s in ["Janzen", "synthetic_TPR", "Tome"]):
                d["relevance"] = "background"
            filtered.append(d)
    save(lp.parent / "task_relevance_review.json", decisions)
    if filtered != documents:
        if not (lp.parent / "documents.preflight.json").exists():
            shutil.copy2(lp, lp.parent / "documents.preflight.json")
        save(lp, filtered)
    documents = filtered
    snapshot = build_snapshot(spec, OUT / "snapshots" / spec["assay"] / "snapshot.json", documents)
    print(
        spec["assay"],
        "papers",
        len(documents),
        "geometry",
        snapshot["geometry"]["status"],
        "msa_fusion",
        snapshot["msa_values"] is not None,
        flush=True,
    )
model = {
    "model": "gemini-3.8-flash",
    "thinking_level": "medium",
    "temperature": 0,
    "generation_seed": 1,
    "max_output_tokens": 32768,
    "timeout_s": 1200,
    "api_key_env": "GEMINI_API_KEY",
    "proxy": "http://127.0.0.1:17892",
    "retries": 0,
}
tasks = [
    {k: s[k] for k in ("assay", "family", "sample_file", "context_file")}
    | {"snapshot_file": str(OUT / "snapshots" / s["assay"] / "snapshot.json")}
    for s in specs
]
save(
    OUT / "campaign.json",
    {
        "schema": "evagent-research-campaign-v1",
        "analysis": ANALYSIS_CONFIG,
        "output_dir": str(OUT / "campaign"),
        "coverage_file": str(OUT / "coverage.json"),
        "model_config": model,
        "budget_mode": "unlimited",
        "token_budget": None,
        "repeats": 3,
        "group_size": 25,
        "seed": 1,
        "max_repairs": 1,
        "tasks": tasks,
    },
)
print("frozen", len(tasks), "tasks", flush=True)
