"""Ledger-owned autonomous query refinement followed by immutable corpus freeze."""

import csv
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import dotenv_values

from evagent.integrated.gateway import RequestGateway, digest
from evagent.integrated.research.contracts import load_snapshot, save
from evagent.integrated.research.discovery import plan_queries
from evagent.integrated.research.literature import audit_document, relevance_gate
from evagent.integrated.research.prepare import collect_papers
from evagent.integrated.tasks import file_hash

OUT = Path(sys.argv[1])
if (OUT / "campaign_final.json").exists():
    raise SystemExit(
        "Preparation is already frozen; validate the release without reopening its ledger."
    )
P = OUT.parents[1]
config = json.loads((OUT / "campaign.json").read_text())
prepare = json.loads((OUT / "prepare.json").read_text())
credentials = dotenv_values("/home/yanjie/.config/evagent/credentials.env")
if credentials.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = credentials["GEMINI_API_KEY"]
review = json.loads((P / "work/stage_b/literature_cards_v3.json").read_text())
refs = list(
    csv.DictReader(
        (P.parent / "RG-LLM/data/RNAGym/fitness/reference_sheet_final.csv").open(
            encoding="utf-8-sig"
        )
    )
)
denied_dois = [r["doi"] for r in review["exclusions"] if r.get("doi")] + [
    "10.1021/acs.biochem.3c00506",
    "10.1073/pnas.2003425117",
    "10.1261/rna.7880404",
]
# Additional visible-title review: flanking/exon sequence effect reports and direct light-controlled activity manipulation.
denied_titles = [r["TITLE"] for r in refs] + [
    r["title"] for r in review["exclusions"] if r.get("title")
]
identity = digest(
    {
        "model": config["model_config"],
        "tasks": [
            {k: s[k] for k in ["assay", "family", "literature_query", "context_file"]}
            for s in prepare["tasks"]
        ],
        "policy": "research-discovery-v1",
    }
)
gateway = RequestGateway(
    model_config=config["model_config"],
    run_dir=OUT / "preparation/provider",
    fingerprint=identity,
    token_budget=None,
    budget_mode="unlimited",
)
try:
    for spec in prepare["tasks"]:
        queries = plan_queries(spec, gateway, OUT / "preparation")
        documents = json.loads(Path(spec["literature_file"]).read_text())
        for query in queries[1:]:
            shared = OUT / "collection_cache" / digest(query)
            p = shared / "documents.json"
            if not p.exists():
                found, _ = collect_papers(
                    query,
                    denied_dois,
                    denied_titles,
                    shared / "quarantine",
                    proxy="http://127.0.0.1:17892",
                )
                save(p, found)
            documents.extend(json.loads(p.read_text()))
        accepted = {}
        decisions = []
        for d in documents:
            reason = relevance_gate(spec["literature_query"], d) or audit_document(
                d, denied_dois, denied_titles
            )
            if not reason:
                # Family/background is an applicability bound, never an exact WT claim.
                d["relevance"] = (
                    "background"
                    if any(x in spec["family"] for x in ["Janzen", "synthetic_TPR", "Tome"])
                    else "family"
                )
                old = accepted.get(d["doi"])
                if old is None or (old["location"] == "abstract" and d["location"] != "abstract"):
                    accepted[d["doi"]] = d
            decisions.append(
                {"doi": d["doi"], "decision": "exclude" if reason else "include", "reason": reason}
            )
        papers = sorted(accepted.values(), key=lambda d: d["doi"])
        src = OUT / "snapshots" / spec["assay"]
        dst = OUT / "snapshots_final" / spec["assay"]
        if dst.exists():
            snapshot = load_snapshot(dst / "snapshot.json")
            if digest(snapshot["literature_documents"]) != digest(papers):
                raise ValueError("Refusing to replace final frozen literature")
            continue
        shutil.copytree(src, dst)
        snapshot = json.loads((dst / "snapshot.json").read_text())
        snapshot.pop("sha256")
        snapshot["literature_documents"] = papers
        snapshot["sources"]["literature"] = {
            "status": "available" if papers else "unavailable",
            "reason": None if papers else "No task-relevant paper passed the frozen review policy",
            "task": {
                "papers": [
                    {k: d[k] for k in ("id", "doi", "title", "location", "relevance")}
                    for d in papers
                ]
            },
            "positions": {},
            "candidates": {},
            "provenance": [
                {
                    "tool": "EuropePMC/SemanticScholar/Crossref+bounded-query-planner",
                    "version": "v1",
                    "sha256": digest(papers),
                }
            ],
            "limitations": [
                "Abstract-only documents support only their visible abstract. No guarantee about model pretraining exposure."
            ],
        }
        save(dst / "literature_review.json", decisions)
        snapshot["assets"].append(
            {"path": "literature_review.json", "sha256": file_hash(dst / "literature_review.json")}
        )
        snapshot["denied_dois"] = denied_dois
        snapshot["denied_titles"] = denied_titles
        snapshot["literature_queries"] = queries
        snapshot["literature_policy"] = (
            "entity-title-match; benchmark/related DOI and title exclusion; conservative full-visible-text contamination filter"
        )
        snapshot["sha256"] = digest(snapshot)
        save(dst / "snapshot.json", snapshot)
        load_snapshot(dst / "snapshot.json")
        print(
            spec["assay"],
            "accepted_papers",
            len(papers),
            "preparation_tokens",
            gateway.summary()["reported_tokens"],
            flush=True,
        )
finally:
    save(OUT / "preparation/usage.json", gateway.summary())
    gateway.close()
ledger = OUT / "preparation/provider/request_ledger.sqlite"
config["prior_ledgers"] = [{"path": str(ledger), "sha256": file_hash(ledger)}]
for r in config["tasks"]:
    r["snapshot_file"] = str(OUT / "snapshots_final" / r["assay"] / "snapshot.json")
final = OUT / "campaign_final.json"
if final.exists() and json.loads(final.read_text()) != config:
    raise ValueError("Final campaign config already frozen")
save(final, config)
print("final campaign frozen", flush=True)
