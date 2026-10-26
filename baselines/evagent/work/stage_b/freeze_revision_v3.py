"""Freeze corrected evidence and prompt-v3 matrix without model calls.

Structure bytes and sample membership must remain identical to v1. v1/v2
ledgers are referenced separately so their complete lineage is debited once.
"""

import copy
import json
import re
from pathlib import Path

import httpx

from evagent.integrated.bundles import file_hash, validate_bundle
from evagent.integrated.evidence_tools import prepare_bundle
from evagent.integrated.gateway import digest

BASE = Path("/data/yanjie_huang/EVAgent/experiments")
OLD = BASE / "cr_ncrna_evidence_b_v1"
PREVIOUS = BASE / "cr_ncrna_evidence_b_v2"
ROOT = BASE / "cr_ncrna_evidence_b_v3"
ASSETS = ROOT / "assets"


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    if p.exists() and p.read_text() != content:
        raise ValueError(f"Refusing to replace frozen revision: {p}")
    p.write_text(content)


def reviewed_record(key, card, library):
    path = ASSETS / "public/literature-review" / f"{key}.json"
    response = json.loads(path.read_text())
    matches = [
        r
        for r in response["response"]["resultList"]["result"]
        if r.get("doi", "").lower() == card["doi"].lower()
    ]
    if key == "rice_twister_qm" and not matches:
        # Publisher full text was reviewed; Crossref provides an independently
        # cached exact-DOI bibliographic receipt for this non-PubMed article.
        path = path.with_name("rice_twister_qm_crossref.json")
        if not path.exists():
            with httpx.Client(timeout=60, trust_env=False) as client:
                r = client.get("https://api.crossref.org/works/" + card["doi"])
                r.raise_for_status()
                save(path, r.json())
        meta = json.loads(path.read_text())["message"]
        assert meta["DOI"].lower() == card["doi"].lower()
        row = {"doi": meta["DOI"], "title": meta["title"][0], "firstPublicationDate": "2017-02-17"}
    else:
        if len(matches) != 1:
            raise ValueError(f"Exact DOI metadata missing or ambiguous: {key}")
        row = matches[0]
    title = re.sub("<[^>]*>", "", row["title"])
    files = {path.name: file_hash(path)}
    for suffix in (".xml", ".bioc.xml"):
        full = ASSETS / "public/literature-review" / f"{key}{suffix}"
        if full.exists():
            files[full.name] = file_hash(full)
    return {
        "id": key,
        "date": row["firstPublicationDate"],
        "doi": card["doi"],
        "title": title,
        "url": "https://doi.org/" + card["doi"],
        "excerpt": card["excerpt"],
        "summary": card["summary"],
        "applicability": card["applicability"],
        "review_scope": card["scope"],
        "bibliography_and_review_artifacts": files,
        "background_only": True,
        "target_experiment": False,
        "direct_variant_effects": False,
        "benchmark_derived": False,
    }


def main():
    library = json.loads(Path(__file__).with_name("literature_cards_v3.json").read_text())
    save(ASSETS / "public/literature-review/review_decisions.json", library)
    records = {k: reviewed_record(k, v, library) for k, v in library["records"].items()}
    registry = json.loads((ASSETS / "registry.json").read_text())
    search = json.loads((ASSETS / "public/rfam-expanded/search_receipt.json").read_text())
    oldcampaign = json.loads((PREVIOUS / "campaign.json").read_text())
    coverage = []
    available = {}
    bundle_hashes = {}
    msa_stats = {}
    for row in registry["ncrna_tasks"]:
        a = row["assay"]
        family = row["audit"]["family"]
        if row["audit"]["status"] != "admitted":
            coverage.append({**row, "available_evidence": []})
            continue
        folder = ASSETS / "evidence" / a
        key = library["assays"].get(a, library["families"].get(family))
        lit = {
            "assay": a,
            "reviewed": True,
            "records": [records[key]] if key else [],
            "review": {
                "date": library["review_date"],
                "scope": library["policy"],
                "unavailable_reason": library["unavailable"].get(family),
                "base_model_exposure_unknown": True,
            },
        }
        save(folder / "raw/literature_v3.json", lit)
        config = json.loads((folder / "prepare.json").read_text())
        config.update(
            task_file=str(ASSETS / "tasks" / f"{a}.json"),
            bundle_file=str(folder / "bundle_v3.json"),
            literature_file="raw/literature_v3.json",
        )
        config["denied_dois"] = sorted({r["doi"] for r in library["exclusions"] if "doi" in r})
        config["denied_titles"] = config.get("denied_titles", []) + [
            r["title"] for r in library["exclusions"] if "title" in r
        ]
        expanded = folder / "expanded_msa_audit.json"
        if expanded.exists():
            audit = json.loads(expanded.read_text())
            # Explicit scientific names distinguish alignment statistics from
            # forbidden benchmark score fields without weakening label guards.
            audit["family_match"] = [
                {("cm_bit_score" if k == "score" else k): v for k, v in h.items()}
                for h in audit.get("family_match", [])
            ]
            audit["source_audit_sha256"] = file_hash(expanded)
            config.update(msa_file="raw/expanded_mapped.sto", msa_audit=audit)
        save(folder / "prepare_v3.json", config)
        staged = folder / "bundle_v3.prepared.json"
        prepare_bundle(config, base=folder, output=staged)
        bundle = json.loads(staged.read_text())
        if not key:
            bundle["sources"]["literature"]["reason"] = library["unavailable"][family]
        if bundle["sources"]["msa"]["status"] == "unavailable":
            bundle["sources"]["msa"]["reason"] = search["tasks"][a]["reason"]
        bundle["sha256"] = digest({k: v for k, v in bundle.items() if k != "sha256"})
        validate_bundle(bundle, root=folder)
        original = json.loads((OLD / "assets/evidence" / a / "bundle.json").read_text())
        if bundle["sources"]["structure"] != original["sources"]["structure"]:
            raise ValueError("Structure evidence changed without authorization")
        save(folder / "bundle_v3.json", bundle)
        available[a] = [s for s, v in bundle["sources"].items() if v["status"] == "available"]
        bundle_hashes[a] = bundle["sha256"]
        msa = bundle["sources"]["msa"]
        if msa["status"] == "available":
            msa_stats[a] = msa["task"]
        coverage.append({**row, "available_evidence": available[a], "literature_record": key})
    cells = []
    calibration = None
    for original in oldcampaign["cells"]:
        cell = copy.deepcopy(original)
        a = cell["assay"]
        n = cell["n"]
        condition = cell["condition"]
        if cell["source_status"] != "admitted" or cell["status"] == "insufficient_samples":
            cells.append(cell)
            continue
        cell.pop("config", None)
        cell.pop("reason", None)
        if condition not in ("cr", "all") and condition not in available[a]:
            cell.update(
                status="evidence_unavailable",
                reason=f"No admissible frozen {condition} evidence; see coverage/review receipts",
            )
        else:
            # Start from the same model, repair, seed and score protocol config.
            template = PREVIOUS / "configs" / a / f"n{n}_cr.json"
            cfg = json.loads(template.read_text())
            cfg.update(
                prompt_version="v3",
                sample_file=str(ASSETS / "samples" / a / f"n{n}.json"),
                context_file=str(ASSETS / "contexts" / f"{a}.json"),
                output_dir=str(ROOT / "results" / a / f"n{n}" / condition),
            )
            if condition != "cr":
                cfg.update(
                    evidence_mode="bundle",
                    evidence_bundle_file=str(ASSETS / "evidence" / a / "bundle_v3.json"),
                    evidence_sources=["msa", "structure", "literature"]
                    if condition == "all"
                    else [condition],
                )
            path = ROOT / "configs" / a / f"n{n}_{condition}.json"
            save(path, cfg)
            cell.update(status="eligible", config=str(path))
            if calibration is None and len(available[a]) == 3:
                calibration = a
        cells.append(cell)
    # Explicitly preserve every prior known-cost attempt, never old predictions.
    ledgers = [d / "campaign/provider/request_ledger.sqlite" for d in (OLD, PREVIOUS)]
    prior = [{"path": str(p), "sha256": file_hash(p)} for p in ledgers]
    save(
        ROOT / "campaign.json",
        {
            **oldcampaign,
            "cells": cells,
            "output_dir": str(ROOT / "campaign"),
            "calibration_assay": calibration,
            "prior_ledgers": prior,
        },
    )
    oldcoverage = json.loads((OLD / "coverage.json").read_text())
    save(ROOT / "coverage.json", {**oldcoverage, "tasks": coverage})
    for subdir in ("samples", "contexts", "secondary", "af3_inputs", "af3_outputs"):
        for p in (OLD / "assets" / subdir).rglob("*"):
            if p.is_file() and file_hash(p) != file_hash(ASSETS / p.relative_to(OLD / "assets")):
                raise ValueError("Frozen original sample/context/structure bytes changed")
    save(
        ROOT / "revision.json",
        {
            "prior_campaign": str(PREVIOUS),
            "prompt_version": "v3",
            "score_protocol": "integrated-cr-v2",
            "changes": [
                "Permit frozen evidence in shared CR prompt",
                "Full Rfam scan and reviewed natural alignments",
                "Task/family/target-specific literature cards",
            ],
            "unchanged": [
                "model",
                "thinking",
                "temperature",
                "max_tokens",
                "samples",
                "context metadata",
                "structure evidence",
                "repair limit",
                "memory mode",
                "seed",
            ],
            "prior_ledgers": prior,
            "prediction_metrics_used_for_decision": False,
            "total_budget": 50_000_000,
            "bundle_hashes": bundle_hashes,
            "msa_stats": msa_stats,
            "literature_available_tasks": sum("literature" in s for s in available.values()),
            "msa_available_tasks": sum("msa" in s for s in available.values()),
        },
    )
    save(
        ROOT / "assets_manifest.json",
        {
            "sha256": {
                str(p.relative_to(ASSETS)): file_hash(p)
                for p in sorted(ASSETS.rglob("*"))
                if p.is_file()
            }
        },
    )
    print(
        json.dumps(
            {
                "eligible": sum(c["status"] == "eligible" for c in cells),
                "calibration": calibration,
                "msa_tasks": len(msa_stats),
                "literature_tasks": sum("literature" in s for s in available.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
