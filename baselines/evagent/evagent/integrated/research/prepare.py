"""Source-admitted evidence preparation; all network material is quarantined before review."""

from __future__ import annotations

import datetime
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ..bundles import validate_bundle
from ..evidence_tools import empty_source, secondary_features
from ..gateway import assert_public, digest
from ..tasks import file_hash, load_task
from .contracts import SCHEMA, load_snapshot, save
from .literature import audit_document, check_query, relevance_gate
from .structure import annotate_cif

FAMILY_TERMS = {
    "mammalian_CPEB3": "CPEB3 ribozyme folding",
    "yeast_tRNA": "Saccharomyces cerevisiae tRNA structure maturation",
    "Janzen_aminoacylation": "aminoacylating ribozyme aminoacylation mechanism",
    "rice_twister": "Oryza sativa twister ribozyme structure",
    "Azoarcus_groupI": "Azoarcus group I intron folding",
    "synthetic_TPR": "triplet polymerase ribozyme RNA template structure",
    "synthetic_classII_ligase": "class II ligase ribozyme folding",
    "human_TS_like": "human twister sister ribozyme RNA folding",
    "synthetic_consensus_glmS": "glmS ribozyme riboswitch glucosamine structure",
    "Tetrahymena_groupI": "Tetrahymena group I intron splicing folding",
}


def collect_papers(query, denied_dois, denied_titles, output, proxy=None, *, policy="legacy"):
    import requests

    session = requests.Session()
    session.trust_env = False
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    output.mkdir(parents=True, exist_ok=True)
    records, receipts = [], []

    def get(url, params=None):
        key = digest([url, params])
        p = output / (key + ".txt")
        if p.exists():
            return p.read_text()
        response = session.get(url, params=params, timeout=45)
        response.raise_for_status()
        p.write_text(response.text)
        return response.text

    if policy == "passage_review_v2":
        from .literature_review import check_background_query
        check_background_query(query)
    elif policy == "legacy":
        check_query(query)
    else:
        raise ValueError("Unknown literature collection policy")
    # Provider lookup failures remain explicit; no hidden retries or model fallback.
    providers = [
        (
            "europepmc",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            {"query": query, "format": "json", "resultType": "core", "pageSize": 12},
        ),
        (
            "semantic_scholar",
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {"query": query, "limit": 8, "fields": "title,abstract,externalIds,year,url"},
        ),
        ("crossref", "https://api.crossref.org/works", {"query.bibliographic": query, "rows": 8}),
    ]
    candidates = []
    for provider, url, params in providers:
        try:
            data = json.loads(get(url, params))
            if provider == "europepmc":
                for x in data.get("resultList", {}).get("result", []):
                    candidates.append(
                        {
                            "doi": x.get("doi", ""),
                            "title": x.get("title", ""),
                            "text": x.get("abstractText", ""),
                            "pmcid": x.get("pmcid"),
                            "date": x.get("firstPublicationDate"),
                            "provider": provider,
                        }
                    )
            elif provider == "semantic_scholar":
                for x in data.get("data", []):
                    candidates.append(
                        {
                            "doi": x.get("externalIds", {}).get("DOI", ""),
                            "title": x.get("title", ""),
                            "text": x.get("abstract") or "",
                            "date": x.get("year"),
                            "provider": provider,
                        }
                    )
            else:
                for x in data.get("message", {}).get("items", []):
                    candidates.append(
                        {
                            "doi": x.get("DOI", ""),
                            "title": (x.get("title") or [""])[0],
                            "text": x.get("abstract", ""),
                            "date": x.get("created", {}).get("date-time"),
                            "provider": provider,
                        }
                    )
            receipts.append(
                {"provider": provider, "status": "ok", "request_sha256": digest([url, params])}
            )
        except (requests.RequestException, ValueError) as error:
            receipts.append(
                {"provider": provider, "status": "failed", "error_type": type(error).__name__}
            )
    seen = set()
    decisions = []
    query_terms = {x.lower() for x in re.findall(r"[A-Za-z0-9]+", query)} - {
        "rna",
        "structure",
        "folding",
        "mechanism",
        "ribozyme",
        "the",
        "and",
    }
    for c in candidates:
        doi = c["doi"].lower()
        if not doi or doi in seen:
            continue
        seen.add(doi)
        c.update(id="paper-" + digest(doi)[:16], location="abstract", relevance="family")
        text = re.sub(r"<[^>]+>", " ", c["text"])
        if not any(t in (c["title"] + " " + text).lower() for t in query_terms):
            decisions.append(
                {"doi": doi, "decision": "exclude", "reason": "no_family_or_construct_match"}
            )
            continue
        c["text"] = text
        if policy == "legacy" and relevance_gate(query, c):
            decisions.append(
                {"doi": doi, "decision": "exclude", "reason": relevance_gate(query, c)}
            )
            continue
        if policy == "passage_review_v2":
            from .literature_review import source_block
            reason = source_block(c, denied_dois, denied_titles)
        else:
            reason = audit_document(c, denied_dois, denied_titles)
        if reason:
            decisions.append({"doi": doi, "decision": "exclude", "reason": reason})
            continue
        if c.get("pmcid"):
            try:
                xml = get(
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                    + c["pmcid"]
                    + "/fullTextXML"
                )
                tree = ET.fromstring(xml)
                body = tree.find("body")
                if body is not None:
                    c["text"] = "\n\n".join(" ".join(p.itertext()) for p in body.iter("p"))
                    c["location"] = "fullTextXML/body/paragraphs"
            except (requests.RequestException, ET.ParseError):
                c["fulltext_status"] = "unavailable_abstract_only"
        reason = (source_block(c, denied_dois, denied_titles) if policy == "passage_review_v2"
                  else audit_document(c, denied_dois, denied_titles))
        if reason:
            decisions.append({"doi": doi, "decision": "exclude", "reason": reason})
            continue
        c["retrieved_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if policy == "passage_review_v2":
            c["review_status"] = "pending_passage_review"
            records.append(c)
            decisions.append({"doi": doi, "decision": "quarantine_pending_review",
                              "location": c["location"]})
            continue
        c["review"] = (
            "Conservative source and contamination filters; passage-level applicability checked by PaperQA and specialist"
        )
        records.append(c)
        decisions.append({"doi": doi, "decision": "include", "location": c["location"]})
    save(output / "decisions.json", decisions)
    save(output / "providers.json", receipts)
    return records, receipts


def build_snapshot(spec, output, documents):
    task = load_task(spec)
    assert_public(task.variants)
    audit = spec["source_audit"]
    if audit.get("status") != "admitted":
        raise ValueError("Source not admitted")
    if output.exists():
        existing = load_snapshot(output, task)
        if spec.get("prepared_agents"):
            from .evidence_agents import load_prepared
            prepared = load_prepared(Path(spec["prepared_agents"]))
            if existing.get("prepared_agents_sha256") != prepared["sha256"]:
                raise ValueError("Prepared evidence changed; use a new snapshot output")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {
        s: empty_source("No frozen evidence available") for s in ("msa", "structure", "literature")
    }
    assets = []
    geometry = {"status": "unavailable", "reason": "No existing mapped WT tertiary structure"}
    if spec.get("legacy_bundle"):
        bp = Path(spec["legacy_bundle"])
        old = validate_bundle(json.loads(bp.read_text()), root=bp.parent)
        if old["wt_sha256"] != digest(task.wt_sequence):
            raise ValueError("Legacy WT mismatch")
        for source in ("msa", "structure"):
            sources[source] = old["sources"][source]
            for p in sources[source].get("provenance", []):
                if p.get("artifact"):
                    src = bp.parent / p["artifact"]
                    rel = "legacy/" + p["artifact"]
                    dst = output.parent / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    p["artifact"] = rel
                    assets.append({"path": rel, "sha256": file_hash(dst)})
                    if src.suffix == ".cif":
                        geometry = annotate_cif(
                            dst,
                            task.wt_sequence,
                            output.parent / "mapped_WT.pdb",
                            predicted="af3" in src.name.lower(),
                        )
                        assets.append(
                            {
                                "path": "mapped_WT.pdb",
                                "sha256": file_hash(output.parent / "mapped_WT.pdb"),
                            }
                        )
    if spec.get("msa_features") and spec.get("msa_reviewed") is True:
        sources["msa"] = json.loads(Path(spec["msa_features"]).read_text())
        sources["msa"].setdefault("provenance", []).append(
            {
                "tool": "frozen-Rfam-MSA",
                "version": "v10",
                "sha256": file_hash(Path(spec["msa_features"])),
            }
        )
    # V10 used the generic key score for Infernal bit scores. Rename only that
    # known metadata location; never relax the label guard or scrub arbitrary labels.
    for provenance in sources["msa"].get("provenance", []):
        for match in provenance.get("audit", {}).get("family_match", []):
            if "score" in match:
                match["cm_bit_score"] = match.pop("score")
    # Refresh complete secondary features consistently for every task.
    secondary = secondary_features(task.wt_sequence.replace("T", "U"), task.variants)
    sources["structure"]["task"].update(secondary["task"])
    sources["structure"]["candidates"] = secondary["candidates"]
    for k, v in secondary["positions"].items():
        sources["structure"]["positions"].setdefault(k, {}).update(v)
    sources["structure"]["status"] = "available"
    sources["structure"].setdefault("limitations", []).extend(secondary["limitations"])
    sources["structure"].setdefault("provenance", []).append(
        {"tool": "ViennaRNA", "version": "2.7.2", "sha256": digest(secondary)}
    )
    prepared = None
    if spec.get("prepared_agents"):
        from .evidence_agents import load_prepared
        prepared = load_prepared(Path(spec["prepared_agents"]))
        if (prepared["assay"] != task.assay or prepared["wt_sha256"] != digest(task.wt_sequence)
                or prepared["variants_sha256"] != digest(task.variants)):
            raise ValueError("Prepared agents/task mismatch")
        sources["msa"] = prepared["msa_source"]
        documents = prepared["literature_documents"]
    msa_values = None
    msa = sources["msa"]
    if msa["status"] == "available":
        if any(p.get("support") == "adequate" for p in msa.get("positions", {}).values()):
            msa_values = {
                v["variant_id"]: sum(
                    msa["candidates"][digest(v["sequence"])]["site_log_frequency_ratios"].values()
                )
                for v in task.variants
            }
    if prepared and prepared["msa_usage"] == "prompt_only":
        msa_values = None
    if documents:
        sources["literature"] = {
            "status": "available",
            "task": {
                "papers": [
                    {k: d[k] for k in ("id", "doi", "title", "location", "relevance")}
                    for d in documents
                ]
            },
            "positions": {},
            "candidates": {},
            "provenance": [],
            "limitations": [
                "External-evidence filtering cannot guarantee absence of model pretraining exposure."
            ],
        }
    snapshot = {
        "schema": SCHEMA,
        "assay": task.assay,
        "source_audit": audit,
        "task_reviewed": spec.get("task_reviewed") is True,
        "wt_sha256": digest(task.wt_sequence),
        "variants_sha256": digest(task.variants),
        "sources": sources,
        "geometry": geometry,
        "literature_documents": documents,
        "msa_values": msa_values,
        "assets": list({a["path"]: a for a in assets}.values()),
        "representation": "text_and_numeric",
    }
    if prepared:
        snapshot.update(prepared_agents_sha256=prepared["sha256"],
                        msa_usage=prepared["msa_usage"], literature_policy=prepared["literature_policy"],
                        denied_dois=prepared["denied_dois"], denied_titles=prepared["denied_titles"])
    assert_public(snapshot)
    snapshot["sha256"] = digest(snapshot)
    save(output, snapshot)
    return load_snapshot(output, task)


def command(path, action):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != "evagent-research-prepare-v1":
        raise ValueError("Invalid preparation config")
    root = Path(config["output_dir"])
    results = []
    for spec in config["tasks"]:
        p = Path(spec.get("snapshot_file", root / "snapshots" / spec["assay"] / "snapshot.json"))
        if action == "prepare":
            documents = (
                json.loads(Path(spec["literature_file"]).read_text())
                if spec.get("literature_file")
                else []
            )
            snapshot = build_snapshot(spec, p, documents)
        else:
            snapshot = load_snapshot(p, load_task(spec))
        results.append(
            {
                "assay": spec["assay"],
                "snapshot_sha256": snapshot["sha256"],
                "source_status": {k: v["status"] for k, v in snapshot["sources"].items()},
            }
        )
    save(root / "evidence_validation.json", results)
    print(json.dumps({"status": "validated", "tasks": len(results)}))
    return 0
