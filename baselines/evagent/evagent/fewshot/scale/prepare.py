"""Offline qualification, metadata-only selection, and label-separated freezing."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from evagent.fewshot.data import load_landscape, sha256_file
from evagent.fewshot.protocol import digest
from evagent.fewshot.runner import load, save


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def permute(records, assay, namespace):
    return sorted(records, key=lambda x: digest([1, assay, namespace, x["sequence"]]))


def public(row, variant_id):
    return {"variant_id": variant_id, **{k: row[k] for k in
            ("sequence", "mutation", "mutation_order")}}


def freeze(landscape, folder):
    """Initial supports are members of the 250 acquisition pool, never test rows."""
    meta = landscape.metadata
    ordered = permute(landscape.records, meta["assay_id"], "test-pool")
    test, pool = ordered[:250], ordered[250:500]
    assert len(test) == 250 and len(pool) >= 32
    # The entire acquisition order is fixed independently of fitness.
    pool = permute(pool, meta["assay_id"], "random-acquisition")
    query = [public(r, f"M{i:03}") for i, r in enumerate(test, 1)]
    candidates = [public(r, f"M{i:03}") for i, r in enumerate(pool, 251)]
    save(folder / "public.json", {"metadata": meta, "test": query, "pool": candidates,
                                  "random_order": [r["variant_id"] for r in candidates],
                                  "multi_round_eligible": len(pool) == 250})
    # Oracle contains acquisition labels ONLY. Test labels have a separate file.
    save(folder / "oracle.json", {r["variant_id"]: v["fitness"] for r, v in zip(candidates, pool)})
    save(folder / "evaluation.json", {r["variant_id"]: v["fitness"] for r, v in zip(query, test)})
    (folder / "oracle.json").chmod(0o600)
    (folder / "evaluation.json").chmod(0o600)
    extrap = []
    for name, low_orders, minimum, maximum in (("1_to_2", {1}, 2, 2),
                                              ("12_to_3plus", {1, 2}, 3, 10**9)):
        low = [r for r in landscape.records if r["mutation_order"] in low_orders]
        high = [r for r in landscape.records if r["mutation_order"] is not None
                and minimum <= r["mutation_order"] <= maximum]
        status = {"split": name, "support_pool": len(low), "test_pool": len(high),
                  "eligible": len(low) >= 32 and len(high) >= 250}
        extrap.append(status)
        if not status["eligible"]:
            continue
        support = permute(low, meta["assay_id"], name + "-support")[:32]
        tests = permute(high, meta["assay_id"], name + "-test")[:250]
        save(folder / name / "public.json", {"metadata": meta,
             "test": [public(r, f"M{i:03}") for i, r in enumerate(tests, 1)],
             "support": [dict(public(r, f"S{i:03}"), fitness=r["fitness"])
                         for i, r in enumerate(support, 1)]})
        save(folder / name / "evaluation.json", {f"M{i:03}": r["fitness"]
                                                 for i, r in enumerate(tests, 1)})
        (folder / name / "evaluation.json").chmod(0o600)
    return {"multi_round_eligible": len(pool) == 250, "extrapolation": extrap}


def descriptors(project):
    """Enumerate all 217/70, including explicit exclusions; public externals only."""
    previous = load(project / "experiments/fewshot_breadth_v2/COHORT.json")["assays"]
    dev = {x["assay_id"] for x in previous} | {"SPG1_STRSG_Wu_2016", "Peri_2022_ribozyme"}
    protein_ref = project / "work/proteingym_v1/source/DMS_substitutions.csv"
    proteins = rows(protein_ref)
    assert len(proteins) == 217
    old_protein_backgrounds = {x["UniProt_ID"] for x in proteins if x["DMS_id"] in dev}
    for x in proteins:
        name = x["DMS_id"]
        paths = [project / "experiments" / b / "inputs" / name / "source.csv" for b in
                 ("proteingym_cr_msa_v1", "proteingym_cr_msa_remainder_v1", "proteingym_safe_expansion_v1")]
        paths = [p for p in paths if p.exists()]
        meta = {"assay_id": name, "domain": "protein", "wt_sequence": x["target_seq"],
                "assay_description": f"{x['molecule_name']}; {x['selection_assay']}; method: {x['selection_type']}.",
                "endpoint": x["selection_assay"] or x["coarse_selection_type"],
                "readout": f"ProteinGym DMS_score derived from {x['raw_DMS_phenotype_name']}. Benchmark direction multiplier {x['raw_DMS_directionality']}; higher DMS_score is better. Unchanged benchmark units; no assumed WT value.",
                "higher_is_better": True}
        yield {"metadata": meta, "source": str(paths[0]) if paths else None,
               "columns": {"sequence": "mutated_sequence", "fitness": "DMS_score", "mutation": "mutant"},
               "family": x["UniProt_ID"], "stratum": x["coarse_selection_type"],
               "study": x["first_author"] + "_" + x["year"], "development": name in dev,
               "development_background": x["UniProt_ID"] in old_protein_backgrounds,
               "public": True, "source_kind": "ProteinGym", "source_url": "https://github.com/OATML-Markslab/ProteinGym",
               "reference": str(protein_ref), "reference_sha256": sha256_file(protein_ref),
               "expected_source_sha256": sha256_file(paths[0]) if paths else None,
               "pre_exclusion": "missing_source" if not paths else None}
    rna_ref = Path("/data/yanjie_huang/RG-LLM/data/RNAGym/fitness/reference_sheet_final.csv")
    reference = {x["DMS_ID"]: x for x in rows(rna_ref)}
    manifest_path = project / "runs_expansion/frozen_manifest.json"
    manifest = load(manifest_path)
    assert len(manifest["assays"]) == 70
    rna_root = Path("/data/yanjie_huang/enzyme1_server/rna_benchmark/z_rnagym_70/assays_classified")
    for frozen in manifest["assays"]:
        name = frozen["assay"]
        x = reference[name]
        paths = list(rna_root.rglob(name + ".csv"))
        native = not frozen["protein_stability_source"] and frozen["rna_type"] != "mRNA-coding"
        seq = x["RAW_CONSTRUCT_SEQ"].strip().upper().replace("T", "U")
        meta = {"assay_id": name, "domain": "rna", "wt_sequence": seq,
                "assay_description": x["ASSAY_DESCRIPTION"],
                "endpoint": x["RNA_TYPE_GRANULAR"] + "; " + x["ASSAY_DESCRIPTION"],
                "readout": "RNAGym DMS_score, unchanged processed benchmark units. Higher is the ranking target. No assumed WT value or unprovided conditions.",
                "higher_is_better": True}
        yield {"metadata": meta, "source": str(paths[0]) if len(paths) == 1 else None,
               "columns": {"sequence": "sequence", "fitness": "DMS_score", "mutation": "mutant"},
               "family": rna_family(name, x), "stratum": x["RNA_TYPE"],
               "study": x["AUTHOR"] + "_" + x["YEAR"], "development": name in dev,
               "development_background": name in dev, "public": True, "source_kind": "RNAGym",
               "source_url": "https://github.com/MarksLab-DasLab/RNAGym", "reference": str(rna_ref),
               "reference_sha256": sha256_file(rna_ref), "frozen_manifest": str(manifest_path),
               "frozen_manifest_sha256": sha256_file(manifest_path),
               "expected_source_sha256": frozen["data_sha256"],
               "pre_exclusion": ("protein_coding_phenotype" if not native else
                                  "unresolved_reference_annotation" if name == "Soo_2021_ribozyme" else
                                  "missing_or_ambiguous_source" if len(paths) != 1 else None)}
    # Existing normalized public experimental libraries retain their source audit.
    for x in previous:
        if x["assay_id"] not in {"F1U_generation1", "Sample2019_human_5UTR", "Cuperus2017_yeast_5UTR"}:
            continue
        yield {"metadata": x["metadata"], "source": x["source_path"], "columns": None,
               "family": x["assay_id"], "stratum": x["endpoint_group"], "study": x["study"],
               "development": True, "development_background": True, "public": True,
               "source_kind": "external_measured", "source_url": x["source_url"],
               "expected_source_sha256": x["source_sha256"], "historical_audit": x, "pre_exclusion": None}
    followup = project / "experiments/fewshot_followup_20260920"
    for audit in load(followup / "public_latest_audit.json"):
        name = audit["assay"]
        episodes = list((followup / "episodes_public_latest").rglob("episode.json"))
        matches = [load(p) for p in episodes if load(p)["assay_id"] == name]
        if not matches:
            continue
        yield {"metadata": matches[0], "source": audit["preparation"]["source_path"],
               "columns": audit["preparation"]["column_mapping"], "family": name,
               "stratum": "public_latest", "study": name, "development": True,
               "development_background": True, "public": True, "source_kind": "external_measured",
               "expected_source_sha256": audit["source_hash"], "historical_audit": audit,
               "source_url": "see historical_audit and docs/DATA_GUIDE.md", "pre_exclusion": None}


def rna_family(name, meta):
    text = (name + " " + meta["ASSAY_DESCRIPTION"]).lower()
    for marker, family in (("cpeb3", "CPEB3"), ("cepeb3", "CPEB3"), ("beck_", "CPEB3"),
                           ("trna", "tRNA"), ("janzen", "Janzen_aminoacylation"),
                           ("line1", "LINE1"), ("twister", "twister"), ("_tw", "twister"),
                           ("kobori_2016", "twister"), ("hammerhead", "hammerhead"),
                           ("_hh_", "hammerhead")):
        if marker in text:
            return family
    return name  # Distinct backgrounds, not a claim of independent evolutionary families.


def select(entries, domain, cap):
    candidates = [e for e in entries if e["metadata"]["domain"] == domain and e["eligible"]]
    selected, backgrounds, strata = [], set(), Counter()
    while candidates and len(selected) < cap:
        # Previously unused backgrounds first; spread endpoint categories and studies.
        candidates.sort(key=lambda e: (e["development_background"], e["development"],
                        e["family"] in backgrounds, strata[e["stratum"]],
                        digest([1, e["metadata"]["assay_id"]])))
        e = candidates.pop(0)
        # Exact reference duplicates are alternative endpoints, not new backgrounds.
        reference = e["metadata"].get("wt_sequence")
        if reference and any(s["metadata"].get("wt_sequence") == reference for s in selected):
            e["selection_exclusion"] = "same_reference_alternative_endpoint"
            continue
        selected.append(e)
        backgrounds.add(e["family"])
        strata[e["stratum"]] += 1
    return selected


def prepare(project, root):
    if (root / "COHORT.json").exists():
        raise ValueError("Already frozen; no overwrite")
    entries = []
    for e in descriptors(project):
        name = e["metadata"]["assay_id"]
        e["eligible"] = False
        try:
            if e["pre_exclusion"]:
                raise ValueError(e["pre_exclusion"])
            if sha256_file(e["source"]) != e["expected_source_sha256"]:
                raise ValueError("source_hash_mismatch")
            landscape = load_landscape(e["source"], e["metadata"], e["columns"])
            e["metadata"] = landscape.metadata
            wt = landscape.metadata["wt_sequence"]
            if wt and any(len(r["sequence"]) != len(wt) for r in landscape.records):
                raise ValueError("construct_length_mismatch")
            e["n_eligible"] = len(landscape.records)
            e["provenance"] = landscape.provenance
            e["orders"] = dict(Counter(str(r["mutation_order"]) for r in landscape.records))
            e["eligible"] = len(landscape.records) >= 282
            e["exclusion"] = None if e["eligible"] else "fewer_than_282_variants"
        except (ValueError, FileNotFoundError) as exc:
            e["exclusion"] = str(exc)[:300]
        entries.append(e)
        print(name, e["eligible"], e.get("n_eligible"), e.get("exclusion"), flush=True)
        save(root / "screening_partial.json", entries)
    dev_families = {e["family"] for e in entries if e["development"]}
    for e in entries:
        e["development_background"] |= e["family"] in dev_families
    selected = select(entries, "protein", 30) + select(entries, "rna", 20)
    for e in selected:
        name = e["metadata"]["assay_id"]
        landscape = load_landscape(e["source"], e["metadata"], e["columns"])
        e.update(freeze(landscape, root / "inputs" / name))
        e["selected"] = True
    # First new eligible background in each domain; full pilot includes feedback rounds.
    pilots = [next(e["metadata"]["assay_id"] for e in selected
                   if e["metadata"]["domain"] == domain and e["multi_round_eligible"])
              for domain in ("protein", "rna")]
    save(root / "screening.json", entries)
    save(root / "COHORT.json", {"assays": selected, "pilots": pilots, "seed": 1,
         "selection": "metadata-stratified; unused backgrounds first; no fitness or model performance selection",
         "rna_family_caveat": "Family annotations are conservative groupings, not guaranteed independent phylogenetic units",
         "sampling": "SHA256(seed=1,assay,namespace,sequence); supports nested; static32 equals random round32",
         "external_scope": "Existing audited public measured sources only; private sources excluded",
         "screened": dict(Counter(e["source_kind"] for e in entries))})
    save(root / "model_config.json", load(project / "experiments/fewshot_followup_20260920/model_config.json"))
    print("SELECTED", Counter(e["metadata"]["domain"] for e in selected), "PILOTS", pilots, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project, args.root)
