"""Label-free ordering contracts. Dataset cleaning and label aggregation are separate."""
from __future__ import annotations
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8")).hexdigest()


def canonical_sites(sites):
    normalized = []
    for s in sites:
        pos = s.get("position")
        if isinstance(pos, bool) or not isinstance(pos, int):
            raise ValueError("Position must be an explicit integer in the frozen coordinate system")
        row = dict(chain=str(s["chain"]), position=pos,
                   insertion_code=str(s.get("insertion_code") or ""),
                   wt=s.get("wt", s.get("wt_aa")), mutant=s.get("mutant", s.get("mutant_aa")))
        if not row["chain"] or not isinstance(row["wt"], str) or len(row["wt"]) != 1 or not isinstance(row["mutant"], str) or len(row["mutant"]) != 1:
            raise ValueError("Chain, WT and mutant must be explicitly provided")
        normalized.append(row)
    normalized.sort(key=lambda s: (s["chain"], s["position"], s["insertion_code"], s["wt"], s["mutant"]))
    if not normalized:
        raise ValueError("Mutation sites cannot be empty")
    positions=[(s["chain"],s["position"],s["insertion_code"]) for s in normalized]
    if len(set(positions)) != len(positions):
        raise ValueError("Repeated/conflicting mutation at one residue")
    return normalized


def complex_groups(records, n, seed=42):
    """One source/assay only; return identities with all underlying measurement IDs.

    No arbitrary representative measurement or label is selected here.
    The priority exactly follows historical sha([42, sha(normalized_sites)]).
    Qualified canonical pools are inputs, never treated as pre-selected N rows.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    records=list(records)
    scopes={(r["source"],r["assay_group_id"]) for r in records}
    if len(scopes) != 1:
        raise ValueError("Exactly one source and assay is required")
    source, assay = next(iter(scopes))
    groups={}
    for r in records:
        sites=canonical_sites(r["chain_mutations"])
        vk=digest(sites)
        ref=r["reference_complex_id"]
        key=(source,assay,ref,vk)
        g=groups.setdefault(key,dict(source=source,assay_id=assay,reference_complex_id=ref,
                                    variant_key=vk,variant_id="v_"+vk[:20],
                                    priority=digest([seed,vk]),sites=sites,source_fact_ids=[]))
        g["source_fact_ids"].append(r["fact_id"])
    if len(groups)<n:
        raise ValueError(f"Insufficient unique variants: {len(groups)} < {n}")
    selected=sorted(groups.values(),key=lambda g:(g["priority"],g["variant_id"],g["reference_complex_id"]))[:n]
    for g in selected:
        g["source_fact_ids"]=sorted(set(g["source_fact_ids"]))
        g["measurement_resolution_required"]=len(g["source_fact_ids"])>1
    return selected


def singlechain_select(unique_valid_rows, n=250, seed=42):
    """After source-specific validity/duplicate adjudication, one assay at a time."""
    import numpy as np
    if isinstance(n,bool) or not isinstance(n,int) or n<1:
        raise ValueError("n must be a positive integer")
    rows=list(unique_valid_rows)
    if any(not isinstance(r.get("sequence"),str) or not r["sequence"] for r in rows):
        raise ValueError("Complete sequence required")
    if len({r["sequence"] for r in rows}) != len(rows):
        raise ValueError("Resolve duplicate sequences before sampling")
    rows.sort(key=lambda r:(r["mutation"],r["sequence"]))
    order=np.random.RandomState(seed).permutation(len(rows))[:min(n,len(rows))]
    return [rows[int(i)] for i in order]
