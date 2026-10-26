from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = next(parent for parent in ROOT.parents
               if (parent / 'code/evagent/integrated/gateway.py').is_file())
ALPHABETS = {"protein": "ACDEFGHIKLMNPQRSTVWY", "rna": "ACGU"}


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Frozen artifact exists: {path}; use a new output directory")
        return
    # Exclusive create: never overwrite historical or concurrent results.
    with path.open("x") as f:
        f.write(text)


def normalize(sequence, modality):
    sequence = sequence.strip()
    if modality in ALPHABETS:
        sequence = sequence.upper()
        if modality == "rna":
            sequence = sequence.replace("T", "U")
        if not sequence or set(sequence) - set(ALPHABETS[modality]):
            raise ValueError("Invalid sequence alphabet")
    elif modality == "ligand":
        from rdkit import Chem
        mol = Chem.MolFromSmiles(sequence)
        if mol is None:
            raise ValueError("Invalid SMILES")
        sequence = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    else:
        raise ValueError("Unknown modality")
    return sequence


def validate_input(item, *, smoke=False):
    allowed = {"schema", "assay", "modality", "n", "seed", "base", "endpoint",
               "candidates", "evidence", "smoke_only"}
    if set(item) != allowed:
        raise ValueError(f"Unexpected/missing input fields: {set(item) ^ allowed}")
    if item["schema"] != "singlechain-v1" or item["seed"] != 42:
        raise ValueError("Expected singlechain-v1, seed=42")
    if not isinstance(item["smoke_only"], bool) or (item["smoke_only"] and not smoke):
        raise ValueError("Smoke data cannot be admitted as N250 experiment")
    if item["n"] != len(item["candidates"]) or (not item["smoke_only"] and item["n"] != 250):
        raise ValueError("Expected exactly N=250 candidates")
    if set(item["endpoint"]) != {"name", "kind", "higher_means", "label_sign"}:
        raise ValueError("Endpoint must define kind, meaning and direction before scoring")
    if item["endpoint"]["label_sign"] not in (-1, 1):
        raise ValueError("Label direction must be +1 or -1")
    if item["endpoint"]["kind"] not in ("regression", "classification"):
        raise ValueError("Unknown endpoint kind")
    if set(item["evidence"]) - {"msa", "reference_structure", "structure", "potts"}:
        raise ValueError("Unexpected evidence type")
    modality = item["modality"]
    if normalize(item["base"], modality) != item["base"]:
        raise ValueError("Base is not canonical")
    ids, sequences = set(), set()
    for row in item["candidates"]:
        if set(row) != {"id", "sequence"} or type(row["id"]) is not int:
            raise ValueError("Candidate must contain only integer id and sequence/SMILES")
        if row["id"] in ids or row["sequence"] in sequences:
            raise ValueError("Duplicate id or sequence")
        ids.add(row["id"]); sequences.add(row["sequence"])
        if normalize(row["sequence"], modality) != row["sequence"]:
            raise ValueError("Candidate is not canonical")
        if modality in ALPHABETS and len(row["sequence"]) != len(item["base"]):
            raise ValueError("This protocol supports substitutions; indels require a new protocol")
    return item


def mutations(wt, seq):
    if len(wt) != len(seq):
        raise ValueError("Length mismatch")
    return [(i, a, b) for i, (a, b) in enumerate(zip(wt, seq)) if a != b]


def records(item, scores, base=0.0, rationale="Model-native score; relative to a fixed reference."):
    if len(scores) != item["n"]:
        raise ValueError("Incomplete predictions")
    ligand = item["modality"] == "ligand"
    out = []
    for candidate, score in zip(item["candidates"], scores):
        if not math.isfinite(float(score)) or not math.isfinite(float(base)):
            raise ValueError("Nonfinite prediction")
        out.append({"id": candidate["id"], "base_fitness": float(base),
                    "pred_fitness": float(score), "delta_fitness": float(score - base),
                    "ref_fitness" if ligand else "wt_fitness": float(base),
                    "cand_fitness" if ligand else "mut_fitness": float(score),
                    "rationale": rationale})
    return out


def validate_predictions(item, values):
    if not isinstance(values, list) or len(values) != item["n"]:
        raise ValueError("Prediction list must cover every candidate exactly once")
    wanted = {c["id"] for c in item["candidates"]}
    found = set()
    base = None
    for row in values:
        if type(row.get("id")) is not int or row["id"] not in wanted or row["id"] in found:
            raise ValueError("Unknown or duplicate prediction id")
        found.add(row["id"])
        for key in ("base_fitness", "pred_fitness", "delta_fitness"):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError(f"Invalid {key}")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError("Missing rationale")
        if not math.isclose(row["delta_fitness"], row["pred_fitness"] - row["base_fitness"],
                            rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("delta_fitness is not pred_fitness - base_fitness")
        if base is not None and not math.isclose(base, row["base_fitness"], abs_tol=1e-9):
            raise ValueError("Base must be identical across the assay")
        base = row["base_fitness"]
        aliases = (("ref_fitness", "base_fitness"), ("cand_fitness", "pred_fitness")) if item["modality"] == "ligand" else (("wt_fitness", "base_fitness"), ("mut_fitness", "pred_fitness"))
        for alias, key in aliases:
            if alias in row and row[alias] != row[key]:
                raise ValueError("Conflicting modality-specific field")
    return {r["id"]: r for r in values}
