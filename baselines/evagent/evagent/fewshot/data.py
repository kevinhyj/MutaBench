"""Offline, label-separated preparation of frozen few-shot episodes.

Only this preparation module and the evaluator may open the full landscape.
Inference receives ``episode.json`` and a prefix of ``support.json``; neither
contains the path, hash, or measurements of the hidden query label table.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_SHOTS = (0, 4, 8, 16, 32)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)
SPLITS = ("random", "low_order_to_high_order")
METADATA_FIELDS = (
    "assay_id", "domain", "wt_sequence", "assay_description", "endpoint",
    "readout", "higher_is_better",
)
VARIANT_FIELDS = ("variant_id", "sequence", "mutation", "mutation_order")
_ALPHABETS = {"protein": set("ACDEFGHIKLMNPQRSTVWY"), "rna": set("ACGU")}


class NotEligibleError(ValueError):
    """An assay cannot satisfy the requested design; never shrink it silently."""


@dataclass(frozen=True)
class Landscape:
    metadata: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence(value: Any, domain: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("empty_sequence")
    sequence = value.strip().upper()
    if domain == "rna":
        sequence = sequence.replace("T", "U")
    if set(sequence) - _ALPHABETS[domain]:
        raise ValueError("invalid_sequence_alphabet")
    return sequence


def _metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an explicit allowlist, never propagate arbitrary source columns."""
    result = {key: metadata[key] for key in METADATA_FIELDS if key in metadata}
    if not isinstance(result.get("assay_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", result["assay_id"]
    ):
        raise ValueError("assay_id must be a safe, nonempty identifier")
    domain = str(result.get("domain", "")).lower()
    if domain not in _ALPHABETS:
        raise ValueError("domain must be protein or rna")
    result["domain"] = domain
    if type(result.get("higher_is_better")) is not bool:
        raise ValueError("higher_is_better must be explicitly true or false")
    for key in ("assay_description", "endpoint", "readout"):
        value = result.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        result[key] = value
    if not result["endpoint"].strip() or not result["assay_description"].strip():
        raise ValueError("assay_description and endpoint must be provided")
    wt = result.get("wt_sequence")
    result["wt_sequence"] = _sequence(wt, domain) if wt else None
    return result


def _explicit_order(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("invalid_mutation_order")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid_mutation_order") from error
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError("invalid_mutation_order")
    return int(numeric)


def load_landscape(
    source_path: str | Path,
    metadata: Mapping[str, Any],
    columns: Mapping[str, str] | None = None,
) -> Landscape:
    """Read a CSV/TSV or JSON list (or ``{"records": [...]}``).

    Column mapping keys are sequence, fitness, mutation, mutation_order. Defaults
    use these same names. Invalid/nonfinite rows are counted in provenance.
    Canonical identical sequences with equal measurements are collapsed;
    conflicting replicate measurements fail instead of silently averaging them.
    WT/reference rows are excluded. With a reference and equal lengths, mutation
    order is exact Hamming distance and mutation descriptions are regenerated.
    Without a reference, only an explicit mutation_order column establishes order.
    """
    source = Path(source_path).resolve()
    meta = _metadata(metadata)
    mapping = {key: key for key in ("sequence", "fitness", "mutation", "mutation_order")}
    if columns:
        unknown = set(columns) - set(mapping)
        if unknown:
            raise ValueError(f"unknown column mappings: {sorted(unknown)}")
        mapping.update(columns)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        rows = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON source must be a record list or contain records")
    else:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if source.suffix.lower() == ".tsv" else ",")
            missing = {mapping["sequence"], mapping["fitness"]} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"missing required CSV columns: {sorted(missing)}")
            rows = list(reader)
    seen: dict[str, dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    order_basis: Counter[str] = Counter()
    wt = meta["wt_sequence"]
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"row {row_number} is not an object")
        try:
            sequence = _sequence(row.get(mapping["sequence"]), meta["domain"])
            raw_fitness = row.get(mapping["fitness"])
            if isinstance(raw_fitness, bool):
                raise ValueError("invalid_fitness")
            try:
                fitness = float(raw_fitness)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid_fitness") from error
            if not math.isfinite(fitness):
                raise ValueError("nonfinite_fitness")
            explicit = _explicit_order(row.get(mapping["mutation_order"]))
            mutation = str(row.get(mapping["mutation"]) or "")
            if wt is not None and len(sequence) == len(wt):
                changes = [(i, a, b) for i, (a, b) in enumerate(zip(wt, sequence), 1) if a != b]
                order = len(changes)
                if explicit is not None and explicit != order:
                    raise ValueError("mutation_order_reference_mismatch")
                mutation = ",".join(f"{a}{i}{b}" for i, a, b in changes)
                basis = "reference_hamming_distance"
            else:
                order = explicit
                basis = "explicit_source_field" if explicit is not None else "unknown"
            if order == 0:
                excluded["reference_sequence"] += 1
                continue
        except ValueError as error:
            excluded[str(error)] += 1
            continue
        record = {"sequence": sequence, "mutation": mutation, "mutation_order": order,
                  "fitness": fitness}
        if sequence in seen:
            previous = seen[sequence]
            if previous["fitness"] != fitness:
                raise ValueError(f"conflicting fitness for duplicate sequence at row {row_number}; "
                                 "resolve replicate policy explicitly before preparation")
            if previous["mutation_order"] != order:
                raise ValueError(f"conflicting mutation_order for duplicate sequence at row {row_number}")
            # Aliases are metadata only; retain lexicographically smallest alias so
            # identical records remain invariant to input-row order even without WT.
            previous["mutation"] = min(previous["mutation"], mutation)
            excluded["duplicate_sequence"] += 1
            continue
        seen[sequence] = record
        order_basis[basis] += 1
    if not seen:
        raise NotEligibleError("no valid non-reference, uniquely measured sequences")
    records = tuple(seen[key] for key in sorted(seen))
    provenance = {
        "source_path": str(source), "source_sha256": sha256_file(source),
        "source_bytes": source.stat().st_size, "column_mapping": mapping,
        "input_rows": len(rows), "eligible_unique_sequences": len(records),
        "excluded_rows_by_reason": dict(sorted(excluded.items())),
        "mutation_order_basis": dict(sorted(order_basis.items())),
        "metadata_sha256": hashlib.sha256(_json_bytes(meta)).hexdigest(),
        "normalization": "uppercase canonical sequences; RNA T converted to U; measurements unchanged",
        "replicate_policy": "collapse identical sequence/measurement pairs; reject conflicting measurements",
    }
    return Landscape(meta, records, provenance)


def _permutation(records: Sequence[dict[str, Any]], seed: int, namespace: str) -> list[dict[str, Any]]:
    """A row-order-independent permutation whose key never reads a measurement."""
    return sorted(records, key=lambda record: (
        hashlib.sha256(f"fewshot-v1\0{namespace}\0{seed}\0{record['sequence']}".encode()).digest(),
        record["sequence"],
    ))


def freeze_episode(
    landscape: Landscape,
    split: str,
    seed: int,
    query_size: int = 250,
    shots: Sequence[int] = DEFAULT_SHOTS,
) -> dict[str, dict[str, Any]]:
    """Freeze a shared query and nested support set for every requested shot."""
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(query_size) is not int or query_size < 1:
        raise ValueError("query_size must be a positive integer")
    if (not shots or any(type(n) is not int or n < 0 for n in shots)
            or list(shots) != sorted(set(shots)) or shots[0] != 0):
        raise ValueError("shots must be sorted unique nonnegative integers starting with zero")
    max_shot = max(shots)
    rows = landscape.records
    if split == "random":
        if len(rows) < query_size + max_shot:
            raise NotEligibleError(f"random requires {query_size + max_shot} unique variants; has {len(rows)}")
        permuted = _permutation(rows, seed, "random")
        query_rows = permuted[:query_size]
        support_rows = permuted[query_size:query_size + max_shot]
        pool_counts = {"query_and_support": len(rows)}
    else:
        low = [row for row in rows if row["mutation_order"] in (1, 2)]
        high = [row for row in rows if row["mutation_order"] is not None and row["mutation_order"] >= 3]
        if len(low) < max_shot or len(high) < query_size:
            raise NotEligibleError(f"low_order_to_high_order requires {max_shot} order-1/2 supports "
                                   f"and {query_size} order>=3 queries; has {len(low)} and {len(high)}")
        support_rows = _permutation(low, seed, "low_order_support")[:max_shot]
        query_rows = _permutation(high, seed, "high_order_query")[:query_size]
        pool_counts = {"order_1_or_2": len(low), "order_3_or_higher": len(high),
                       "unknown_order_excluded": sum(row["mutation_order"] is None for row in rows)}
    def public_variant(row: dict[str, Any], variant_id: str) -> dict[str, Any]:
        return {"variant_id": variant_id, "sequence": row["sequence"],
                "mutation": row["mutation"], "mutation_order": row["mutation_order"]}
    meta = dict(landscape.metadata)
    episode_id = f"{meta['assay_id']}__{split}__seed{seed}"
    query = [public_variant(row, f"M{i:03d}") for i, row in enumerate(query_rows, 1)]
    examples = [dict(public_variant(row, f"S{i:03d}"), fitness=row["fitness"])
                for i, row in enumerate(support_rows, 1)]
    episode = dict(meta, schema="fewshot-episode-v1", episode_id=episode_id,
                   split=split, seed=seed, shots=list(shots), query=query)
    support = {"schema": "fewshot-support-v1", "episode_id": episode_id, "examples": examples}
    evaluation = {"schema": "fewshot-evaluation-v1", "episode_id": episode_id,
                  "higher_is_better": meta["higher_is_better"],
                  "labels": {variant["variant_id"]: row["fitness"] for variant, row in zip(query, query_rows)}}
    manifest = {
        "schema": "fewshot-freeze-v1", "episode_id": episode_id,
        "source": dict(landscape.provenance), "pool_counts": pool_counts,
        "sampling": "sha256(seed,split-namespace,sequence); no fitness stratification or sorting",
        "split": split, "seed": seed, "query_size": query_size, "shots": list(shots),
        "support_policy": "take examples[:shots]; every method shares exactly these prefixes",
        "reference_policy": "order-zero/reference rows excluded",
        "label_isolation": "query measurements exist only in evaluation_labels.json; no whole-landscape normalization",
        "files": {name: hashlib.sha256(_json_bytes(value)).hexdigest() for name, value in (
            ("episode.json", episode), ("support.json", support), ("evaluation_labels.json", evaluation))},
    }
    return {"episode": episode, "support": support, "evaluation_labels": evaluation, "manifest": manifest}


def write_episode(artifacts: Mapping[str, Mapping[str, Any]], output_path: str | Path) -> dict[str, str]:
    """Write immutable artifacts, or verify an identical previous preparation.

    The returned paths are for orchestration only: never pass the manifest or the
    evaluation path to an inference process. Label files are owner-readable only.
    """
    output = Path(output_path)
    payloads = {f"{name}.json": _json_bytes(artifacts[name])
                for name in ("episode", "support", "evaluation_labels", "manifest")}
    for name, value in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != value:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(value)
            if name in ("support.json", "evaluation_labels.json"):
                path.chmod(0o600)
    return {name.removesuffix(".json"): str(output / name) for name in payloads}
