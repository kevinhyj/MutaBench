"""Load frozen samples into a label-free model contract; score only after prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evagent.graph import model_metadata
from evagent.runtime import digest

from .ranking import RankingCase


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and fingerprint the same captured bytes, never two file reads."""
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _input_string(value: Any, name: str, *, sequence: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a single line")
    if sequence and any(character.isspace() for character in value):
        raise ValueError(f"{name} must be one complete, whitespace-free sequence")
    return value


def _model_metadata(value: Any, assay: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Context metadata must be an object")
    selected = model_metadata(value)
    for key, item in selected.items():
        if item is not None and type(item) not in (str, int, float, bool):
            raise ValueError(f"Context metadata {key} must be a scalar value")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"Context metadata {key} must be finite")
    if selected.get("DMS_ID") not in (None, "", assay):
        raise ValueError("Context metadata DMS_ID does not match sample")
    return selected


def _assay_summary(record: dict[str, Any], assay: str) -> dict[str, Any]:
    if "status" in record and record["status"] != "complete":
        raise ValueError("Frozen assay summary is not complete")
    data = record.get("data", record)
    if not isinstance(data, dict):
        raise ValueError("Frozen assay summary data must be an object")
    if data.get("assay_id", assay) != assay:
        raise ValueError("Assay summary does not match sample")
    memory: dict[str, Any] = {}
    for key in ("assay_summary", "property_direction"):
        if key in data:
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"Assay summary {key} must be a non-empty string")
            memory[key] = data[key]
    for key in ("functional_hypotheses", "sequence_features"):
        if key in data:
            if not isinstance(data[key], list) or any(not isinstance(item, str) for item in data[key]):
                raise ValueError(f"Assay summary {key} must be a list of strings")
            memory[key] = list(data[key])
    return memory


@dataclass(frozen=True)
class ModelTask:
    """No experimental labels or evaluator handles are present in this object."""

    assay: str
    variants: list[dict[str, str]]
    metadata: dict[str, Any]
    wt_sequence: str
    case: RankingCase
    assay_memory: dict[str, Any]
    input_hashes: dict[str, str]


def load_task(config: dict[str, Any]) -> ModelTask:
    sample_path = Path(config["sample_file"])
    sample, sample_hash = _json_snapshot(sample_path)
    assay = _input_string(sample.get("assay"), "Sample assay")
    rows = sample.get("variants")
    if not assay or not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Sample must name its assay and contain at least two variants")
    if "size" in sample and (type(sample["size"]) is not int or sample["size"] != len(rows)):
        raise ValueError("Frozen sample size does not match its variants")
    context: dict[str, Any] = {}
    hashes = {"sample": sample_hash}
    if config.get("context_file"):
        path = Path(config["context_file"])
        context, hashes["context"] = _json_snapshot(path)
        if context.get("assay", assay) != assay:
            raise ValueError("Context and sample assay do not match")
    elif config.get("rg_llm_root"):
        path = Path(config["rg_llm_root"]) / "data/RNAGym/fitness/reference_sheet_final.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                if record.get("DMS_ID") == assay:
                    context = {"metadata": model_metadata(record),
                               "wt_sequence": record.get("RAW_CONSTRUCT_SEQ", "")}
                    break
        if not context:
            raise ValueError(f"No reference context found for {assay}")
        hashes["context"] = digest(context)
    else:
        raise ValueError("Provide context_file or rg_llm_root for frozen task context")
    metadata = _model_metadata(context.get("metadata", {}), assay)
    wt = _input_string(context.get("wt_sequence"), "WT sequence", sequence=True)
    variants: list[dict[str, str]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("Every sample variant must be an object")
        candidate = {
            "variant_id": _input_string(row.get("variant_id", f"M{index:02d}"), "Candidate ID"),
            "mutation": _input_string(row.get("mutation", row.get("id")), "Candidate mutation"),
            "sequence": _input_string(row.get("sequence", row.get("seq")), "Candidate sequence", sequence=True),
        }
        if re.fullmatch(r"M\d+", candidate["variant_id"]) is None:
            raise ValueError("Integrated CR/ACR candidate IDs must use M followed by digits")
        variants.append(candidate)
    ids = [row["variant_id"] for row in variants]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs in sample")
    if len(ids) != len({int(value[1:]) for value in ids}):
        raise ValueError("Candidate IDs contain duplicate numeric identities (for example M1 and M01)")
    memory: dict[str, Any] = {}
    if config.get("assay_memory_file"):
        path = Path(config["assay_memory_file"])
        record, hashes["assay_memory"] = _json_snapshot(path)
        memory = _assay_summary(record, assay)
    # This layout matches the migrated CR/ACR parser. Only allowed input fields enter it.
    system = ("Use the supplied task context and candidates. Rank the mutants by their predicted "
              "effect on the assayed RNA property, then output the ranking in the requested JSON format.")
    user = "\n".join([
        "Assay metadata:", json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "", "Wild-type sequence:", wt, "",
        f"**{len(ids)} candidate mutants to rank** (id, mutation(s) relative to WT, full sequence):",
        *[f"{row['variant_id']} [{row['mutation']}]: {row['sequence']}" for row in variants],
        "", f"Rank all {len(ids)} mutants best to worst. Return only a JSON object with a ranking array.",
    ])
    case = RankingCase(assay=assay, ids=ids, system_prompt=system, user_prompt=user,
                       prompt_sha256=digest([system, user]), ground_truth_scores=None,
                       source={"kind": "integrated-frozen-sample", "sha256": hashes["sample"]})
    return ModelTask(assay, variants, metadata, wt, case, memory, hashes)


def evaluate_result(sample_file: Path, expected_sample_hash: str,
                    ids: list[str], result: dict[str, Any]) -> dict[str, Any]:
    """The sole label loader; called after a complete prediction, never by an agent."""
    if result.get("status") != "ok":
        return {"status": "not_evaluated_incomplete"}
    ranking = result.get("ranking", [])
    if len(ranking) != len(ids) or set(ranking) != set(ids):
        raise ValueError("Evaluator requires a complete permutation")
    sample, observed_hash = _json_snapshot(sample_file)
    if observed_hash != expected_sample_hash:
        raise ValueError("Sample changed after request preparation")
    rows = sample.get("variants")
    if not isinstance(rows, list) or len(rows) != len(ids) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Evaluator sample must contain the expected candidate count")
    sample_ids = [_input_string(row.get("variant_id", f"M{index:02d}"), "Candidate ID")
                  for index, row in enumerate(rows, 1)]
    if sample_ids != ids:
        raise ValueError("Evaluator candidate identities/order do not match the frozen input")
    if not all("score" in row or "experimental_score" in row for row in rows):
        return {"status": "labels_unavailable"}
    truth = [float(row.get("score", row.get("experimental_score"))) for row in rows]
    if not all(math.isfinite(value) for value in truth):
        raise ValueError("Evaluator labels must be finite")
    from scipy.stats import kendalltau, spearmanr

    score_map = result.get("scores")
    if score_map:
        predicted = [float(score_map[value]) for value in ids]
    else:
        ranks = {value: -index for index, value in enumerate(ranking)}
        predicted = [ranks[value] for value in ids]
    rho = float(spearmanr(predicted, truth).statistic)
    tau = float(kendalltau(predicted, truth).statistic)
    return {"status": "ok" if math.isfinite(rho) else "undefined_constant_input",
            "n": len(ids), "spearman": rho if math.isfinite(rho) else None,
            "kendall_tau": tau if math.isfinite(tau) else None}
