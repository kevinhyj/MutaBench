"""Pure model-facing CR requests with a separate, explicit measured support set."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass

from scipy.stats import rankdata

from evagent.integrated.chunked_scoring import (
    ScoringCandidate,
    ScoringChunk,
    build_scoring_prompt,
)
from evagent.integrated.ranking import RankingCase

from . import PROTOCOL

SYSTEM = (
    "Use the supplied task context and candidates. Estimate an independent continuous "
    "fitness score for every supplied mutant. Explicitly supplied measured examples may "
    "be used as references. Never infer access to experimental results for unmeasured candidates."
)
SHOTS = (0, 4, 8, 16, 32)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Condition:
    name: str
    shots: int
    representation: str


CONDITIONS = tuple(Condition(f"numeric_{k}", k, "numeric") for k in SHOTS) + (
    Condition("unlabeled_16", 16, "unlabeled"),
    Condition("shuffled_16", 16, "shuffled"),
    Condition("ordinal_16", 16, "ordinal"),
)
AVAILABLE_CONDITIONS = CONDITIONS + (Condition("unlabeled_32", 32, "unlabeled"),)


def resolve_conditions(names=None):
    if names is None:
        return CONDITIONS
    names = tuple(names)
    available = {c.name: c for c in AVAILABLE_CONDITIONS}
    if not names or len(set(names)) != len(names) or any(n not in available for n in names):
        raise ValueError("Conditions must be nonempty, unique, registered names")
    return tuple(available[n] for n in names)


def validate_inputs(episode, support):
    if not isinstance(episode.get("episode_id"), str):
        raise ValueError("Missing episode identity")
    if support.get("episode_id") != episode["episode_id"]:
        raise ValueError("Support/episode mismatch")
    if type(episode.get("higher_is_better")) is not bool:
        raise ValueError("Explicit measurement direction is required")
    if episode.get("domain") not in {"protein", "rna", "RNA"}:
        raise ValueError("Explicit protein/RNA domain is required")
    if not isinstance(episode.get("query"), list) or not episode["query"]:
        raise ValueError("Empty query")
    ids, sequences = set(), set()
    for kind, rows in (("query", episode["query"]), ("support", support["examples"])):
        for row in rows:
            for field in ("variant_id", "mutation", "sequence"):
                if not isinstance(row.get(field), str) or (field != "mutation" and not row[field].strip()):
                    raise ValueError(f"Missing {kind} {field}")
                if "\n" in row[field] or "\r" in row[field]:
                    raise ValueError("Multiline variant field")
            sequence_key = row["sequence"].upper()
            if episode["domain"].lower() == "rna":
                sequence_key = sequence_key.replace("T", "U")
            if row["variant_id"] in ids or sequence_key in sequences:
                raise ValueError("Duplicate or overlapping query/support")
            ids.add(row["variant_id"])
            sequences.add(sequence_key)
            if kind == "support":
                if type(row.get("fitness")) not in (int, float) or not math.isfinite(row["fitness"]):
                    raise ValueError("Support measurement must be finite")
    if any(not re.fullmatch(r"M\d+", row["variant_id"]) for row in episode["query"]):
        raise ValueError("Query IDs must use the legacy CR M namespace")
    if any(not re.fullmatch(r"S\d+", row["variant_id"]) for row in support["examples"]):
        raise ValueError("Measured references must use the separate S namespace")
    for rows in (episode["query"], support["examples"]):
        if len({int(row["variant_id"][1:]) for row in rows}) != len(rows):
            raise ValueError("Numeric candidate ID aliases are ambiguous")


def measured_examples(episode, support, condition):
    if condition not in AVAILABLE_CONDITIONS:
        raise ValueError("Unknown frozen condition")
    if len(support["examples"]) < condition.shots:
        raise ValueError("Insufficient measured examples; no silent shot reduction")
    selected = support["examples"][:condition.shots]
    rows = [{k: row[k] for k in ("variant_id", "mutation", "sequence")} for row in selected]
    if condition.representation == "unlabeled":
        return rows
    values = [row["fitness"] for row in selected]
    if condition.representation == "shuffled":
        # One fixed permutation shared across every query chunk of this episode.
        rng = random.Random(digest([PROTOCOL, episode["episode_id"], "shuffle"]))
        permutation = list(range(len(values)))
        rng.shuffle(permutation)
        if permutation == list(range(len(values))) and len(values) > 1:
            permutation = permutation[1:] + permutation[:1]
        values = [values[i] for i in permutation]
    if condition.representation == "ordinal":
        sign = -1 if episode["higher_is_better"] else 1
        ranks = rankdata([sign * v for v in values], method="average")
        for row, rank in zip(rows, ranks, strict=True):
            row["measured_rank"] = float(rank)
    else:
        for row, value in zip(rows, values, strict=True):
            row["measured_value"] = value
    return rows


def make_request(episode, support, condition, chunk_index, *, chunk_size=25):
    validate_inputs(episode, support)
    if chunk_size != 25:
        raise ValueError("This protocol freezes B=25")
    start = chunk_index * chunk_size
    if chunk_index < 0 or start >= len(episode["query"]):
        raise ValueError("Invalid chunk")
    query = episode["query"][start:start + chunk_size]
    # Strict safe projection: incidental fields, including injected labels, never render.
    context = {k: episode.get(k) for k in
               ("assay_id", "domain", "assay_description", "endpoint", "readout", "higher_is_better")}
    wt = episode.get("wt_sequence")
    preamble = "Assay context:\n" + json.dumps(context, ensure_ascii=False, sort_keys=True)
    preamble += "\n\nWild-type/reference sequence:\n" + (wt or "Unavailable; do not invent a WT.")
    preamble += "\n\n"
    rows = "\n".join(f"{q['variant_id']} [{q['mutation']}]: {q['sequence']}" for q in query)
    case = RankingCase(
        assay=episode["assay_id"], system_prompt=SYSTEM,
        user_prompt=preamble + f"**{len(query)} candidate mutants to rank**\n" + rows + "\nRank all candidates.",
        ids=[q["variant_id"] for q in query],
    )
    chunk = ScoringChunk(chunk_index, [ScoringCandidate(q["variant_id"], q["mutation"], q["sequence"]) for q in query])
    user = build_scoring_prompt(case, chunk, assay_memory={}, prompt_variant="v3")
    user = user.replace(
        "Follow this benchmark instruction exactly. Do not inspect labels, previous answers, or result files.",
        "Follow this benchmark instruction exactly. Use only the measured examples explicitly provided here. "
        "Do not inspect hidden candidate results, previous answers, or result files.", 1,
    )
    if not wt:
        user = user.replace("mutation(s) relative to WT", "mutation annotation if available")
        user = user.replace("Estimate each mutant relative to wild type and the assay definition",
                            "Estimate each mutant using the assay definition and available measured references; no WT is supplied")
        user = user.replace("0 is approximately neutral", "0 is the middle of the prediction scale, not an assumed measured WT")
    examples = measured_examples(episode, support, condition)
    instruction = (
        "These are sequence-only reference examples; no measurements are supplied."
        if condition.representation == "unlabeled" else
        "Measured ranks are within the supplied support only: 1 is best; equal measurements share average ranks. No absolute measurement scale is supplied."
        if condition.representation == "ordinal" else
        "Measured values use the raw assay readout and direction specified above. Do not assume the WT was measured or assign it a value."
    )
    block = ("\n\nReference examples (S IDs are context, never prediction targets):\n" + instruction
             + "\n" + json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if not examples:
        block = "\n\nNo measured reference examples are supplied."
    marker = f"**{len(query)} candidate mutants to score**"
    user = user.replace(marker, block + "\n\n" + marker, 1)
    result = {"system": SYSTEM, "user": user, "expected_ids": chunk.ids,
              "episode_id": episode["episode_id"], "condition": asdict(condition),
              "chunk_index": chunk_index, "support_ids": [r["variant_id"] for r in examples],
              "protocol": PROTOCOL}
    result["request_hash"] = digest(result)
    return result


def all_requests(episode, support, conditions=None):
    # Interleave arms by chunk; rotate deterministically to reduce scheduling confounding.
    selected = resolve_conditions(conditions)
    shift = int(digest(episode["episode_id"])[:8], 16) % len(selected)
    arms = selected[shift:] + selected[:shift]
    return [make_request(episode, support, condition, i)
            for i in range((len(episode["query"]) + 24) // 25) for condition in arms]
