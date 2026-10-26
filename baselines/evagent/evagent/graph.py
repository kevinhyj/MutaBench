"""Fixed LangGraph workflow for scalable fitness-landscape reconstruction."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Protocol

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from .config import PipelineConfig
from .model import ModelError, StructuredResponse
from .ranking import (
    aggregate_rankings,
    estimate_uncertainty,
    make_anchor_groups,
    validate_group_coverage,
    verification_candidates,
)
from .schemas import (
    AssayMemory,
    EvidenceRef,
    GraphState,
    LocalRanking,
    PredictionResult,
    VariantEvidence,
)

METADATA_FIELDS = {
    "DMS_ID",
    "TITLE",
    "AUTHOR",
    "YEAR",
    "RNA_TYPE",
    "RNA_TYPE_GRANULAR",
    "ASSAY_DESCRIPTION",
}


def model_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key in METADATA_FIELDS}


ASSAY_SYSTEM = """You are the assay-level analyst in an RNA mutation-effect system.
Infer only general, zero-shot hypotheses from the supplied assay metadata and wild-type
sequence. Do not claim access to experimental mutant fitness values. Return JSON only."""

ASSAY_PROMPT = PromptTemplate.from_template(
    """Analyze this assay and return exactly:
{{"assay_summary":"...","property_direction":"what higher fitness means",
"functional_hypotheses":["..."],"sequence_features":["..."]}}

Assay metadata:
{metadata}

Wild-type sequence ({length} nt):
{wt_sequence}"""
)

VARIANT_SYSTEM = """You are the variant-level mechanistic reasoner in a zero-shot RNA
fitness system. Analyze each candidate independently. Use only supplied sequence and assay
information; never infer or invent experimental scores. predicted_score is a relative prior
in [-1,1], where larger means higher assayed fitness. Return JSON only."""

VARIANT_PROMPT = PromptTemplate.from_template(
    """Assay memory:
{assay_memory}

Analyze every candidate below. Return exactly:
{{"variants":[{{"variant_id":"M001","mutation":"...",
"direction":"beneficial|neutral|deleterious|uncertain","predicted_score":0.0,
"mechanisms":["..."],"evidence":[{{"evidence_id":"sequence:M001",
"evidence_type":"sequence","claim":"..."}}],"confidence":0.0}}]}}

Candidates with local WT/mutant sequence context:
{candidates}"""
)

RANK_SYSTEM = """You are a local ranking specialist. Rank all supplied variants from
highest to lowest predicted fitness for the stated assay. Ground the ordering in the
structured evidence. Return JSON only, with every supplied ID exactly once."""

RANK_PROMPT = PromptTemplate.from_template(
    """Assay memory:
{assay_memory}

Variant evidence:
{evidence}

Return exactly {{"ranking":[all IDs best-to-worst],"rationale":["brief checks"]}}."""
)

VERIFY_SYSTEM = """You verify a difficult local region of a global RNA fitness ranking.
Resolve conflicts using the supplied structured evidence. Return every supplied ID exactly
once from highest to lowest predicted assay fitness. Return JSON only."""

VERIFY_PROMPT = PromptTemplate.from_template(
    """Assay memory:
{assay_memory}

Current uncertain region, already listed approximately best-to-worst:
{ordered_ids}

Evidence:
{evidence}

Return exactly {{"ranking":[all IDs best-to-worst],"rationale":["brief conflict checks"]}}."""
)


def prompt_fingerprint() -> str:
    content = "\n\n".join(
        [
            "metadata-whitelist-v2:" + ",".join(sorted(METADATA_FIELDS)),
            ASSAY_SYSTEM,
            ASSAY_PROMPT.template,
            VARIANT_SYSTEM,
            VARIANT_PROMPT.template,
            RANK_SYSTEM,
            RANK_PROMPT.template,
            VERIFY_SYSTEM,
            VERIFY_PROMPT.template,
        ]
    )
    return hashlib.sha256(content.encode()).hexdigest()


class StructuredModel(Protocol):
    def generate_json(
        self,
        *,
        system: str,
        user: str,
        run_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> StructuredResponse: ...


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _parallel_map(function, payloads: list[Any], max_workers: int) -> list[Any]:
    """Map in parallel while propagating the parent LangSmith trace context."""
    if max_workers <= 1 or len(payloads) <= 1:
        return [function(payload) for payload in payloads]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_context().run, function, payload) for payload in payloads]
        return [future.result() for future in futures]


def _call_records(response: StructuredResponse, stage: str, unit: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for call in response.calls:
        row = call.model_dump()
        row.update({"record_type": "model_call", "stage": stage, "unit": unit})
        records.append(row)
    return records


def _canonical_id(value: Any, expected: set[str]) -> str | None:
    text = str(value).strip().upper()
    if text in expected:
        return text
    match = re.fullmatch(r"M0*(\d+)", text)
    if not match:
        return None
    number = int(match.group(1))
    for candidate in expected:
        candidate_match = re.fullmatch(r"M0*(\d+)", candidate)
        if candidate_match and int(candidate_match.group(1)) == number:
            return candidate
    return None


def _strict_ranking(raw: Any, expected_ids: list[str]) -> tuple[list[str], list[str]]:
    if not isinstance(raw, dict):
        raise ValueError("ranking response must be a JSON object")
    expected = set(expected_ids)
    normalized: list[str] = []
    for value in raw.get("ranking", []):
        canonical = _canonical_id(value, expected)
        if canonical is None or canonical in normalized:
            raise ValueError("ranking contains an unknown or duplicate ID")
        normalized.append(canonical)
    if set(normalized) != expected or len(normalized) != len(expected_ids):
        raise ValueError("ranking is not a strict permutation of the candidate IDs")
    rationale = [str(x) for x in raw.get("rationale", [])]
    return normalized, rationale


def _sequence_context(wt: str, mutant: str, flank: int = 12) -> dict[str, Any]:
    if wt == mutant:
        return {"note": "sequence equals WT"}
    if len(wt) != len(mutant):
        if len(mutant) <= 300:
            return {"length_change": len(mutant) - len(wt), "mutant_sequence": mutant}
        return {
            "length_change": len(mutant) - len(wt),
            "mutant_prefix": mutant[:80],
            "mutant_suffix": mutant[-80:],
        }
    positions = [i for i, (a, b) in enumerate(zip(wt, mutant)) if a != b]
    contexts = []
    for position in positions[:12]:
        start, end = max(0, position - flank), min(len(wt), position + flank + 1)
        contexts.append(
            {
                "position_1based": position + 1,
                "wt_context": wt[start:end],
                "mutant_context": mutant[start:end],
            }
        )
    return {"changed_positions": len(positions), "contexts": contexts}


class EVAgentWorkflow:
    def __init__(self, model: StructuredModel, config: PipelineConfig):
        config.validate()
        self.model = model
        self.config = config

    def load_assay(self, state: GraphState) -> dict[str, Any]:
        required = {"assay_id", "metadata", "wt_sequence", "variants"}
        missing = required - state.keys()
        if missing:
            raise ValueError(f"initial state missing: {sorted(missing)}")
        ids = [str(x["variant_id"]) for x in state["variants"]]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("variant IDs must be unique and contain at least two candidates")
        return {"status": "loaded", "usage_records": [], "reasoning_records": []}

    def assay_analyst(self, state: GraphState) -> dict[str, Any]:
        user = ASSAY_PROMPT.format(
            metadata=_compact_json(model_metadata(state["metadata"])),
            length=len(state["wt_sequence"]),
            wt_sequence=state["wt_sequence"],
        )
        records: list[dict[str, Any]] = []
        try:
            response = self.model.generate_json(
                system=ASSAY_SYSTEM,
                user=user,
                run_name="assay_analyst",
                metadata={"assay": state["assay_id"]},
            )
            records = _call_records(response, "assay_analyst", state["assay_id"])
            raw = response.data if isinstance(response.data, dict) else {}
            memory = AssayMemory(
                assay_id=state["assay_id"],
                assay_summary=str(raw.get("assay_summary", "")),
                property_direction=str(raw.get("property_direction", "higher is better")),
                functional_hypotheses=[str(x) for x in raw.get("functional_hypotheses", [])],
                sequence_features=[str(x) for x in raw.get("sequence_features", [])],
            )
        except (ModelError, ValueError, TypeError) as error:
            records.extend(
                _call_records(
                    StructuredResponse(None, getattr(error, "calls", [])),
                    "assay_analyst",
                    state["assay_id"],
                )
            )
            memory = AssayMemory(
                assay_id=state["assay_id"],
                assay_summary=str(state["metadata"].get("ASSAY_DESCRIPTION", "")),
                property_direction="Higher experimental DMS_score means higher fitness.",
                functional_hypotheses=["No model-generated assay hypothesis was available."],
                sequence_features=[],
            )
            records.append(
                {"record_type": "fallback", "stage": "assay_analyst", "error": str(error)}
            )
        reasoning = records + [{"record_type": "assay_memory", **memory.model_dump()}]
        return {
            "assay_memory": memory.model_dump(),
            "usage_records": state.get("usage_records", []) + records,
            "reasoning_records": state.get("reasoning_records", []) + reasoning,
            "status": "assay_analyzed",
        }

    def _reason_batch(self, payload: tuple[GraphState, list[dict[str, Any]], int]):
        state, batch, batch_id = payload
        candidates = [
            {
                "variant_id": item["variant_id"],
                "mutation": item["mutation"],
                "sequence_context": _sequence_context(state["wt_sequence"], item["sequence"]),
            }
            for item in batch
        ]
        user = VARIANT_PROMPT.format(
            assay_memory=_compact_json(state["assay_memory"]),
            candidates=_compact_json(candidates),
        )
        expected = {x["variant_id"]: x for x in batch}
        try:
            response = self.model.generate_json(
                system=VARIANT_SYSTEM,
                user=user,
                run_name=f"variant_reasoner.batch_{batch_id}",
                metadata={"assay": state["assay_id"], "batch": batch_id},
            )
        except (ModelError, ValueError, TypeError) as error:
            fallback = [
                VariantEvidence(
                    variant_id=variant_id,
                    mutation=source["mutation"],
                    direction="uncertain",
                    predicted_score=0.0,
                    mechanisms=["Variant reasoning failed; neutral fallback used."],
                    evidence=[],
                    confidence=0.0,
                    fallback=True,
                )
                for variant_id, source in expected.items()
            ]
            paid = _call_records(
                StructuredResponse(None, getattr(error, "calls", [])),
                "variant_reasoner",
                f"batch_{batch_id}",
            )
            return fallback, paid + [
                {
                    "record_type": "fallback",
                    "stage": "variant_reasoner",
                    "unit": f"batch_{batch_id}",
                    "error": str(error),
                }
            ]
        rows = response.data.get("variants", []) if isinstance(response.data, dict) else []
        parsed: dict[str, VariantEvidence] = {}
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            variant_id = _canonical_id(raw.get("variant_id"), set(expected))
            if variant_id is None or variant_id in parsed:
                continue
            source = expected[variant_id]
            evidence_rows = []
            for index, item in enumerate(raw.get("evidence", [])):
                if not isinstance(item, dict):
                    continue
                try:
                    evidence_rows.append(
                        EvidenceRef(
                            evidence_id=str(
                                item.get("evidence_id") or f"sequence:{variant_id}:{index}"
                            ),
                            evidence_type=item.get("evidence_type", "sequence"),
                            claim=str(item.get("claim", "")),
                        )
                    )
                except ValueError:
                    continue
            try:
                parsed[variant_id] = VariantEvidence(
                    variant_id=variant_id,
                    mutation=source["mutation"],
                    direction=raw.get("direction", "uncertain"),
                    predicted_score=max(-1.0, min(1.0, float(raw.get("predicted_score", 0.0)))),
                    mechanisms=[str(x) for x in raw.get("mechanisms", [])],
                    evidence=evidence_rows,
                    confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
                )
            except (ValueError, TypeError):
                continue
        for variant_id, source in expected.items():
            if variant_id not in parsed:
                parsed[variant_id] = VariantEvidence(
                    variant_id=variant_id,
                    mutation=source["mutation"],
                    direction="uncertain",
                    predicted_score=0.0,
                    mechanisms=["Missing structured model output; neutral fallback used."],
                    evidence=[],
                    confidence=0.0,
                    fallback=True,
                )
        return list(parsed.values()), _call_records(
            response, "variant_reasoner", f"batch_{batch_id}"
        )

    def variant_reasoner(self, state: GraphState) -> dict[str, Any]:
        size = self.config.reasoning_batch_size
        batches = [state["variants"][i : i + size] for i in range(0, len(state["variants"]), size)]
        outputs = _parallel_map(
            self._reason_batch,
            [(state, batch, i) for i, batch in enumerate(batches)],
            self.config.max_concurrency,
        )
        evidence: list[VariantEvidence] = []
        records: list[dict[str, Any]] = []
        for batch_evidence, batch_records in outputs:
            evidence.extend(batch_evidence)
            records.extend(batch_records)
        order = {x["variant_id"]: i for i, x in enumerate(state["variants"])}
        evidence.sort(key=lambda x: order[x.variant_id])
        reasoning = [{"record_type": "variant_evidence", **x.model_dump()} for x in evidence]
        return {
            "variant_evidence": [x.model_dump() for x in evidence],
            "usage_records": state.get("usage_records", []) + records,
            "reasoning_records": state.get("reasoning_records", []) + records + reasoning,
            "status": "variants_analyzed",
        }

    def _rank_group(self, payload: tuple[GraphState, dict[str, Any]]):
        state, group = payload
        evidence_by_id = {x["variant_id"]: x for x in state["variant_evidence"]}
        local_evidence = [evidence_by_id[x] for x in group["variant_ids"]]
        user = RANK_PROMPT.format(
            assay_memory=_compact_json(state["assay_memory"]),
            evidence=_compact_json(local_evidence),
        )
        fallback = False
        response: StructuredResponse | None = None
        unit = f"p{group['partition_id']}.g{group['group_id']}"
        try:
            response = self.model.generate_json(
                system=RANK_SYSTEM,
                user=user,
                run_name=(f"local_ranker.p{group['partition_id']}.g{group['group_id']}"),
                metadata={
                    "assay": state["assay_id"],
                    "partition": group["partition_id"],
                    "group": group["group_id"],
                },
            )
            ranking, rationale = _strict_ranking(response.data, group["variant_ids"])
            records = _call_records(response, "local_ranker", unit)
        except (ModelError, ValueError, TypeError) as error:
            fallback = True
            ranking = sorted(
                group["variant_ids"],
                key=lambda x: (-float(evidence_by_id[x]["predicted_score"]), x),
            )
            rationale = [f"Structured ranking failed; evidence-score fallback: {error}"]
            records = (
                _call_records(response, "local_ranker", unit)
                if response
                else _call_records(
                    StructuredResponse(None, getattr(error, "calls", [])), "local_ranker", unit
                )
            )
            records.append(
                {
                    "record_type": "fallback",
                    "stage": "local_ranker",
                    "unit": unit,
                    "error": str(error),
                }
            )
        local = LocalRanking(
            **group,
            ranking=ranking,
            rationale=rationale,
            fallback=fallback,
        )
        return local, records

    def local_ranker(self, state: GraphState) -> dict[str, Any]:
        groups = make_anchor_groups(
            state["variant_evidence"],
            group_size=self.config.local_group_size,
            anchor_count=self.config.anchor_count,
            repeats=self.config.repeats,
            seed=self.config.grouping_seed
            if self.config.grouping_seed is not None
            else self.config.seed,
        )
        ids = [x["variant_id"] for x in state["variants"]]
        validate_group_coverage(groups, ids, self.config.repeats)
        outputs = _parallel_map(
            self._rank_group,
            [(state, group) for group in groups],
            self.config.max_concurrency,
        )
        rankings: list[LocalRanking] = []
        records: list[dict[str, Any]] = []
        for local, local_records in outputs:
            rankings.append(local)
            records.extend(local_records)
        rankings.sort(key=lambda x: (x.partition_id, x.group_id))
        reasoning = [{"record_type": "local_ranking", **x.model_dump()} for x in rankings]
        return {
            "local_rankings": [x.model_dump() for x in rankings],
            "usage_records": state.get("usage_records", []) + records,
            "reasoning_records": state.get("reasoning_records", []) + records + reasoning,
            "status": "locally_ranked",
        }

    def global_aggregator(self, state: GraphState) -> dict[str, Any]:
        started = time.perf_counter()
        ids = [x["variant_id"] for x in state["variants"]]
        initial = {x["variant_id"]: float(x["predicted_score"]) for x in state["variant_evidence"]}
        scores = aggregate_rankings(
            ids,
            state["local_rankings"],
            initial_scores=initial,
            l2=self.config.l2_regularization,
        )
        uncertainty, budget_scores = estimate_uncertainty(
            ids,
            state["local_rankings"],
            initial_scores=initial,
            repeats=self.config.repeats,
            l2=self.config.l2_regularization,
        )
        budget_scores.setdefault(str(self.config.repeats), scores)
        return {
            "aggregate_scores": scores,
            "uncertainties": uncertainty,
            "budget_scores": budget_scores,
            "status": "globally_aggregated",
            "aggregation_s": time.perf_counter() - started,
        }

    def uncertainty_verifier(self, state: GraphState) -> dict[str, Any]:
        ids = [x["variant_id"] for x in state["variants"]]
        evidence_by_id = {x["variant_id"]: x for x in state["variant_evidence"]}
        confidence = {x: float(evidence_by_id[x]["confidence"]) for x in ids}
        candidates = verification_candidates(
            ids,
            state["aggregate_scores"],
            state["uncertainties"],
            confidence,
            self.config.verification_candidates,
            self.config.local_group_size,
        )
        if len(candidates) < 2:
            return {"status": "verified"}
        local_evidence = [evidence_by_id[x] for x in candidates]
        user = VERIFY_PROMPT.format(
            assay_memory=_compact_json(state["assay_memory"]),
            ordered_ids=_compact_json(candidates),
            evidence=_compact_json(local_evidence),
        )
        records: list[dict[str, Any]] = []
        rankings = list(state["local_rankings"])
        response: StructuredResponse | None = None
        try:
            response = self.model.generate_json(
                system=VERIFY_SYSTEM,
                user=user,
                run_name="uncertainty_verifier",
                metadata={"assay": state["assay_id"], "candidate_count": len(candidates)},
            )
            ranking, rationale = _strict_ranking(response.data, candidates)
            verification = LocalRanking(
                partition_id=-1,
                group_id=0,
                variant_ids=candidates,
                anchor_ids=[],
                ranking=ranking,
                rationale=rationale,
                weight=2.0,
            )
            rankings.append(verification.model_dump())
            records = _call_records(response, "uncertainty_verifier", "uncertain_region")
            reasoning = records + [
                {"record_type": "verification_ranking", **verification.model_dump()}
            ]
        except (ModelError, ValueError, TypeError) as error:
            records = (
                _call_records(response, "uncertainty_verifier", "uncertain_region")
                if response
                else _call_records(
                    StructuredResponse(None, getattr(error, "calls", [])),
                    "uncertainty_verifier",
                    "uncertain_region",
                )
            )
            reasoning = records + [
                {"record_type": "fallback", "stage": "uncertainty_verifier", "error": str(error)}
            ]
        initial = {x: float(evidence_by_id[x]["predicted_score"]) for x in ids}
        started = time.perf_counter()
        verified_scores = aggregate_rankings(
            ids, rankings, initial_scores=initial, l2=self.config.l2_regularization
        )
        budgets = dict(state["budget_scores"])
        budgets["verified"] = verified_scores
        return {
            "local_rankings": rankings,
            "aggregate_scores": verified_scores,
            "budget_scores": budgets,
            "usage_records": state.get("usage_records", []) + records,
            "reasoning_records": state.get("reasoning_records", []) + reasoning,
            "status": "verified",
            "verification_aggregation_s": time.perf_counter() - started,
        }

    def final_ranking(self, state: GraphState) -> dict[str, Any]:
        evidence_by_id = {x["variant_id"]: x for x in state["variant_evidence"]}
        variants_by_id = {x["variant_id"]: x for x in state["variants"]}
        ordered = sorted(
            variants_by_id,
            key=lambda x: (-float(state["aggregate_scores"][x]), x),
        )
        provenance: dict[str, list[str]] = defaultdict(list)
        for ranking in state["local_rankings"]:
            label = f"p{ranking['partition_id']}.g{ranking['group_id']}"
            for variant_id in ranking["variant_ids"]:
                provenance[variant_id].append(label)
        predictions: list[PredictionResult] = []
        for rank, variant_id in enumerate(ordered, start=1):
            evidence = evidence_by_id[variant_id]
            uncertainty = float(state["uncertainties"].get(variant_id, 0.0))
            confidence = 0.5 * float(evidence["confidence"]) + 0.5 * (1.0 - uncertainty)
            predictions.append(
                PredictionResult(
                    variant_id=variant_id,
                    mutation=variants_by_id[variant_id]["mutation"],
                    global_score=float(state["aggregate_scores"][variant_id]),
                    rank=rank,
                    confidence=max(0.0, min(1.0, confidence)),
                    uncertainty=max(0.0, min(1.0, uncertainty)),
                    evidence_ids=[x["evidence_id"] for x in evidence.get("evidence", [])],
                    provenance=provenance[variant_id],
                    experimental_score=variants_by_id[variant_id].get("experimental_score"),
                )
            )
        return {
            "predictions": [x.model_dump() for x in predictions],
            "status": "complete",
        }

    def compile(self, checkpointer=None):
        builder = StateGraph(GraphState)
        builder.add_node("load_assay", self.load_assay)
        builder.add_node("assay_analyst", self.assay_analyst)
        builder.add_node("variant_reasoner", self.variant_reasoner)
        builder.add_node("local_ranker", self.local_ranker)
        builder.add_node("global_aggregator", self.global_aggregator)
        builder.add_node("uncertainty_verifier", self.uncertainty_verifier)
        builder.add_node("final_ranking", self.final_ranking)
        builder.add_edge(START, "load_assay")
        builder.add_edge("load_assay", "assay_analyst")
        builder.add_edge("assay_analyst", "variant_reasoner")
        builder.add_edge("variant_reasoner", "local_ranker")
        builder.add_edge("local_ranker", "global_aggregator")
        builder.add_edge(
            "global_aggregator",
            "uncertainty_verifier" if self.config.verification_candidates >= 2 else "final_ranking",
        )
        builder.add_edge("uncertainty_verifier", "final_ranking")
        builder.add_edge("final_ranking", END)
        return builder.compile(checkpointer=checkpointer)
