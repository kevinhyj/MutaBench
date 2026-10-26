"""Serializable contracts shared by graph nodes and output artifacts."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VariantInput(StrictModel):
    variant_id: str
    mutation: str
    sequence: str
    experimental_score: float | None = None


class EvidenceRef(StrictModel):
    evidence_id: str
    evidence_type: Literal["sequence", "structure", "conservation", "literature", "assay"]
    claim: str


class AssayMemory(StrictModel):
    assay_id: str
    assay_summary: str
    property_direction: str
    functional_hypotheses: list[str] = Field(default_factory=list)
    sequence_features: list[str] = Field(default_factory=list)


class VariantEvidence(StrictModel):
    variant_id: str
    mutation: str
    direction: Literal["beneficial", "neutral", "deleterious", "uncertain"]
    predicted_score: float = Field(ge=-1.0, le=1.0)
    mechanisms: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    fallback: bool = False


class LocalRanking(StrictModel):
    partition_id: int
    group_id: int
    variant_ids: list[str]
    anchor_ids: list[str] = Field(default_factory=list)
    ranking: list[str]
    rationale: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0.0)
    fallback: bool = False

    @field_validator("ranking")
    @classmethod
    def ranking_has_no_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("ranking contains duplicates")
        return value


class PredictionResult(StrictModel):
    variant_id: str
    mutation: str
    global_score: float
    rank: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    experimental_score: float | None = None


class ModelCallRecord(StrictModel):
    run_name: str
    answer: str
    thought_summary: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    elapsed_s: float = 0.0
    attempts: int = 1
    transport_reconnects: int = 0
    model_version: str = ""
    finish_reason: str = ""
    queue_s: float = 0.0
    request_sha256: str = ""
    cache_hit: bool = False


class GraphState(TypedDict, total=False):
    aggregation_s: float
    verification_aggregation_s: float
    assay_id: str
    metadata: dict[str, Any]
    wt_sequence: str
    variants: list[dict[str, Any]]
    assay_memory: dict[str, Any]
    variant_evidence: list[dict[str, Any]]
    local_rankings: list[dict[str, Any]]
    aggregate_scores: dict[str, float]
    uncertainties: dict[str, float]
    budget_scores: dict[str, dict[str, float]]
    predictions: list[dict[str, Any]]
    usage_records: list[dict[str, Any]]
    reasoning_records: list[dict[str, Any]]
    trace_id: str
    run_id: str
    status: str
