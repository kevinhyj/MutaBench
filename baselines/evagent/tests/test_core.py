from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from evagent.config import ModelConfig, PipelineConfig
from evagent.graph import EVAgentWorkflow
from evagent.main import _budget_summaries, _nested_variant_order
from evagent.model import GeminiAdapter, StructuredResponse, extract_json
from evagent.ranking import (
    aggregate_rankings,
    estimate_uncertainty,
    make_anchor_groups,
    validate_group_coverage,
)
from evagent.schemas import ModelCallRecord


def test_extract_json_repairs_common_wrappers():
    assert extract_json('Reasoning...\n```json\n{"ranking":["M001","M002",],}\n```') == {
        "ranking": ["M001", "M002"]
    }


def _evidence(n: int):
    return [{"variant_id": f"M{i + 1:03d}", "predicted_score": (i - n / 2) / n} for i in range(n)]


def test_anchor_groups_cover_every_partition_and_are_deterministic():
    evidence = _evidence(73)
    kwargs = dict(group_size=16, anchor_count=4, repeats=3, seed=1)
    first = make_anchor_groups(evidence, **kwargs)
    second = make_anchor_groups(evidence, **kwargs)
    assert first == second
    assert all(len(x["variant_ids"]) <= 16 for x in first)
    validate_group_coverage(first, [x["variant_id"] for x in evidence], 3)
    anchors = {tuple(x["anchor_ids"]) for x in first}
    assert len(anchors) == 1


def test_plackett_luce_recovers_consistent_order():
    ids = ["A", "B", "C", "D"]
    rankings = [
        {"ranking": ["A", "B", "C"], "weight": 1},
        {"ranking": ["A", "B", "D"], "weight": 1},
        {"ranking": ["B", "C", "D"], "weight": 1},
    ]
    first = aggregate_rankings(ids, rankings)
    second = aggregate_rankings(ids, rankings)
    assert sorted(ids, key=lambda x: -first[x]) == ["A", "B", "C", "D"]
    assert first == pytest.approx(second)


def test_nested_order_preserves_frozen_prefix():
    rows = [(f"A{i}G", f"SEQ{i}", float(i)) for i in range(12)]
    prefix = [rows[4], rows[1], rows[9]]
    first = _nested_variant_order(rows, prefix, assay="mock", seed=1)
    second = _nested_variant_order(rows, prefix, assay="mock", seed=1)
    assert first == second
    assert first[:3] == prefix
    assert len(first) == len(set(first)) == len(rows)


def test_all_cumulative_budgets_and_costs_are_reported():
    ids = ["A", "B", "C"]
    rankings = [
        {"partition_id": 0, "ranking": ["A", "B", "C"]},
        {"partition_id": 1, "ranking": ["A", "C", "B"]},
        {"partition_id": 2, "ranking": ["B", "A", "C"]},
    ]
    _, budgets = estimate_uncertainty(
        ids, rankings, initial_scores={x: 0.0 for x in ids}, repeats=3, l2=1e-3
    )
    assert set(budgets) == {"1", "2", "3"}
    records = [
        {"record_type": "model_call", "stage": "assay_analyst", "usage": {}},
        {"record_type": "model_call", "stage": "variant_reasoner", "usage": {}},
        *[
            {
                "record_type": "model_call",
                "stage": "local_ranker",
                "unit": f"p{i}.g0",
                "usage": {},
            }
            for i in range(3)
        ],
        {"record_type": "model_call", "stage": "uncertainty_verifier", "usage": {}},
    ]
    usage, _ = _budget_summaries(records, 3)
    assert usage["1"]["model_calls"] == 3
    assert usage["2"]["model_calls"] == 4
    assert usage["3"]["model_calls"] == 5
    assert usage["verified"]["model_calls"] == 6


class FakeModel:
    def generate_json(self, *, system, user, run_name, metadata=None):
        del system, metadata
        if run_name == "assay_analyst":
            data = {
                "assay_summary": "mock assay",
                "property_direction": "higher is better",
                "functional_hypotheses": ["sequence changes alter function"],
                "sequence_features": ["short synthetic RNA"],
            }
        elif run_name.startswith("variant_reasoner"):
            candidates = json.loads(
                user.split("Candidates with local WT/mutant sequence context:\n", 1)[1]
            )
            data = {
                "variants": [
                    {
                        "variant_id": item["variant_id"],
                        "mutation": item["mutation"],
                        "direction": "beneficial",
                        "predicted_score": int(item["variant_id"][1:]) / 100,
                        "mechanisms": ["mock mechanism"],
                        "evidence": [
                            {
                                "evidence_id": f"sequence:{item['variant_id']}",
                                "evidence_type": "sequence",
                                "claim": "mock evidence",
                            }
                        ],
                        "confidence": 0.8,
                    }
                    for item in candidates
                ]
            }
        else:
            ids = sorted(set(__import__("re").findall(r"M\d{3}", user)), reverse=True)
            data = {"ranking": ids, "rationale": ["mock ordering"]}
        call = ModelCallRecord(run_name=run_name, answer=json.dumps(data), usage={})
        return StructuredResponse(data=data, calls=[call])


class InvalidRankingModel(FakeModel):
    def generate_json(self, *, system, user, run_name, metadata=None):
        if run_name.startswith("local_ranker"):
            data = {"ranking": ["M001", "M001"], "rationale": []}
            call = ModelCallRecord(run_name=run_name, answer=json.dumps(data), usage={})
            return StructuredResponse(data=data, calls=[call])
        return super().generate_json(system=system, user=user, run_name=run_name, metadata=metadata)


def test_invalid_ranking_counts_paid_call_before_fallback():
    workflow = EVAgentWorkflow(
        InvalidRankingModel(),
        PipelineConfig(local_group_size=4, anchor_count=1, repeats=1),
    )
    state = {
        "assay_id": "mock",
        "assay_memory": {"assay_summary": "mock"},
        "variant_evidence": [
            {
                "variant_id": f"M{i:03d}",
                "mutation": f"A{i}G",
                "predicted_score": float(i),
            }
            for i in [1, 2]
        ],
    }
    ranking, records = workflow._rank_group(
        (
            state,
            {
                "partition_id": 0,
                "group_id": 0,
                "anchor_ids": [],
                "variant_ids": ["M001", "M002"],
            },
        )
    )
    assert ranking.fallback is True
    assert [x["record_type"] for x in records] == ["model_call", "fallback"]


@pytest.mark.parametrize("verify_count", [0, 3])
def test_mock_graph_produces_strict_global_ranking(verify_count):
    config = PipelineConfig(
        reasoning_batch_size=4,
        local_group_size=6,
        anchor_count=2,
        repeats=2,
        max_concurrency=2,
        verification_candidates=verify_count,
    )
    graph = EVAgentWorkflow(FakeModel(), config).compile(MemorySaver())
    variants = [
        {
            "variant_id": f"M{i + 1:03d}",
            "mutation": f"A{i + 1}G",
            "sequence": "A" * i + "G" + "A" * (9 - i),
            "experimental_score": float(i),
        }
        for i in range(10)
    ]
    result = graph.invoke(
        {
            "assay_id": "mock",
            "metadata": {"ASSAY_DESCRIPTION": "mock"},
            "wt_sequence": "A" * 10,
            "variants": variants,
            "trace_id": "trace",
            "run_id": "run",
            "status": "initialized",
        },
        {"configurable": {"thread_id": "test"}},
    )
    ids = [x["variant_id"] for x in result["predictions"]]
    assert result["status"] == "complete"
    assert len(ids) == len(set(ids)) == 10
    assert set(ids) == {x["variant_id"] for x in variants}
    assert set(result["budget_scores"]) == ({"1", "2", "verified"} if verify_count else {"1", "2"})
    if not verify_count:
        assert not any(
            r.get("stage") == "uncertainty_verifier" for r in result["reasoning_records"]
        )


def test_sqlite_checkpoint_persists_completed_graph(tmp_path):
    config = PipelineConfig(
        reasoning_batch_size=4,
        local_group_size=4,
        anchor_count=1,
        repeats=1,
        max_concurrency=1,
        verification_candidates=2,
    )
    variants = [
        {
            "variant_id": f"M{i + 1:03d}",
            "mutation": f"A{i + 1}G",
            "sequence": "A" * i + "G" + "A" * (3 - i),
            "experimental_score": float(i),
        }
        for i in range(4)
    ]
    connection = sqlite3.connect(tmp_path / "checkpoint.sqlite", check_same_thread=False)
    try:
        graph = EVAgentWorkflow(FakeModel(), config).compile(SqliteSaver(connection))
        graph_config = {"configurable": {"thread_id": "sqlite-test"}}
        result = graph.invoke(
            {
                "assay_id": "mock",
                "metadata": {"ASSAY_DESCRIPTION": "mock"},
                "wt_sequence": "AAAA",
                "variants": variants,
                "trace_id": "trace",
                "run_id": "run",
                "status": "initialized",
            },
            graph_config,
        )
        snapshot = graph.get_state(graph_config)
    finally:
        connection.close()
    assert result["status"] == "complete"
    assert snapshot.values["status"] == "complete"
    assert snapshot.next == ()


@pytest.mark.live
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")
def test_live_gemini_smoke():
    root = Path(os.getenv("RG_LLM_ROOT", "/data/yanjie_huang/RG-LLM"))
    model = GeminiAdapter(ModelConfig(), root, schema_retries=0)
    response = model.generate_json(
        system="Return JSON only.",
        user='Return exactly {"status":"ok"}.',
        run_name="pytest_smoke",
    )
    assert response.data == {"status": "ok"}


def test_model_config_allows_explicit_direct_connection(tmp_path, monkeypatch):
    config_path = tmp_path / "gemini.json"
    config_path.write_text("{}")
    monkeypatch.setenv("EVAGENT_PROXY_URL", "direct")

    assert ModelConfig.from_rg_llm(config_path).proxy is None
