import pytest

from evagent.compact import protocol
from evagent.config import PipelineConfig
from evagent.graph import EVAgentWorkflow
from evagent.ranking import make_anchor_groups
from evagent.runtime import ExecutionPaused, RequestStore


def test_compact_matrix():
    source = {
        "assays": [],
        "matrix": [
            {"assay": str(i), "size": n, "seed": s, "eligible": i < 68 or n < 250}
            for s in (1, 2, 3)
            for n in (250, 100, 50)
            for i in range(70)
        ],
    }
    result = protocol(source)
    assert sum(j["eligible"] for j in result["matrix"]) == 138
    assert len(result["matrix"]) == 140
    assert {j["size"] for j in result["matrix"]} == {50, 250}
    assert {j["seed"] for j in result["matrix"]} == {1}
    assert result == protocol(source)


def test_first_partition_is_identical():
    evidence = [{"variant_id": str(i), "predicted_score": i % 7} for i in range(250)]
    kwargs = dict(group_size=32, anchor_count=8, seed=1)
    one = make_anchor_groups(evidence, repeats=1, **kwargs)
    three = make_anchor_groups(evidence, repeats=3, **kwargs)
    assert one == [g for g in three if g["partition_id"] == 0]
    assert len(one) == 11


def test_no_verifier_graph_edge():
    graph = EVAgentWorkflow(None, PipelineConfig(repeats=1, verification_candidates=0)).compile()
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("global_aggregator", "final_ranking") in edges
    assert ("global_aggregator", "uncertainty_verifier") not in edges


def test_pause_and_resume_cache(tmp_path):
    store = RequestStore(tmp_path / "requests.sqlite", 10)
    raw = {"usage": {"totalTokenCount": 4}}
    store.invoke("first", lambda: raw)
    (tmp_path / "PAUSE").touch()
    with pytest.raises(ExecutionPaused):
        store.invoke("second", lambda: raw)
    assert store.invoke("first", lambda: pytest.fail("cache billed again"))["cache_hit"]
    (tmp_path / "PAUSE").unlink()
    store.invoke("second", lambda: raw)
    store.invoke("third", lambda: raw)
    assert store.summary()["reported_tokens"] == 12
    with pytest.raises(ExecutionPaused):
        store.invoke("fourth", lambda: raw)
