import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import logsumexp

from evagent.experiments import audit_baseline, correlation, mean_local_rank
from evagent.graph import model_metadata
from evagent.ranking import _pl_objective
from evagent.runtime import ExecutionPaused, RequestStore


def test_vectorized_pl_matches_reference_and_gradient():
    rng = np.random.default_rng(42)
    orders = [(rng.choice(25, 12, replace=False), 1.3), (rng.choice(25, 9, replace=False), 2.0)]
    for scale in (1, 100):
        x = rng.normal(size=25) * scale
        loss, grad = 0.5 * 0.001 * (x @ x), 0.001 * x.copy()
        for order, weight in orders:
            for i in range(len(order) - 1):
                rest = order[i:]
                z = logsumexp(x[rest])
                loss += weight * (z - x[order[i]])
                grad[rest] += weight * np.exp(x[rest] - z)
                grad[order[i]] -= weight
        actual_loss, actual_grad = _pl_objective(x, orders, 0.001)
        assert actual_loss == pytest.approx(loss, rel=1e-10)
        np.testing.assert_allclose(actual_grad, grad, atol=1e-10)
    x = rng.normal(size=25)
    assert (
        check_grad(
            lambda a: _pl_objective(a, orders, 0.001)[0],
            lambda a: _pl_objective(a, orders, 0.001)[1],
            x,
        )
        < 1e-4
    )


def test_prior_ties_preserved_and_unknown_ids_rejected():
    assert correlation({"A": 0, "B": 0, "C": 1}, {"A": 1, "B": 2, "C": 3})[
        "spearman"
    ] == pytest.approx(np.sqrt(0.75))
    audit = audit_baseline(
        {"ids": ["M01", "M02"], "response_text": '{"ranking":["M01","M999","M02"]}'}
    )
    assert not audit["valid"] and audit["unknown"] == ["M999"]
    assert audit_baseline({"ids": ["M01", "M02"], "response_text": '{"ranking":["M001","M002",]}'})[
        "valid"
    ]
    assert not audit_baseline(
        {"ids": ["M01", "M02"], "response_text": '{"ranking":["M01","M01"]}'}
    )["valid"]


def test_request_budget_durable_cache_and_network_pause(tmp_path):
    path = tmp_path / "calls.sqlite"
    store = RequestStore(path, 10)
    calls = []

    def paid():
        calls.append(1)
        return {"answer": "{}", "usage": {"totalTokenCount": 12}}

    store.invoke("a", paid)
    assert store.invoke("a", paid)["cache_hit"]
    with pytest.raises(ExecutionPaused):
        store.invoke("b", paid)
    store.db.close()
    resumed = RequestStore(path, 10)
    assert resumed.cached("a")["answer"] == "{}" and len(calls) == 1
    with pytest.raises(ExecutionPaused):
        resumed.invoke("b", paid)
    broken = RequestStore(tmp_path / "broken.sqlite", 100)

    def fail():
        raise ConnectionError("private header must not be persisted")

    with pytest.raises(ExecutionPaused):
        broken.invoke("c", fail)
    assert broken.stop.is_set()
    assert broken.summary()["requests_by_status"]["failed_unknown_cost"] == 1


def test_metadata_whitelist_and_local_rank_coverage():
    assert model_metadata(
        {"DMS_ID": "x", "WT_DMS_SCORE": 12345, "experimental_score": 6789, "Notes": "answer leak"}
    ) == {"DMS_ID": "x"}
    assert mean_local_rank(["A", "B"], [{"ranking": ["A", "B"]}]) == {"A": 1.0, "B": 0.0}
    with pytest.raises(ValueError):
        mean_local_rank(["A", "B", "C"], [{"ranking": ["A", "B"]}])


def test_small_assay_sampling_is_nested_and_does_not_require_250(tmp_path, monkeypatch):
    import sys
    import types

    import pandas as pd

    from evagent.config import AppConfig
    from evagent.main import nested_subsample_split

    frame = pd.DataFrame(
        {
            "mutant": [f"A{i}G" for i in range(120)],
            "sequence": [f"SEQ{i}" for i in range(120)],
            "DMS_score": list(range(120)),
        }
    )
    module = types.ModuleType("rg_llm.dataset")
    module.load_assay = lambda *args: frame
    monkeypatch.setitem(sys.modules, "rg_llm.dataset", module)
    config = AppConfig(rg_llm_root=tmp_path, data_root=tmp_path)
    a = nested_subsample_split(config, "small", 50, 1, tmp_path / "frozen")
    b = nested_subsample_split(config, "small", 100, 1, tmp_path / "frozen")
    assert a == b[:50]
    assert a == nested_subsample_split(config, "small", 50, 1, tmp_path / "frozen")
    with pytest.raises(ValueError):
        nested_subsample_split(config, "small", 250, 1, tmp_path / "frozen")


def test_infrastructure_pause_is_not_a_scientific_fallback():
    from evagent.config import PipelineConfig
    from evagent.graph import EVAgentWorkflow

    class BrokenModel:
        def generate_json(self, **kwargs):
            raise ExecutionPaused("Network unavailable")

    workflow = EVAgentWorkflow(BrokenModel(), PipelineConfig())
    with pytest.raises(ExecutionPaused):
        workflow.assay_analyst({"assay_id": "test", "metadata": {}, "wt_sequence": "ACGU"})


def test_malformed_output_keeps_paid_call_records(monkeypatch, tmp_path):
    from evagent import model
    from evagent.config import ModelConfig

    monkeypatch.setattr(
        model,
        "_gemini_generate",
        lambda **kwargs: {"answer": "invalid", "usage": {"totalTokenCount": 7}, "status": "ok"},
    )
    adapter = model.GeminiAdapter(ModelConfig(), tmp_path, schema_retries=1)
    with pytest.raises(model.ModelError) as caught:
        adapter.generate_json(system="s", user="u", run_name="test")
    assert len(caught.value.calls) == 2
    assert sum(c.usage["totalTokenCount"] for c in caught.value.calls) == 14
