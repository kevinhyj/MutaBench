import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import check_grad

from evagent import acr_seq as acr
from evagent import runtime
from evagent.config import ModelConfig


def test_nested_groups_and_cache():
    ids = [f"M{i:03d}" for i in range(1, 501)]
    anchors = ["M002", "M011", "M028", "M050"]
    all_groups = [acr.groups("test", ids[:n], anchors) for n in acr.SIZES]
    assert [len(g) for g in all_groups] == [3, 5, 12, 24]
    assert len({g["group_hash"] for groups in all_groups for g in groups}) == 27
    for n, gs in zip(acr.SIZES, all_groups):
        assert set().union(*(set(g["variant_ids"]) for g in gs)) == set(ids[:n])
        assert all(len(g["variant_ids"]) <= 25 and set(anchors) <= set(g["variant_ids"]) for g in gs)
        assert gs == acr.groups("test", ids[:n], anchors)
    assert all_groups[0][0] == all_groups[-1][0]


def test_safe_complete_prompt():
    sequence = "ACGU" * 10000
    text = acr.prompt({"DMS_ID": "test", "fitness": "SECRET_LABEL", "score": 123}, "ACGU", [
        {"variant_id": "M001", "mutation": "A1C", "sequence": sequence,
         "experimental_score": "SECRET_LABEL"}])
    assert sequence in text
    assert "SECRET_LABEL" not in text
    assert "experimental_score" not in text
    assert "predicted_score" not in text


def test_parser_and_empty_classification():
    assert acr.validate_answer('```json\n{"ranking":["M1","M002",],}\n```', ["M001", "M002"])["ranking"] == ["M001", "M002"]
    for text in ('{"ranking":["M001"]}', '{"ranking":["M001","M001"]}',
                 '{"ranking":["M001","M003"]}', ''):
        with pytest.raises(ValueError):
            acr.validate_answer(text, ["M001", "M002"])
    assert acr.classify({"provider_raw": {"usageMetadata": {"totalTokenCount": 20}}}) == "provider_unclassified"
    assert acr.classify({"provider_raw": {"promptFeedback": {"blockReason": "SAFETY"}}}) == "provider_blocked"
    assert acr.classify({"provider_raw": {"candidates": [{"finishReason": "STOP"}]}}) == "response_received"


def test_wire_matches_combined_input():
    wire = acr.wire_request(ModelConfig(), "system", "user")
    assert wire["contents"] == [{"parts": [{"text": "system\n\nuser"}]}]
    assert "systemInstruction" not in wire
    assert wire["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "medium"


def test_pl_connected_deterministic_gradient():
    ids = ["a", "b", "c", "d"]
    ranks = [{"ranking": ["a", "b", "c"]}, {"ranking": ["a", "c", "d"]}]
    scores, diagnostics = acr.fit(ids, ranks)
    assert diagnostics["success"]
    assert scores == acr.fit(ids, ranks)[0]
    assert scores["a"] > scores["b"] > scores["c"] > scores["d"]
    with pytest.raises(ValueError, match="Disconnected"):
        acr.fit(ids, [{"ranking": ["a", "b"]}, {"ranking": ["c", "d"]}])
    order = [(np.array([0, 2, 1, 3]), 1.)]
    assert check_grad(lambda x: acr._pl_objective(x, order, .001)[0],
                      lambda x: acr._pl_objective(x, order, .001)[1], np.arange(4.)/3) < 1e-5


def test_ledger_budget_cached_and_resume(tmp_path):
    store = acr.Ledger(tmp_path / "requests.sqlite", 1000, tmp_path, 100)
    raw = {"answer": "{}", "provider_raw": {"candidates": [{"finishReason": "STOP"}]},
           "usage": {"totalTokenCount": 100}}
    assert store.invoke("one", lambda: raw)["cache_hit"] is False
    assert store.own_tokens() == 100
    assert store.invoke("one", lambda: pytest.fail("cached"))["cache_hit"] is True
    with pytest.raises(runtime.ExecutionPaused):
        store.invoke("two", lambda: raw)
    store.db.close()
    restarted = acr.Ledger(tmp_path / "requests.sqlite", 1000, tmp_path, 100)
    with pytest.raises(runtime.ExecutionPaused):
        restarted.invoke("two", lambda: raw)


def test_ambiguous_accounted_and_durable_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(acr.time, "sleep", lambda _: None)
    store = acr.Ledger(tmp_path / "requests.sqlite", 10000, tmp_path, 10000)
    raw = {"answer": "", "provider_raw": {}, "usage": {"totalTokenCount": 12}}
    with pytest.raises(acr.old.TerminalRequest, match="retry_exhausted"):
        store.invoke("empty", lambda: raw)
    assert store.own_tokens() == 60
    store.db.close()
    store = acr.Ledger(tmp_path / "requests.sqlite", 10000, tmp_path, 10000)
    with pytest.raises(acr.old.TerminalRequest):
        store.invoke("empty", lambda: pytest.fail("must not reset ceiling"))


def test_pause_and_token_limit(tmp_path):
    store = acr.Ledger(tmp_path / "requests.sqlite", 1000, tmp_path, 100)
    (tmp_path / "PAUSE").touch()
    with pytest.raises(runtime.ExecutionPaused):
        store.check_admission()
    (tmp_path / "PAUSE").unlink()
    config = SimpleNamespace(model=ModelConfig(), rg_llm_root=Path("/unused"))
    service = acr.Service(config, tmp_path, store)
    service.limit = 10
    service.metadata_request = lambda *args: {"totalTokens": 11}
    with pytest.raises(acr.old.TerminalRequest, match="context_limit"):
        service.count("s", "u")


def test_mock_cell(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "output"
    root.mkdir()
    variants = [{"variant_id": f"M{i:03d}", "mutation": f"A{i}C", "sequence": "ACGU",
                 "experimental_score": -float(i)} for i in range(1, 51)]
    anchors = [v["variant_id"] for v in variants[:4]]
    acr.old.save(root / "anchors/random/mock.json", anchors)
    acr.old.save(root / "groups/random/mock/n50.json", acr.groups("mock", [v["variant_id"] for v in variants], anchors))
    exp = object.__new__(acr.Experiment)
    exp.root, exp.source = root, source
    exp.config = SimpleNamespace(model=ModelConfig())
    exp.metadata = {"mock": {"RAW_CONSTRUCT_SEQ": "ACGU"}}
    exp.variants = lambda a, n: variants
    def request(folder, user, metadata):
        candidates = json.loads(user.split("(not truncated):\n")[1])
        return {"answer": json.dumps({"ranking": sorted(v["variant_id"] for v in candidates)}),
                "provider_raw": {"candidates": [{"finishReason": "STOP"}]},
                "usage": {"totalTokenCount": 10}}
    exp.service = SimpleNamespace(request=request)
    exp.cell("mock", 50, "random")
    r = acr.read(root / "evaluations/random/mock/n50/result.json")
    assert r["status"] == "complete" and not r["fallback"]
    # All four anchors are deliberately at one extreme: perfect local ranks do
    # not identify cross-group order inside the unanchored interval.
    assert np.isfinite(r["metrics"]["spearman"])
    assert set(r["scores"]) == {v["variant_id"] for v in variants}
