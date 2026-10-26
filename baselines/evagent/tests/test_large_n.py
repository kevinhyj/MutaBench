import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from evagent import large_n as mod
from evagent.config import AppConfig
from evagent.fullseq import validate_evidence
from evagent.runtime import ExecutionPaused


def variants(n=25):
    return [
        {
            "variant_id": f"M{i + 1:03d}",
            "mutation": f"A{i + 1}G",
            "sequence": "A" * 6000 + "G",
            "experimental_score": float(i),
        }
        for i in range(n)
    ]


def evidence(rows):
    return [
        {
            "variant_id": v["variant_id"],
            "mutation": v["mutation"],
            "direction": "neutral",
            "predicted_score": 0.2,
            "confidence": 0.5,
            "mechanisms": [],
            "evidence": [],
        }
        for v in rows
    ]


def test_unwrap_is_lossless_and_does_not_invent_predictions():
    rows = variants()
    raw = evidence(rows)
    nested = json.dumps({"response": {"data": raw}})
    parsed = validate_evidence(mod.unwrap(nested, True), rows)
    assert [v["predicted_score"] for v in parsed] == [0.2] * 25
    assert raw == evidence(rows)
    for broken in (raw[:-1], [*raw[:-1], raw[0]], [{**raw[0], "predicted_score": None}, *raw[1:]]):
        with pytest.raises(ValueError):
            validate_evidence(mod.unwrap(json.dumps({"data": broken}), True), rows)
    with pytest.raises(ValueError):
        mod.unwrap("")


def test_memory_identity_and_empty_summary():
    raw = {"assay_summary": "RNA assay", "property_direction": "higher is better"}
    assert mod.validate_memory(raw, "a")["assay_id"] == "a"
    with pytest.raises(ValueError):
        mod.validate_memory({**raw, "assay_id": "b"}, "a")
    with pytest.raises(ValueError):
        mod.validate_memory({**raw, "assay_summary": " "}, "a")


def test_explicit_six_size_sampling_preserves_every_old_prefix(tmp_path, monkeypatch):
    rows = [(f"m{i}", "AGCU", float(i)) for i in range(2000)]
    monkeypatch.setattr(mod, "nested_subsample_split", lambda *args: rows)
    root, origin = tmp_path / "new", tmp_path / "old"
    for n in (50, 100, 250, 500):
        mod.save(
            origin / "samples/a" / f"n{n}_b1.json",
            {"variants": [{"id": m, "seq": s, "score": v} for m, s, v in rows[:n]]},
        )
    hashes = mod.extend_samples(None, "a", mod.SIZES, root, origin)
    assert len(hashes) == 6
    for n in mod.SIZES:
        assert len(mod.read(root / "a" / f"n{n}_b1.json")["variants"]) == n
    assert hashes == mod.extend_samples(None, "a", mod.SIZES, root, origin)
    changed = rows.copy()
    changed[0], changed[1] = changed[1], changed[0]
    monkeypatch.setattr(mod, "nested_subsample_split", lambda *args: changed)
    with pytest.raises(ExecutionPaused):
        mod.extend_samples(None, "a", mod.SIZES, root, origin)


def test_generation_hash_ignores_only_transport():
    a = AppConfig().model.public_dict()
    b = {**a, "timeout_s": 300, "retries": 1, "proxy": "different"}
    assert mod.signature(a, "system", "user") == mod.signature(b, "system", "user")
    assert mod.signature(a, "system", "user") != mod.signature(
        {**b, "thinking_level": "high"}, "system", "user"
    )
    assert mod.signature(a, "system", "user") != mod.signature(b, "system", "other")


def test_regeneration_is_one_durable_semantic_slot(tmp_path):
    config = AppConfig()
    paid = []
    service = mod.Service(config, tmp_path, SimpleNamespace(check_admission=lambda: None))
    service.count = lambda *args: 10
    service.adapter = SimpleNamespace(
        _call=lambda *args: (
            paid.append(1)
            or SimpleNamespace(model_dump=lambda: {"answer": "", "usage": {"totalTokenCount": 10}})
        )
    )

    def validator(value):
        return validate_evidence(value, variants())

    first = service.predict(
        tmp_path / "repair", "system", "user", "CR.repair", {}, validator, repair_once=True
    )
    second = service.predict(
        tmp_path / "repair", "system", "user", "CR.repair", {}, validator, repair_once=True
    )
    assert first["status"] == second["status"] == "invalid_output"
    assert len(paid) == 1
    with pytest.raises(ExecutionPaused):
        service.predict(
            tmp_path / "repair", "system", "changed", "CR.repair", {}, validator, repair_once=True
        )


def test_import_checks_dependency_and_does_not_modify_source(tmp_path):
    run = mod.Experiment.__new__(mod.Experiment)
    run.config = AppConfig()
    run.origin = tmp_path / "old"
    source = run.origin / "result.json"
    rows = variants()
    raw = evidence(rows)
    mod.save(source, {"status": "invalid_output", "calls": [{"answer": json.dumps({"data": raw})}]})
    before = source.read_bytes()
    request = run.origin / "request.json"
    mod.save(request, {"model": run.config.model.public_dict(), "system": "s", "user": "u"})
    run.manifest = {"source_results": {"result.json": mod.sha(source)}}
    value = run.import_result(
        tmp_path / "new/result.json",
        source,
        request,
        "s",
        "u",
        lambda d: validate_evidence(d, rows),
        variant=True,
    )
    assert value["origin"] == "offline_parse_repair"
    assert source.read_bytes() == before
    with pytest.raises(ExecutionPaused):
        run.import_result(
            tmp_path / "different/result.json",
            source,
            request,
            "s",
            "different memory",
            lambda d: d,
        )


def fake_run(tmp_path):
    run = mod.Experiment.__new__(mod.Experiment)
    run.root, run.origin = tmp_path / "new", tmp_path / "old"
    run.root.mkdir()
    run.origin.mkdir()
    run.started, run.phase = time.time(), "test"
    run.config = AppConfig()
    run.source = {"assays": [{"assay": "a", "n_valid": 2000}]}
    run.classes = {"a": {"rna_type": "mock", "protein_stability_source": False}}
    run.metadata = {"a": {"RAW_CONSTRUCT_SEQ": "AAAA"}}
    run.manifest = {"eligible_cells": 6, "large_cells": 2, "samples": {"a": {"50": "sample"}}}
    run.store = SimpleNamespace(accounting=lambda: {"reported_tokens": 0})
    mod.save(
        run.root / "queue_manifest.json",
        [{"assay": "a", "N": n, "eligible": True} for n in mod.SIZES],
    )
    return run


def test_failed_dependency_skips_new_chunks_but_global_runs(tmp_path):
    run = fake_run(tmp_path)
    rows = variants(50)
    run.variants = lambda *args: rows
    run.memory = lambda *args: {"status": "complete", "data": {}, "calls": []}
    calls = []

    def global_call(assay, n, rows):
        calls.append("Global")
        result = {"status": "invalid_output", "calls": [], "ranking": [], "returned_ranking": []}
        mod.save(run.root / "assays/a/evaluations/n50/global.json", result)
        return result

    def chunk(assay, memory, rows, index, offline=False):
        assert offline, "must not pay for chunks behind a failed dependency"
        return {"status": "invalid_output"} if index == 0 else None

    run.global_ranking, run.chunk = global_call, chunk
    run.cell("a", 50)
    assert calls == ["Global"]
    e = mod.read(run.root / "assays/a/evaluations/n50/evaluation.json")
    assert e["terminal"] and e["cr_status"] == "dependency_failed"
    progress = run.report()
    assert progress["terminal_cells"] == 1 and progress["pending_cells"] == 5


def test_infrastructure_is_not_scientific_completion(tmp_path):
    run = fake_run(tmp_path)
    run.variants = lambda *args: variants(50)
    run.memory = lambda *args: {"status": "transport_retry_exhausted", "calls": []}

    def global_call(*args):
        value = {"status": "context_limit_exceeded", "calls": []}
        mod.save(run.root / "assays/a/evaluations/n50/global.json", value)
        return value

    run.global_ranking = global_call
    run.cell("a", 50)
    e = mod.read(run.root / "assays/a/evaluations/n50/evaluation.json")
    assert not e["terminal"] and e["cr_status"] == "dependency_pending"
    assert run.report()["terminal_cells"] == 0


def test_bootstrap_shares_indices_across_N():
    values = np.array([[1, 2], [3, 6], [5, 10]], dtype=float)
    ci = mod.bootstrap(values)
    np.testing.assert_allclose(ci[1], ci[0] * 2)
    np.testing.assert_array_equal(ci, mod.bootstrap(values))


def test_global_2000_ids_and_invalid_output_are_not_completed(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    run = fake_run(tmp_path)
    rows = variants(2000)
    ids = [f"M{i + 1:02d}" for i in range(2000)]
    package = ModuleType("rg_llm")
    package.prompt = SimpleNamespace(
        SYSTEM_PROMPT="system", build_user_prompt=lambda *args: ("user", ids)
    )
    monkeypatch.setitem(sys.modules, "rg_llm", package)
    calls = []

    def call(*args):
        calls.append(1)
        return {"answer": json.dumps({"ranking": ids}), "usage": {"totalTokenCount": 10}}

    run.service = SimpleNamespace(call=call)
    path = run.root / "global.json"
    result = run._global_request("a", 2000, rows, path)
    assert result["status"] == "complete" and result["ranking"][-1] == "M2000"
    ids[-1] = ids[0]
    result = run._global_request("a", 2000, rows, path)
    assert result["status"] == "invalid_output" and result["ranking"] == []
    assert len(calls) == 2  # No generation retry until a Global answer becomes valid.


def test_provider_version_change_is_durable_before_pause(tmp_path):
    import threading

    store = SimpleNamespace(check_admission=lambda: None, stop=threading.Event())
    service = mod.Service(AppConfig(), tmp_path, store)
    service.count = lambda *args: 1
    service.adapter = SimpleNamespace(
        _call=lambda *args: SimpleNamespace(
            model_dump=lambda: {"answer": "{}", "model_version": "changed-provider-version"}
        )
    )
    path = tmp_path / "response.json"
    with pytest.raises(ExecutionPaused):
        service.call(path, "system", "user", "CR", {})
    assert path.exists() and store.stop.is_set()
