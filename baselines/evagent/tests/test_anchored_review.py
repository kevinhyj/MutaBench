import json
import re
from pathlib import Path

import pytest

from evagent.integrated.gateway import RequestGateway, digest
from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.research.review import (
    ROLE_SOURCES, cards_for, normalize_calls, run_reviewer, validate_checks,
)
from evagent.runtime import ExecutionPaused
from test_research_agents import MODEL, fixture, response


def native(calls):
    return {**response(""), "tool_calls": [{"name": "evidence", "args": {"source": s}} for s in calls]}


def mock(request):
    meta = request["metadata"]
    user = request["messages"][-1]["content"]
    if meta.get("phase") == "analysis":
        history = json.loads(user)
        task = history[0]["task"]
        seen = {x["tool_result"]["source"] for x in history if "tool_result" in x}
        missing = [s for s in ROLE_SOURCES[meta["role"]] if s not in seen]
        if missing:
            return native(missing)
        facts = [c for x in history if "tool_result" in x for c in x["tool_result"]["evidence"]]
        return response(json.dumps({"checks": [
            {"variant_id": v["variant_id"], "decision": "review", "reason": "Check the frozen candidate evidence.",
             "evidence_ids": [c["id"] for c in facts if c["variant_id"] == v["variant_id"]]}
            for v in task["variants"]], "limitations": ["Mock"]}))
    ids = re.search(r"Required IDs: ([^\n]+)", user).group(1).split(", ")
    return response(json.dumps({"variants": [{"variant_id": i, "predicted_score": int(i[1:]) / 100
                                              * (1 if meta["condition"] == "reference" else -1)} for i in ids]}))


def config(tmp_path):
    p = fixture(tmp_path)
    value = json.loads(p.read_text())
    value.update(analysis_profile="anchored_review_v2", repeats=1)
    value["model_config"]["research_tool_protocol"] = "native_review_v2"
    p.write_text(json.dumps(value))
    return p


def test_single_seed_complete_matrix_shared_draft_and_resume(tmp_path):
    path = config(tmp_path)
    assert run_campaign(path, dry_run=True)["method_results"] == 3
    sent = []
    def transport(request):
        sent.append(request)
        return mock(request)
    result = run_campaign(path, transport=transport)
    assert result["status"] == "complete" and result["completed_units"] == 6
    assert result["statuses"] == {"ok": 3}
    assert all(len(c["pairs"]) == 1 for c in result["comparisons"] if c["n"] == 50)
    reference = [r for r in sent if r["metadata"]["condition"] == "reference"]
    assert len(reference) == 2  # A new draft for each group, shared across all conditions.
    assert result["ledger"]["provider_attempts"] == len(sent)
    for p in (tmp_path / "out/units").glob("*.json"):
        u = json.loads(p.read_text())
        if u["condition"] != "reference":
            for i, value in u["cr"].items():
                assert value == pytest.approx(0.5 * u["draft_cr"][i])
            for r in u["reports"].values():
                assert len(r["checks"]) == 25
    again = run_campaign(path, transport=lambda r: pytest.fail("Resume sent a new request"))
    assert again["ledger"]["provider_attempts"] == len(sent)


@pytest.mark.parametrize("protocol,expected", [("native_v1", "terminal_failure"), ("native_review_v2", "ok")])
def test_native_batch_is_versioned_and_counted_once(tmp_path, protocol, expected):
    g = RequestGateway(model_config={**MODEL, "research_tool_protocol": protocol}, run_dir=tmp_path,
                       fingerprint="f", budget_mode="unlimited", token_budget=None,
                       transport=lambda r: native(["sequence", "structure"]))
    r = g.generate(system="x", user="x", run_name="batch", metadata={"phase": "analysis", "role": "review_single"})
    assert r["status"] == expected
    assert g.summary()["provider_attempts"] == 1
    g.close()


def test_whole_batch_rejects_unauthorized_function():
    assert normalize_calls([{"name": "evidence", "args": {"source": "sequence"}},
                            {"name": "shell", "args": {}}], "review_single") is None


def test_unknown_usage_pauses_and_cannot_be_repaired(tmp_path):
    path = config(tmp_path)
    def bad(request):
        if request["metadata"]["phase"] == "analysis":
            return {"finish_reason": "STOP", "final_text": "{}"}
        return mock(request)
    r = run_campaign(path, transport=bad)
    assert r["status"] == "paused" and r["completed_units"] == 1
    assert r["ledger"]["paused"]
    assert r["ledger"]["provider_attempts"] == 2


def test_compact_structure_hides_msa_and_low_confidence_geometry():
    task = {"wt_sequence": "AGCU", "variants": [{"variant_id": "M01", "mutation": "A1C", "sequence": "CGCU"}]}
    snapshot = {"msa_values": {"M01": 999}, "sources": {"msa": {"frequency": 999}, "structure": {
        "task": {"tertiary_structure": {"kind": "predicted_WT"}},
        "status": "available",
        "candidates": {digest("CGCU"): {"changed_positions": [1], "delta_mfe_kcal_mol": 0.5}},
        "positions": {"1": {"plddt": 42, "contact_positions": [3], "wt_unpaired_probability": 0.8}}}}}
    card = cards_for(task, snapshot, "structure")[0]
    assert card["positions"]["1"]["geometry_status"] == "withheld_low_or_unknown_confidence"
    assert "999" not in json.dumps(card) and "contact_positions" not in json.dumps(card)
    assert card["secondary"]["delta_mfe_kcal_mol"] == 0.5
    with pytest.raises(ValueError):
        cards_for(task, snapshot, "msa")
    sequence = cards_for(task, snapshot, "sequence")[0]
    assert sequence["substitutions"] == [{"position": 1, "wt": "A", "candidate": "C"}]
    assert "wt_sequence" not in sequence and "sequence" not in sequence
    snapshot["sources"]["structure"]["candidates"] = {"M01": {"changed_positions": [1]}}
    with pytest.raises(ValueError, match="sequence hash"):
        cards_for(task, snapshot, "structure")


def test_complete_checks_reject_missing_duplicate_and_wrong_citation():
    row = {"variant_id": "M01", "decision": "retain", "evidence_ids": ["e1"], "reason": "uncertain"}
    observed = {"e1": {"variant_id": "M01"}, "e2": {"variant_id": "M02"}}
    for checks in [[], [row, row], [{**row, "evidence_ids": ["e2"]}]]:
        with pytest.raises(ValueError):
            validate_checks({"checks": checks, "limitations": []}, ["M01"], observed)
    assert validate_checks({"checks": [row], "limitations": []}, ["M01"], observed)


def test_legacy_repeats_and_new_fingerprint_isolation(tmp_path):
    path = fixture(tmp_path)
    legacy, _ = prepare(path)
    p = json.loads(path.read_text()); p["repeats"] = 1; path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        prepare(path)
    p.update(analysis_profile="anchored_review_v2")
    p["model_config"]["research_tool_protocol"] = "native_review_v2"
    path.write_text(json.dumps(p))
    new, _ = prepare(path)
    assert new["fingerprint"] != legacy["fingerprint"]
    p["execution"] = {"unit_workers": 1, "max_concurrent_requests": 1, "parallel_specialists": False}
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        prepare(path)


def test_all_retain_is_audited_identity_without_integration_call(tmp_path):
    path = config(tmp_path)
    calls = []
    def keep(request):
        calls.append(request)
        output = mock(request)
        if request["metadata"]["phase"] == "analysis" and output["final_text"]:
            value = json.loads(output["final_text"])
            for row in value["checks"]:
                row["decision"] = "retain"
            return response(json.dumps(value))
        return output
    report = run_campaign(path, transport=keep)
    assert report["statuses"] == {"ok": 3}
    assert all(r["metadata"]["condition"] == "reference" for r in calls if r["metadata"]["phase"] == "integration")
    assert all(r["changed_predictions"] == 0 for r in report["rows"])


def test_expert_loop_is_bounded_and_every_tool_is_logged(tmp_path):
    g = RequestGateway(model_config={**MODEL, "research_tool_protocol": "native_review_v2"},
                       run_dir=tmp_path / "provider", fingerprint="loop", budget_mode="unlimited",
                       token_budget=None, transport=lambda r: native(["sequence"]))
    task = {"wt_sequence": "AGCU", "variants": [{"variant_id": "M01", "sequence": "CGCU"}]}
    report = run_reviewer(task, {"sources": {"structure": {}}}, "review_sequence", g, {"assay": "synthetic"})
    assert report["status"] == "limit_or_invalid"
    assert len(report["model_calls"]) == 6 and len(report["tool_calls"]) == 6
    assert g.summary()["provider_attempts"] == 6
    events = [json.loads(line) for p in (tmp_path / "tool_events").glob("*.jsonl") for line in p.read_text().splitlines()]
    assert len(events) == 12 and sum(e["status"] == "ok" for e in events) == 6
    g.close()


def test_labels_stay_outside_every_model_request(tmp_path):
    path = config(tmp_path)
    sample_path = tmp_path / "sample.json"
    sample = json.loads(sample_path.read_text())
    for n, row in enumerate(sample["variants"]):
        row["score"] = 987654321 + n
    sample_path.write_text(json.dumps(sample))
    def isolated(request):
        assert "987654" not in json.dumps(request)
        return mock(request)
    assert run_campaign(path, transport=isolated)["status"] == "complete"
