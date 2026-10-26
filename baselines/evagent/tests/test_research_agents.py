import json
import re

import pytest

from evagent.integrated.gateway import RequestGateway
from evagent.integrated.research.campaign import run_campaign
from evagent.integrated.research.contracts import (
    SCHEMA,
    BoundedModel,
    CallLimit,
    digest,
    fuse,
    save,
    validate_claims,
)
from evagent.integrated.research.literature import PaperEvidence, audit_document, check_query
from evagent.integrated.research.tools import FrozenTools
from evagent.runtime import ExecutionPaused

MODEL = {
    "model": "gemini-3.8-flash",
    "thinking_level": "medium",
    "temperature": 0,
    "generation_seed": 1,
    "max_output_tokens": 32768,
}


def response(text):
    return {
        "final_text": text,
        "finish_reason": "STOP",
        "usage_summary": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    }


def transport(request):
    user = request["messages"][-1]["content"]
    if request["metadata"].get("phase") == "paperqa_evidence":
        return response(
            json.dumps({"summary": "RNA folding depends on its environment.", "relevance_score": 8})
        )
    if request["metadata"].get("phase") == "analysis":
        history = json.loads(user)
        source_schema = history[0]["task"]["tools"]["evidence"]["arguments"]["source"]
        assert source_schema["type"] == "string"
        sources = source_schema["enum"]
        read = {x.get("request", {}).get("arguments", {}).get("source") for x in history}
        missing = [source for source in sources if source not in read]
        if missing:
            return response(json.dumps({"tool": "evidence", "arguments": {"source": missing[0]}}))
        evidence = next(x["response"]["evidence"][0]["id"] for x in history if "response" in x)
        return response(
            json.dumps(
                {
                    "claims": [
                        {
                            "text": "The supplied source reports its coverage.",
                            "evidence_ids": [evidence],
                            "candidate_ids": [],
                            "positions": [],
                        }
                    ],
                    "limitations": ["Context only"],
                    "abstain": False,
                }
            )
        )
    ids = re.search(r"Required IDs: ([^\n]+)", user).group(1).split(", ")
    return response(
        json.dumps(
            {"variants": [{"variant_id": i, "predicted_score": int(i[1:]) / 300} for i in ids]}
        )
    )


def gateway(tmp_path, fn=transport, **kwargs):
    return RequestGateway(
        model_config=MODEL,
        run_dir=tmp_path,
        fingerprint="test",
        token_budget=None,
        budget_mode="unlimited",
        transport=fn,
        **kwargs,
    )


def test_unlimited_accounting_and_pause(tmp_path):
    g = gateway(tmp_path, lambda r: response("OK"))
    for i in range(5):
        g.generate(system="test", user=str(i), run_name="test")
    assert g.summary()["reported_tokens"] == 100 and not g.summary()["paused"]
    assert g.generate(system="test", user="0", run_name="test")["cache_hit"]
    g.transport = lambda r: {"finish_reason": "STOP", "final_text": "x"}
    with pytest.raises(ExecutionPaused):
        g.generate(system="test", user="unknown", run_name="test")
    with pytest.raises(ExecutionPaused):
        g.generate(system="test", user="another", run_name="test")
    assert g.summary()["provider_attempts"] == 6
    g.close()


def test_budget_validation_and_local_allowance(tmp_path):
    with pytest.raises(ValueError):
        RequestGateway(
            model_config=MODEL,
            run_dir=tmp_path / "bad",
            fingerprint="x",
            token_budget=2,
            budget_mode="unlimited",
        )
    g = gateway(tmp_path / "good", lambda r: response("OK"))
    b = BoundedModel(g, {}, 1)
    b.generate("x", "y")
    with pytest.raises(CallLimit):
        b.generate("x", "z")
    assert g.summary()["provider_attempts"] == 1
    g.close()


@pytest.mark.parametrize("query", ["C25U activity", "best mutant", "fitness ranking", "DMS score"])
def test_literature_query_guard(query):
    with pytest.raises(ValueError):
        check_query(query)


def test_document_contamination_and_claims():
    d = {
        "doi": "10.1/test",
        "title": "RNA",
        "text": "Mutations change activity",
        "location": "abstract",
        "relevance": "family",
    }
    assert audit_document(d, []) is not None
    d["text"] = "RNA folds."
    assert audit_document(d, ["10.1/test"]) == "benchmark_or_related_DOI"
    assert audit_document(d, []) is None
    with pytest.raises(ValueError):
        validate_claims(
            {
                "claims": [
                    {
                        "text": "x",
                        "evidence_ids": ["invented"],
                        "candidate_ids": [],
                        "positions": [],
                    }
                ],
                "limitations": [],
                "abstain": False,
            },
            set(),
            set(),
            set(),
        )


def test_real_paperqa_calls_are_bounded_and_recorded(tmp_path):
    g = gateway(tmp_path)
    b = BoundedModel(g, {}, 6)
    d = PaperEvidence(
        [
            {
                "id": "paper1",
                "doi": "10.1/test",
                "title": "RNA folding",
                "text": "RNA folding depends on its ionic environment.",
                "location": "abstract",
                "relevance": "family",
            }
        ]
    )
    result = d.search("RNA folding environment", b)
    assert result["evidence"][0]["id"] == "paper1-char0"
    assert len(b.records) == 1 and g.summary()["reported_tokens"] == 20
    g.close()


def test_paperqa_unknown_response_propagates(tmp_path):
    g = gateway(tmp_path, lambda r: {})
    b = BoundedModel(g, {}, 6)
    d = PaperEvidence(
        [
            {
                "id": "paper1",
                "doi": "10.1/test",
                "title": "RNA folding",
                "text": "RNA folding environment.",
                "location": "abstract",
                "relevance": "family",
            }
        ]
    )
    with pytest.raises(ExecutionPaused):
        d.search("RNA folding environment", b)
    assert g.summary()["provider_attempts"] == 1
    g.close()


def fixture(tmp_path):
    variants = [
        {"variant_id": f"M{i:02d}", "mutation": "A1C", "sequence": "CGCU", "score": float(i)}
        for i in range(1, 51)
    ]
    save(tmp_path / "sample.json", {"assay": "synthetic_test", "size": 50, "variants": variants})
    save(
        tmp_path / "context.json",
        {
            "assay": "synthetic_test",
            "wt_sequence": "AGCU",
            "metadata": {"DMS_ID": "synthetic_test", "RNA_TYPE": "synthetic"},
        },
    )
    public = [{k: v[k] for k in ("variant_id", "mutation", "sequence")} for v in variants]
    source = {
        "status": "unavailable",
        "reason": "fixture",
        "task": {},
        "positions": {},
        "candidates": {},
        "provenance": [],
    }
    snapshot = {
        "schema": SCHEMA,
        "assay": "synthetic_test",
        "source_audit": {"status": "admitted"},
        "task_reviewed": True,
        "wt_sha256": digest("AGCU"),
        "variants_sha256": digest(public),
        "sources": {s: source for s in ("msa", "structure", "literature")},
        "msa_values": None,
        "assets": [],
    }
    snapshot["sha256"] = digest(snapshot)
    save(tmp_path / "snapshot.json", snapshot)
    config = {
        "schema": "evagent-research-campaign-v1",
        "output_dir": str(tmp_path / "out"),
        "model_config": MODEL,
        "budget_mode": "unlimited",
        "token_budget": None,
        "repeats": 3,
        "group_size": 25,
        "seed": 1,
        "max_repairs": 1,
        "tasks": [
            {
                "assay": "synthetic_test",
                "family": "synthetic",
                "sample_file": str(tmp_path / "sample.json"),
                "context_file": str(tmp_path / "context.json"),
                "snapshot_file": str(tmp_path / "snapshot.json"),
            }
        ],
    }
    save(tmp_path / "campaign.json", config)
    return tmp_path / "campaign.json"


def test_complete_three_condition_matrix_and_restart(tmp_path):
    p = fixture(tmp_path)
    dry = run_campaign(p, dry_run=True)
    assert dry["initial_CR_units"] == 18 and dry["api_calls"] == 0
    report = run_campaign(p, transport=transport)
    assert report["status"] == "complete" and report["completed_units"] == 18
    assert report["statuses"] == {"ok": 9}
    again = run_campaign(p, transport=lambda r: pytest.fail("Cached run submitted request"))
    assert again["ledger"]["reported_tokens"] == report["ledger"]["reported_tokens"]
    config = json.loads(p.read_text())
    config["model_config"]["max_output_tokens"] = 16000
    save(p, config)
    with pytest.raises(ValueError, match="changed"):
        run_campaign(p, transport=transport)


def test_unknown_response_stops_whole_campaign(tmp_path):
    p = fixture(tmp_path)
    report = run_campaign(p, transport=lambda r: {})
    assert report["status"] == "paused" and report["ledger"]["provider_attempts"] == 1
    assert report["completed_units"] == 0


def test_fusion_ties_complete_and_missing_msa():
    assert fuse(["a", "b"], {"a": 1.0, "b": 1.0}, None) == {"a": 1.5, "b": 1.5}
    with pytest.raises(ValueError):
        fuse(["a", "b"], {"a": 1.0, "b": 2.0}, {"a": 1.0})


def test_task_specific_relevance_not_generic_rna():
    from evagent.integrated.research.literature import relevance_gate

    assert relevance_gate(
        "glmS ribozyme folding",
        {"title": "The current riboswitch landscape in Clostridioides difficile"},
    )
    assert relevance_gate("glmS ribozyme folding", {"title": "A hammerhead ribozyme sensor"})
    assert (
        relevance_gate(
            "glmS ribozyme folding", {"title": "Co-transcriptional folding of the glmS ribozyme"}
        )
        is None
    )


def test_discovery_is_label_free_and_ledger_owned(tmp_path):
    from evagent.integrated.research.discovery import plan_queries

    save(
        tmp_path / "context.json",
        {"metadata": {"ASSAY_DESCRIPTION": "RNA folding background"}, "score": 999},
    )
    observed = []

    def fake(r):
        observed.append(r)
        return response(
            json.dumps({"queries": ["glmS ribozyme folding", "glmS ribozyme processing"]})
        )

    g = gateway(tmp_path / "provider", fake)
    spec = {
        "assay": "synthetic_test",
        "literature_query": "glmS ribozyme structure",
        "family": "synthetic_consensus_glmS",
        "context_file": str(tmp_path / "context.json"),
    }
    assert len(plan_queries(spec, g, tmp_path)) == 3
    assert (
        "999" not in json.dumps(observed) and "score" not in observed[0]["messages"][-1]["content"]
    )
    assert len(plan_queries(spec, g, tmp_path)) == 3 and len(observed) == 1
    g.close()


def test_agent_api_failure_does_not_enter_semantic_repair(tmp_path):
    p = fixture(tmp_path)
    phases = []

    def fail_analysis(r):
        phases.append(r["metadata"]["phase"])
        return {} if r["metadata"]["phase"] == "analysis" else transport(r)

    result = run_campaign(p, transport=fail_analysis)
    assert result["status"] == "paused"
    assert phases == ["integration", "analysis"]
    assert result["completed_units"] == 1


def test_tool_whitelist_positions_and_other_candidate_isolation(tmp_path):
    g = gateway(tmp_path)
    model = BoundedModel(g, {"role": "msa"}, 6)
    task = {
        "assay": "synthetic",
        "metadata": {},
        "wt_sequence": "AGCU",
        "variants": [{"variant_id": "M01", "sequence": "CGCU"}],
    }
    source = {"status": "unavailable", "reason": "test"}
    snapshot = {
        "sha256": "test",
        "sources": {"msa": source, "structure": source, "literature": source},
    }
    tools = FrozenTools(task, snapshot, "msa", model)
    for name, args in [
        ("execute", {}),
        ("evidence", {"source": "structure"}),
        ("evidence", {"source": "msa", "positions": [0]}),
        ("evidence", {"source": "msa", "candidate_ids": ["M02"]}),
    ]:
        with pytest.raises(ValueError):
            tools.call(name, args)
    assert len(tools.records) == 4 and not tools.observed
    assert len(list((tmp_path.parent / "tool_events").glob("*.jsonl"))) == 1
    g.close()


def test_research_snapshot_mapping_and_literature_validation(tmp_path):
    from evagent.integrated.research.contracts import load_snapshot
    from evagent.integrated.tasks import load_task

    p = fixture(tmp_path)
    config = json.loads(p.read_text())
    task = load_task(config["tasks"][0])
    path = tmp_path / "snapshot.json"
    original = json.loads(path.read_text())
    for key, value in [("position", {"0": {"wt": "A"}}), ("base", {"1": {"wt": "C"}})]:
        snapshot = json.loads(json.dumps(original))
        snapshot["sources"]["msa"]["positions"] = value
        snapshot["sha256"] = digest({k: v for k, v in snapshot.items() if k != "sha256"})
        save(path, snapshot)
        with pytest.raises(ValueError):
            load_snapshot(path, task)
    snapshot = json.loads(json.dumps(original))
    snapshot["literature_documents"] = [
        {
            "id": "leak",
            "doi": "10.1/leak",
            "title": "Variant effects",
            "text": "Mutation results",
            "location": "abstract",
            "relevance": "family",
        }
    ]
    snapshot["sha256"] = digest({k: v for k, v in snapshot.items() if k != "sha256"})
    save(path, snapshot)
    with pytest.raises(ValueError, match="contamination"):
        load_snapshot(path, task)


def test_local_model_exhaustion_is_not_a_valid_cr_fallback(tmp_path):
    p = fixture(tmp_path)

    def never_final(request):
        if request["metadata"].get("phase") == "analysis":
            return response('{"tool":"evidence","arguments":{"source":"msa"}}')
        return transport(request)

    report = run_campaign(p, transport=never_final)
    assert report["status"] == "paused"
    assert report["statuses"]["incomplete"] > 0
    assert report["ledger"]["provider_attempts"] <= 2 + 2 * 18 + 2 * 3 * 6
    units = [json.loads(p.read_text()) for p in (tmp_path / "out/units").glob("*.json")]
    assert all(
        u["status"] == "analysis_invalid" and u["cr"] is None
        for u in units
        if u["condition"] != "reference"
    )


def test_finite_research_budget_stops_before_submission(tmp_path):
    p = fixture(tmp_path)
    config = json.loads(p.read_text())
    config.update(budget_mode="limited", token_budget=1)
    save(p, config)
    report = run_campaign(
        p, transport=lambda r: pytest.fail("Budget admitted an oversized request")
    )
    assert report["status"] == "paused" and report["ledger"]["provider_attempts"] == 0
    assert report["completed_units"] == 0


def test_selected_experiment_reports_only_selected_cell(tmp_path):
    p = fixture(tmp_path)
    selection = {
        "assay": "synthetic_test",
        "condition": "reference",
        "repeat": 2,
        "output_dir": str(tmp_path / "selected"),
    }
    dry = run_campaign(p, dry_run=True, selection=selection)
    assert dry["method_results"] == 1 and dry["initial_CR_units"] == 2
    report = run_campaign(p, transport=transport, selection=selection)
    assert len(report["rows"]) == 1 and report["rows"][0]["repeat"] == 2


@pytest.mark.parametrize(
    "text",
    [
        "A single base change in tRNA affects cleavage",
        "Nucleotide substitution changes processing",
        "G42A alters turnover",
        "Deletion of part of an intron",
    ],
)
def test_mutation_effect_synonyms_are_quarantined(text):
    record = {
        "doi": "10.1/test",
        "title": text,
        "text": text,
        "location": "abstract",
        "relevance": "family",
    }
    assert audit_document(record, []) == "mutation_effect_or_supplement_contamination"


def test_partial_stream_usage_is_reported_as_unknown_cost(tmp_path):
    from evagent.integrated.gateway import ProviderStreamInterrupted

    p = fixture(tmp_path)

    def interrupted(request):
        raise ProviderStreamInterrupted(
            {
                "final_text": "",
                "finish_reason": None,
                "usage_summary": {"input_tokens": 10, "output_tokens": None, "total_tokens": 10},
            }
        )

    report = run_campaign(p, transport=interrupted)
    assert report["status"] == "paused" and report["completed_units"] == 0
    assert report["ledger"]["reported_tokens"] == 10
    assert (
        report["audit"]["execution_cost_by_phase_condition_role"][0]["unknown_usage_attempts"] == 1
    )


def test_frozen_preparation_cannot_reopen_its_paid_ledger(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    save(tmp_path / "campaign_final.json", {})
    script = Path(__file__).resolve().parents[1] / "work/multiagent_v1/enrich_literature.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode != 0 and "already frozen" in result.stderr
    assert not (tmp_path / "preparation").exists()
