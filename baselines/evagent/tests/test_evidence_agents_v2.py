import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evagent.integrated.bundles import file_hash
from evagent.integrated.gateway import digest
from evagent.integrated.research.evidence_agents import SCHEMA, command, load_prepared
from evagent.integrated.research.literature import audit_document
from evagent.integrated.research.literature_review import (
    POLICY,
    check_background_query,
    screen_document,
    validate_passage,
)
from evagent.integrated.research.msa_agent import (
    MSAPreparationAgent,
    extract_homologs,
    search_hits,
    validate_usage,
)
from evagent.integrated.research.tools import FrozenTools


def asset(path, natural=False, **kwargs):
    return {"path": str(path), "sha256": file_hash(path), "source": "offline-test-fixture",
            "reviewed": True, **({"origin": "natural_homologs"} if natural else {}), **kwargs}


def paper():
    text = "Mutation studies reveal that RNA folding depends on magnesium.\nA12G decreased activity."
    cut = text.index("\n")
    return {"doi": "10.1234/mechanism", "title": "RNA mutation mechanisms", "text": text,
            "location": "body/paragraph-2", "date": "2020-01-01",
            "passage_review": {"policy": POLICY, "reviewer": "offline-curation-fixture",
                               "reviewed_at": "2026-09-13", "source_sha256": digest(text),
                               "spans": [
                                   {"start": 0, "end": cut, "classification": "mechanism", "relevance": "family",
                                    "applicability": "Same family, ionic environment only",
                                    "reason": "General mechanism, no candidate outcome", "limitations": "No position mapping"},
                                   {"start": cut + 1, "end": len(text), "classification": "direct_effect"}]}}


def precomputed(tmp_path):
    path = tmp_path / "input.a2m"
    path.write_text(">WT\nAcGU\n>alias\nACGU\n>natural1\nAC-U\n>natural2\nAU-U\n>duplicate\nAC-U\n")
    return {"mode": "precomputed", "usage": "external_fusion", "alignment": asset(
        path, True, format="a2m", query_identifier="WT")}


def test_precomputed_never_constructs_and_preserves_wt_insertions(tmp_path):
    cfg = precomputed(tmp_path)
    agent = MSAPreparationAgent(cfg, tmp_path, tmp_path / "result",
                                runner=lambda *a, **k: pytest.fail("Precomputed mode ran a tool"))
    source = agent.prepare("ACGU", [{"variant_id": "M1", "sequence": "AUGU"}])
    assert source["task"]["sequences_used"] == 2
    assert source["task"]["query_aliases_excluded"] == 2
    assert len(source["positions"]) == 4
    assert source["positions"]["3"]["gap_fraction"] == 1
    assert source["positions"]["3"]["support"] == "low"
    assert source["task"]["wt_to_alignment_column"] == {"1": 1, "2": 2, "3": 3, "4": 4}
    assert not agent.calls


def test_missing_precomputed_is_explicit_not_search(tmp_path):
    result = MSAPreparationAgent({"mode": "precomputed"}, tmp_path, tmp_path / "out").prepare("ACGU")
    assert result["status"] == "unavailable" and "disabled" in result["reason"]


def test_long_wt_msa_receipt_hash_survives_json_roundtrip(tmp_path):
    path = tmp_path / "long.a2m"
    wt = "ACGUACGUACGU"
    path.write_text(">WT\n" + wt + "\n>natural1\nACGUACGUACGA\n>natural2\nACGUACGUACGC\n")
    cfg = {"mode": "precomputed", "usage": "external_fusion", "alignment": asset(
        path, True, format="a2m", query_identifier="WT")}
    out = tmp_path / "long_result"
    source = MSAPreparationAgent(cfg, tmp_path, out).prepare(wt)
    receipt = json.loads((out / "receipt.json").read_text())
    receipt_hash = receipt.pop("sha256")
    assert receipt_hash == digest(receipt)
    assert digest(source) == digest(receipt["source"])
    assert receipt["source"]["task"]["wt_to_alignment_column"]["12"] == 12


@pytest.mark.parametrize("field,value", [("origin", "candidate_library"), ("reviewed", False), ("sha256", "changed")])
def test_msa_requires_verified_natural_asset(tmp_path, field, value):
    cfg = precomputed(tmp_path)
    cfg["alignment"][field] = value
    with pytest.raises(ValueError):
        MSAPreparationAgent(cfg, tmp_path, tmp_path / "out").prepare("ACGU")


def test_no_double_msa_and_no_prompt_tool(tmp_path):
    from evagent.integrated.research.native_tools import declarations

    with pytest.raises(ValueError, match="both"):
        validate_usage("external_fusion", ["msa"])
    model = SimpleNamespace(gateway=SimpleNamespace(run_dir=tmp_path / "gateway"), metadata={})
    task = {"variants": [{"variant_id": "M1", "sequence": "ACGU"}], "wt_sequence": "ACGU"}
    snapshot = {"msa_usage": "external_fusion", "sources": {}}
    tools = FrozenTools(task, snapshot, "single", model)
    assert tools.sources == ["structure", "literature"]
    schema = declarations("single", {"evidence_sources": tools.sources, "literature_policy": POLICY})
    assert schema[0]["parameters_json_schema"]["properties"]["source"]["enum"] == tools.sources
    assert "mutation mechanisms are allowed" in schema[1]["description"]
    assert "msa" in declarations("single")[0]["parameters_json_schema"]["properties"]["source"]["enum"]
    with pytest.raises(ValueError, match="Source unavailable"):
        tools.call("evidence", {"source": "msa"})
    with pytest.raises(ValueError, match="preparation"):
        FrozenTools(task, snapshot, "msa", model)


def test_mechanism_survives_but_effect_is_quarantined():
    record = paper()
    assert audit_document(record, []) is not None  # Legacy protocol unchanged.
    approved, decisions = screen_document(record)
    assert len(approved) == 1 and approved[0]["text"].startswith("Mutation studies")
    assert "A12G" not in json.dumps(approved)
    assert any(d["decision"] == "quarantine_passage" for d in decisions)
    validate_passage(approved[0])
    changed = copy.deepcopy(approved[0])
    changed["text"] += " edited"
    with pytest.raises(ValueError, match="changed"):
        validate_passage(changed)


@pytest.mark.parametrize("flag", ["target_experiment", "benchmark_derived", "target_supplement"])
def test_target_sources_remain_wholly_quarantined(flag):
    record = paper()
    record[flag] = True
    assert screen_document(record)[0] == []
    assert screen_document(paper(), ["https://doi.org/10.1234/mechanism"])[0] == []


def test_stale_or_missing_reviews_fail_closed():
    record = paper()
    record["text"] += " new"
    approved, decisions = screen_document(record)
    assert not approved and decisions[0]["decision"] == "pending_review"
    record = paper()
    record["passage_review"]["spans"][1].update(classification="mechanism", relevance="family",
                                               applicability="family", reason="wrong approval", limitations="none")
    assert len(screen_document(record)[0]) == 1  # Review cannot override direct-effect guard.


@pytest.mark.parametrize("query", ["RNA mutation mechanism", "CPEB3 magnesium folding"])
def test_mechanism_queries_allowed(query):
    check_background_query(query)


@pytest.mark.parametrize("query", ["A12G activity", "RNA DMS scores", "best mutant", "mutation effects"])
def test_candidate_queries_blocked(query):
    with pytest.raises(ValueError):
        check_background_query(query)


def test_family_claim_valid_without_position_or_candidate(tmp_path):
    tool = FrozenTools({"variants": []}, {"literature_policy": POLICY}, "literature", None)
    tool.results = [{"evidence": [{"id": "passage-one", "excerpt": "mechanism", "relevance": "family"}]}]
    report = {"claims": [{"evidence_ids": ["passage-one"], "candidate_ids": [], "positions": []}]}
    tool.validate_report(report)
    report["claims"][0]["positions"] = [12]
    with pytest.raises(ValueError, match="mapping"):
        tool.validate_report(report)


def table_hit(name, length=4):
    return f"{name} - family RF00000 cm 1 {length} 1 {length} + no 1 0.5 0.0 50.0 1e-9 ! description\n"


def test_build_executes_bounded_tools_on_wt_and_natural_database_only(tmp_path):
    cm = tmp_path / "family.cm"
    cm.write_text("test CM")
    db = tmp_path / "natural.fa"
    db.write_text(">h1\nACCU\n>h2\nAUCU\n")
    binary = tmp_path / "infernal"
    binary.write_text("offline runner")
    cfg = {"mode": "build", "max_families": 1, "families": [asset(cm)], "database": asset(db, True),
           "tools": {name: asset(binary) for name in ("cmsearch", "cmalign")}}
    seen = []

    def runner(argv, **kwargs):
        seen.append(argv)
        if "--tblout" in argv:
            path = Path(argv[argv.index("--tblout") + 1])
            path.write_text(table_hit("WT_query") if argv[-1].endswith("query.fa")
                            else table_hit("h1") + table_hit("h2"))
        elif "--outformat" in argv:
            Path(argv[argv.index("-o") + 1]).write_text("# STOCKHOLM 1.0\nWT_query ACGU\nhomolog_0 ACCU\nhomolog_1 AUCU\n//\n")

    agent = MSAPreparationAgent(cfg, tmp_path, tmp_path / "out", runner=runner)
    source = agent.prepare("ACGU", [{"variant_id": "NEVER_QUERY", "sequence": "UUUU"}])
    assert source["status"] == "available" and len(seen) == 5
    for argv in seen:
        if "--tblout" in argv:
            assert "UUUU" not in Path(argv[-1]).read_text()
    assert len((tmp_path / "out/tool_events.jsonl").read_text().splitlines()) == 10
    cfg["max_tool_calls"] = 1
    with pytest.raises(ValueError, match="allowance"):
        MSAPreparationAgent(cfg, tmp_path, tmp_path / "bounded", runner=runner).prepare("ACGU")


def test_search_reverse_coordinates_and_malformed_outputs(tmp_path):
    db = tmp_path / "db.fa"
    db.write_text(">h\nAACGUA\n")
    rows = extract_homologs(db, [{"id": "h", "start": 5, "end": 2, "strand": "-"}])
    assert rows[0][1] == "ACGU"
    bad = tmp_path / "bad.tbl"
    bad.write_text("broken")
    with pytest.raises(ValueError, match="Malformed"):
        search_hits(bad, .001)


def test_prepare_validate_and_frozen_input_tamper(tmp_path):
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"assay": "fixture", "wt_sequence": "ACGU", "variants": []}))
    lit = tmp_path / "papers.json"
    lit.write_text(json.dumps([paper()]))
    config = {"schema": SCHEMA, "task_file": "task.json", "literature_file": "papers.json",
              "output_dir": "result", "source_audit": {"status": "admitted"},
              "msa_agent": precomputed(tmp_path)}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert command(path, "prepare") == 0 and command(path, "validate") == 0
    result = load_prepared(tmp_path / "result/prepared.json")
    assert len(result["literature_documents"]) == 1
    assert "A12G" not in json.dumps(result)
    with pytest.raises(FileExistsError):
        command(path, "prepare")
    (tmp_path / "result/quarantine/documents.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        command(path, "validate")


def test_paperqa_reviewed_excerpts_use_gateway_and_original_offsets(tmp_path):
    from test_research_agents import gateway

    from evagent.integrated.research.contracts import BoundedModel
    from evagent.integrated.research.literature import PaperEvidence

    documents, _ = screen_document(paper())
    g = gateway(tmp_path)
    model = BoundedModel(g, {}, 6)
    result = PaperEvidence(documents, passage_policy=True).search("RNA mutation mechanism", model)
    assert result["evidence"]
    for excerpt in result["evidence"]:
        assert paper()["text"][excerpt["source_start"]:excerpt["source_end"]] == excerpt["excerpt"]
        assert "A12G" not in json.dumps(excerpt)
    assert g.summary()["provider_attempts"] == len(model.records) == 1
    g.close()


def test_external_msa_remains_outside_all_prediction_calls(tmp_path):
    from test_research_agents import fixture, transport

    from evagent.integrated.research.campaign import run_campaign

    config = fixture(tmp_path)
    path = tmp_path / "snapshot.json"
    snapshot = json.loads(path.read_text())
    snapshot.update(msa_usage="external_fusion")
    snapshot["sources"]["msa"]["task"] = {"sentinel": "MSA_PRIVATE_SENTINEL"}
    snapshot["sha256"] = digest({k: v for k, v in snapshot.items() if k != "sha256"})
    path.write_text(json.dumps(snapshot))
    calls = []

    def observed(request):
        assert "MSA_PRIVATE_SENTINEL" not in json.dumps(request)
        assert request["metadata"].get("role") != "msa"
        calls.append(request)
        return transport(request)

    result = run_campaign(config, transport=observed)
    assert result["status"] == "complete" and result["statuses"] == {"ok": 9}
    assert result["ledger"]["provider_attempts"] == len(calls)


def test_automatic_family_discovery_keeps_reference_library_immutable(tmp_path):
    library = tmp_path / "reference.cm"
    library.write_text("pinned family library")
    indices = {}
    for suffix in (".i1f", ".i1i", ".i1m", ".i1p"):
        path = tmp_path / ("reference.cm" + suffix)
        path.write_text("pinned index " + suffix)
        indices[suffix] = asset(path)
    tools = {}
    for name in ("cmsearch", "cmalign", "cmscan", "cmfetch"):
        path = tmp_path / name
        path.write_text("offline executable fixture")
        tools[name] = asset(path)
    database = tmp_path / "natural.fa"
    database.write_text(">h1\nACCU\n>h2\nAUCU\n")
    config = {"mode": "build", "family_library": asset(library, indices=indices),
              "database": asset(database, True), "tools": tools}
    checksum = file_hash(library)

    def runner(argv, **kwargs):
        if "-h" in argv:
            return
        if Path(argv[0]).name == "cmfetch":
            Path(argv[argv.index("-o") + 1]).write_text("extracted family")
        elif "--tblout" in argv:
            text = table_hit("family_A") if Path(argv[0]).name == "cmscan" else table_hit("h1") + table_hit("h2")
            Path(argv[argv.index("--tblout") + 1]).write_text(text)
        else:
            Path(argv[argv.index("-o") + 1]).write_text("# STOCKHOLM 1.0\nWT_query ACGU\nh1 ACCU\nh2 AUCU\n//\n")

    agent = MSAPreparationAgent(config, tmp_path, tmp_path / "new", runner=runner)
    result = agent.prepare("ACGU")
    assert result["task"]["family"] == "family_A"
    assert len(agent.calls) == 8 and file_hash(library) == checksum


def test_reviewed_paperqa_unknown_usage_is_not_abstention(tmp_path):
    from test_research_agents import gateway

    from evagent.integrated.research.contracts import BoundedModel
    from evagent.integrated.research.literature import PaperEvidence
    from evagent.runtime import ExecutionPaused

    g = gateway(tmp_path, lambda request: {})
    documents, _ = screen_document(paper())
    with pytest.raises(ExecutionPaused):
        PaperEvidence(documents, passage_policy=True).search("RNA mechanism", BoundedModel(g, {}, 6))
    assert g.summary()["paused"] and g.summary()["provider_attempts"] == 1
    g.close()


def test_prepared_evidence_attaches_to_research_snapshot(tmp_path):
    from test_research_agents import fixture

    from evagent.integrated.research.prepare import build_snapshot
    from evagent.integrated.tasks import load_task

    campaign = json.loads(fixture(tmp_path).read_text())
    spec = campaign["tasks"][0]
    task = load_task(spec)
    public = {"assay": task.assay, "wt_sequence": task.wt_sequence, "variants": task.variants}
    (tmp_path / "public.json").write_text(json.dumps(public))
    (tmp_path / "papers.json").write_text(json.dumps([paper()]))
    config = {"schema": SCHEMA, "task_file": "public.json", "output_dir": "prepared",
              "source_audit": {"status": "admitted"}, "literature_file": "papers.json"}
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(config))
    command(path, "prepare")
    spec.update(source_audit={"status": "admitted"}, task_reviewed=True,
                prepared_agents=str(tmp_path / "prepared/prepared.json"))
    result = build_snapshot(spec, tmp_path / "new-snapshot/snapshot.json", [])
    assert result["literature_policy"] == POLICY
    assert result["msa_usage"] == "external_fusion"
    assert len(result["literature_documents"]) == 1
    assert result["sources"]["structure"]["status"] == "available"
