import json

import pytest
from test_anchored_review import config as old_config
from test_anchored_review import mock as old_mock
from test_research_agents import MODEL, response

from evagent.integrated.gateway import ExecutionPaused, RequestGateway, digest
from evagent.integrated.research.campaign import prepare, run_campaign
from evagent.integrated.research.contracts import BoundedModel
from evagent.integrated.research.review import run_reviewer
from evagent.integrated.research.structure_analysis import (
    StructureAnalysis,
    coordinate_asset,
    preflight,
)
from evagent.integrated.research.structure_review import (
    SYSTEM,
    ActiveReviewTools,
    declarations,
    factory,
    normalize_calls,
)
from evagent.integrated.tasks import file_hash


def task_fixture():
    return {
        "assay": "synthetic_tool_test",
        "wt_sequence": "GGGAAACCC",
        "metadata": {},
        "variants": [
            {"variant_id": "M01", "sequence": "GGGAAACCC", "mutation": "WT"},
            {"variant_id": "M02", "sequence": "CGGAAACCC", "mutation": "G1C"},
            {"variant_id": "M03", "sequence": "CGGAAACCG", "mutation": "G1C:C9G"},
        ],
    }


def snapshot_fixture():
    return {
        "sha256": "synthetic_snapshot",
        "assets": [],
        "sources": {
            "structure": {
                "status": "unavailable",
                "reason": "synthetic fixture",
                "task": {},
                "positions": {},
                "candidates": {},
            }
        },
    }


def backend(tmp_path):
    return StructureAnalysis(
        task_fixture(), snapshot_fixture(), tmp_path / "snapshot.json", tmp_path / "cache"
    )


def test_real_vienna_full_sequences_pairing_ensemble_and_joint(tmp_path):
    b = backend(tmp_path)
    rows = b.secondary(["M01", "M02", "M03"])
    assert rows[0]["delta_mfe_kcal_mol"] == 0
    assert rows[0]["mean_pair_probability_change"] == 0
    assert rows[1]["changed_positions"] == [1] and rows[2]["changed_positions"] == [1, 9]
    for row in rows:
        assert all(1 <= i <= 9 for pair in row["pair_changes"] for i in pair["positions"])
        assert all(
            0 <= pair[k] <= 1
            for pair in row["pair_changes"]
            for k in ("wt_probability", "candidate_probability")
        )
    ens = b.secondary(["M03"], mode="ensemble")[0]
    assert len(ens["candidate"]["centroid"]) == len(ens["candidate"]["mea"]) == 9
    joint = b.secondary(["M03"], mode="joint_changes")[0]
    assert len(joint["single_change_counterfactuals"]) == 2
    assert joint["nonadditivity_mfe_kcal_mol"] == pytest.approx(
        joint["delta_mfe_kcal_mol"]
        - sum(x["delta_mfe_kcal_mol"] for x in joint["single_change_counterfactuals"])
    )
    assert joint["candidate_sequence_sha256"] == digest("CGGAAACCG")
    assert b.secondary(["M02"], mode="joint_changes")[0]["status"] == "unavailable"
    before = len(b.cache.events)
    b.secondary(["M03"])
    assert all(x["cache_hit"] for x in b.cache.events[before:])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ids=["OTHER"]),
        dict(ids=["M01", "M01"]),
        dict(ids=["M01"], positions=[0]),
        dict(ids=["M01"], positions=[True]),
        dict(ids=["M01"], positions=[10]),
        dict(ids=["M01"], mode="arbitrary"),
    ],
)
def test_coordinate_and_candidate_argument_validation(tmp_path, kwargs):
    with pytest.raises(ValueError):
        backend(tmp_path).secondary(**kwargs)


def test_cache_corruption_and_tool_parameters_invalidate(tmp_path):
    b = backend(tmp_path)
    b.fold(b.wt)
    path = next((tmp_path / "cache").glob("*.json"))
    value = json.loads(path.read_text())
    value["value"]["mfe_kcal_mol"] = 999
    path.write_text(json.dumps(value))
    with pytest.raises(ExecutionPaused, match="integrity"):
        b.fold(b.wt)
    b.cache.signature = {**b.cache.signature, "test_new_parameters": "different"}
    assert b.fold(b.wt)["mfe_kcal_mol"] != 999
    assert not b.cache.events[-1]["cache_hit"]
    path = b.cache.root / (b.cache.events[-1]["computation_key"] + ".json")
    path.write_text("{broken")
    with pytest.raises(ExecutionPaused, match="unreadable"):
        b.fold(b.wt)


def cif_fixture(path):
    path.write_text("""data_test
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 polyribonucleotide ACGU
loop_
_atom_site.label_entity_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.label_atom_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
1 A 1 P 0 0 0 20 1
1 A 2 P 1 0 0 90 1
1 A 3 P 2 0 0 90 1
1 A 4 P 3 0 0 90 1
""")
    s = snapshot_fixture()
    s["assets"] = [{"path": path.name, "sha256": file_hash(path)}]
    s["sources"]["structure"]["task"]["tertiary_structure"] = {"kind": "experimental_WT_construct"}
    t = {"wt_sequence": "ACGU", "variants": [{"variant_id": "M01", "sequence": "CCGU"}]}
    return t, s


def test_experimental_kind_not_treated_as_low_plddt_and_mapping_verified(tmp_path):
    t, s = cif_fixture(tmp_path / "wt.cif")
    p = preflight(t, s, tmp_path / "snapshot.json")
    assert p["eligible_geometry_positions"] == [1, 2, 3, 4]
    s["sources"]["structure"]["task"]["tertiary_structure"]["kind"] = "predicted_WT"
    assert preflight(t, s, tmp_path / "snapshot.json")["eligible_geometry_positions"] == [2, 3, 4]
    wrong = {**t, "wt_sequence": "ACGA"}
    assert preflight(wrong, s, tmp_path / "snapshot.json")["geometry_status"] == "unavailable"
    (tmp_path / "wt.cif").write_text("changed")
    with pytest.raises(ExecutionPaused, match="asset"):
        coordinate_asset(s, tmp_path / "snapshot.json")


def test_neighborhood_endpoints_and_torsion_dependencies_are_filtered(tmp_path, monkeypatch):
    b = backend(tmp_path)
    b.coverage["eligible_geometry_positions"] = [1, 3, 9]
    geometry = {
        "status": "available",
        "geometry_kind": "predicted_WT",
        "coordinate_asset_sha256": "x",
        "positions": {
            str(i): {"plddt": 90, "backbone_angles_radians": [0.1] * 7} for i in [1, 3, 9]
        },
        "heavy_atoms": {"1": [[0, 0, 0]], "3": [[2, 0, 0]], "9": [[7, 0, 0]]},
        "annotations": [
            {"positions": [1, 3], "kind": "stacking", "classification": ">>"},
            {"positions": [1, 2], "kind": "base_pair", "classification": "WC"},
        ],
    }
    monkeypatch.setattr(b, "geometry", lambda: geometry)
    card = b.tertiary(["M03"], mode="contacts", radius_A=3)[0]
    assert card["contacts"] == [{"positions": [1, 3], "minimum_heavy_atom_distance_A": 2.0}]
    assert card["mutated_site_distances"] == [
        {"positions": [1, 9], "minimum_heavy_atom_distance_A": 7.0}
    ]
    annot = b.tertiary(["M03"], mode="annotations")[0]
    assert len(annot["annotations"]) == 1
    assert annot["backbone_torsions_radians"]["1"]["alpha"] is None
    assert annot["backbone_torsions_radians"]["1"]["chi"] == 0.1
    absent = b.tertiary(["M02"], mode="contacts", positions=[2])[0]
    assert absent["status"] == "unavailable" and absent["withheld_positions"] == [2]


def test_single_and_structure_expert_have_identical_structural_permissions():
    a = {d["name"]: d for d in declarations("review_single")}
    b = {d["name"]: d for d in declarations("review_structure")}
    for name in ("rna_secondary", "rna_contacts", "rna_annotations"):
        assert a[name] == b[name]
    assert [d["name"] for d in declarations("review_sequence")] == ["evidence"]
    assert normalize_calls([{"name": "rna_contacts", "args": {}}], "review_sequence") is None
    assert (
        normalize_calls(
            [{"name": "rna_contacts", "args": {}}, {"name": "shell", "args": {}}],
            "review_structure",
        )
        is None
    )


def test_mapped_structure_requires_both_actual_geometry_queries(tmp_path, monkeypatch):
    t, s = cif_fixture(tmp_path / "wt.cif")
    g = RequestGateway(
        model_config={**MODEL, "research_tool_protocol": "native_structure_v3"},
        run_dir=tmp_path / "provider",
        fingerprint="geometry-tools",
        token_budget=None,
        budget_mode="unlimited",
    )
    tools = ActiveReviewTools(
        t,
        s,
        "review_structure",
        BoundedModel(g, {"role": "review_structure"}, 6),
        snapshot_path=tmp_path / "snapshot.json",
    )
    geometry = {
        "status": "available",
        "geometry_kind": "experimental_WT_construct",
        "coordinate_asset_sha256": "test",
        "positions": {str(i): {"backbone_angles_radians": [0.1] * 7} for i in range(1, 5)},
        "heavy_atoms": {str(i): [[float(i), 0, 0]] for i in range(1, 5)},
        "annotations": [{"positions": [1, 3], "kind": "stacking", "classification": ">>"}],
    }
    monkeypatch.setattr(tools.backend, "geometry", lambda: geometry)
    secondary = tools.call("rna_secondary", {})
    check = {
        "checks": [
            {
                "variant_id": "M01",
                "decision": "retain",
                "reason": "Computed comparison",
                "evidence_ids": [secondary["evidence"][0]["id"]],
            }
        ]
    }
    with pytest.raises(ValueError, match="contacts"):
        tools.validate_completion(check)
    contact = tools.call("rna_contacts", {})
    assert contact["evidence"][0]["contacts"]
    with pytest.raises(ValueError, match="annotations"):
        tools.validate_completion(check)
    annotations = tools.call("rna_annotations", {})
    assert annotations["evidence"][0]["annotations"]
    tools.validate_completion(check)
    assert g.summary()["provider_attempts"] == 0
    g.close()


def new_config(tmp_path):
    p = old_config(tmp_path)
    raw = json.loads(p.read_text())
    raw["analysis_profile"] = "tool_review_v3"
    raw["model_config"]["research_tool_protocol"] = "native_structure_v3"
    p.write_text(json.dumps(raw))
    return p


def native(actions):
    return {**response(""), "tool_calls": [{"name": name, "args": args} for name, args in actions]}


def mock(request):
    if request["metadata"].get("phase") != "analysis":
        return old_mock(request)
    history = json.loads(request["messages"][-1]["content"])
    role = request["metadata"]["role"]
    results = [x["tool_result"] for x in history if "tool_result" in x]
    if not results:
        actions = [("evidence", {"source": "sequence"})] if role != "review_structure" else []
        if role != "review_sequence":
            actions += [("rna_secondary", {"mode": "pairing"})]
        return native(actions)
    facts = [c for out in results for c in out["evidence"]]
    return response(
        json.dumps(
            {
                "checks": [
                    {
                        "variant_id": v["variant_id"],
                        "decision": "review",
                        "reason": "Review calculated candidate evidence.",
                        "evidence_ids": [
                            c["id"] for c in facts if c["variant_id"] == v["variant_id"]
                        ],
                    }
                    for v in history[0]["task"]["variants"]
                ],
                "limitations": ["Mock review"],
            }
        )
    )


def test_complete_mock_campaign_actual_tools_ledger_and_label_isolation(tmp_path):
    p = new_config(tmp_path)
    sent = []
    sample = json.loads((tmp_path / "sample.json").read_text())
    for row in sample["variants"]:
        row["score"] += 76543210
    (tmp_path / "sample.json").write_text(json.dumps(sample))

    def transport(r):
        assert "765432" not in json.dumps(r)
        sent.append(r)
        return mock(r)

    dry = run_campaign(p, dry_run=True)
    assert dry["initial_CR_units"] == 6 and dry["api_calls"] == 0
    report = run_campaign(p, transport=transport)
    assert report["status"] == "complete" and report["statuses"] == {"ok": 3}
    assert report["ledger"]["provider_attempts"] == len(sent)
    events = [
        json.loads(line)
        for f in (tmp_path / "out/tool_events").glob("*.jsonl")
        for line in f.read_text().splitlines()
    ]
    assert sum(e["status"] == "ok" and e["tool"] == "rna_secondary" for e in events) == 4
    assert any(e.get("computations") for e in events)
    assert list((tmp_path / "out/structure_tool_results").glob("*.json"))
    repeat = run_campaign(
        p, transport=lambda r: pytest.fail("A completed checkpoint made a request")
    )
    assert repeat["ledger"]["provider_attempts"] == len(sent)


def test_summary_only_reviewer_cannot_claim_completed_analysis(tmp_path):
    t, s = task_fixture(), snapshot_fixture()

    def summaries(r):
        history = json.loads(r["messages"][-1]["content"])
        if len(history) == 1:
            return native([("evidence", {"source": "structure"})])
        cards = next(x["tool_result"]["evidence"] for x in history if "tool_result" in x)
        return response(
            json.dumps(
                {
                    "checks": [
                        {
                            "variant_id": c["variant_id"],
                            "decision": "retain",
                            "reason": "Summary only",
                            "evidence_ids": [c["id"]],
                        }
                        for c in cards
                    ],
                    "limitations": [],
                }
            )
        )

    g = RequestGateway(
        model_config={**MODEL, "research_tool_protocol": "native_structure_v3"},
        run_dir=tmp_path / "provider",
        fingerprint="test",
        token_budget=None,
        budget_mode="unlimited",
        transport=summaries,
    )
    result = run_reviewer(
        t,
        s,
        "review_structure",
        g,
        {"assay": "synthetic"},
        tools_factory=factory(tmp_path / "snapshot.json"),
        system=SYSTEM,
    )
    assert result["status"] == "limit_or_invalid" and len(result["model_calls"]) == 6
    g.close()


def test_tool_unknown_usage_pauses_without_semantic_repair(tmp_path):
    p = new_config(tmp_path)

    def bad(r):
        return {} if r["metadata"].get("phase") == "analysis" else mock(r)

    result = run_campaign(p, transport=bad)
    assert result["status"] == "paused" and result["completed_units"] == 1
    assert result["ledger"]["provider_attempts"] == 2


def test_protocol_and_fingerprint_are_isolated(tmp_path):
    old = old_config(tmp_path)
    before, _ = prepare(old)
    new = new_config(tmp_path)
    after, _ = prepare(new)
    assert before["fingerprint"] != after["fingerprint"]
    raw = json.loads(new.read_text())
    raw["model_config"]["research_tool_protocol"] = "native_review_v2"
    new.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="protocol"):
        prepare(new)


def test_sdk_receives_versioned_structure_tools_without_hidden_calls(monkeypatch):
    from types import SimpleNamespace as NS

    import google.genai

    from evagent.integrated.gateway import google_stream_transport, public_model_config

    calls = []

    class Client:
        def __init__(self, **kwargs):
            self.models = self
            assert kwargs["http_options"].retry_options.attempts == 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate_content_stream(self, **kwargs):
            calls.append(kwargs["config"])
            assert kwargs["config"].automatic_function_calling.disable
            yield NS(
                usage_metadata=NS(
                    prompt_token_count=8,
                    candidates_token_count=2,
                    total_token_count=10,
                    thoughts_token_count=0,
                ),
                candidates=[
                    NS(
                        finish_reason="STOP",
                        content=NS(parts=[NS(text="OK", thought=False, function_call=None)]),
                    )
                ],
                prompt_feedback=None,
                model_version="test",
            )

    monkeypatch.setattr(google.genai, "Client", Client)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-offline")
    model = public_model_config({**MODEL, "research_tool_protocol": "native_structure_v3"})
    for phase, role in [
        ("analysis", "review_structure"),
        ("analysis", "review_single"),
        ("analysis", "review_sequence"),
        ("integration", "review_structure"),
    ]:
        google_stream_transport(
            {
                "model": "test",
                "model_config": model,
                "messages": [{"role": "user", "content": "test"}],
                "metadata": {"phase": phase, "role": role},
            }
        )
    assert {d.name for d in calls[0].tools[0].function_declarations} == {
        "evidence",
        "rna_secondary",
        "rna_contacts",
        "rna_annotations",
    }
    assert {d.name for d in calls[1].tools[0].function_declarations} == {
        d.name for d in calls[0].tools[0].function_declarations
    }
    assert {d.name for d in calls[2].tools[0].function_declarations} == {"evidence"}
    assert not calls[3].tools and calls[3].tool_config.function_calling_config.mode.value == "NONE"
