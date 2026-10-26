import json

import pytest

from evagent.integrated.bundles import SCHEMA, FrozenEvidence, validate_bundle
from evagent.integrated.campaign import run_campaign
from evagent.integrated.evidence_tools import empty_source, msa_features
from evagent.integrated.gateway import digest


def test_query_only_alignment_is_unavailable(tmp_path):
    p = tmp_path/'query.a2m'
    p.write_text('>WT_query\nACGU\n')
    with pytest.raises(ValueError, match='Query-only'):
        msa_features('ACGU', [], p, {'origin': 'natural_homologs', 'reviewed': True})


def test_artifact_change_and_escape_rejected(tmp_path):
    from evagent.integrated.bundles import file_hash
    _, _, _, data = fixture_bundle(tmp_path)
    p = tmp_path/'raw.txt'
    p.write_text('frozen')
    data['sources']['structure']['provenance'][0].update(artifact=p.name, sha256=file_hash(p))
    data['sha256'] = digest({k: v for k, v in data.items() if k != 'sha256'})
    validate_bundle(data, root=tmp_path)
    p.write_text('changed')
    with pytest.raises(ValueError, match='artifact'):
        validate_bundle(data, root=tmp_path)


@pytest.mark.parametrize('bad', ['doi', 'title', 'effect'])
def test_literature_target_interception(tmp_path, bad):
    from evagent.integrated.evidence_tools import prepare_bundle
    task = save(tmp_path/'task.json', {'assay': 'synthetic', 'wt_sequence': 'ACGU', 'variants': []})
    record = {'url': 'https://example.org/paper', 'date': '2000-01-01', 'background_only': True,
              'summary': 'RNA background', 'doi': 'safe', 'title': 'safe'}
    record.update({'doi': 'denied'} if bad == 'doi' else {'title': 'Denied, title!'} if bad == 'title' else {'direct_variant_effects': True})
    lit = save(tmp_path/'lit.json', {'assay': 'synthetic', 'reviewed': True, 'records': [record]})
    with pytest.raises(ValueError, match='prohibited'):
        prepare_bundle({'task_file': str(task), 'source_audit': {'status': 'admitted'},
            'literature_file': str(lit), 'denied_dois': ['denied'], 'denied_titles': ['Denied title']},
            base=tmp_path, output=tmp_path/'bundle.json')


def test_cif_mapping_rejects_homolog_and_accepts_wt(tmp_path):
    pytest.importorskip('gemmi')
    from evagent.integrated.structure_tools import cif_features
    p = tmp_path/'wt.cif'
    text = '''data_test
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
'''
    text += ''.join(f'1 A {i} P {i} 0 0 80 1\n' for i in range(1, 5))
    p.write_text(text)
    record = cif_features(p, 'ACGU', predicted=True, provider='fixture', version='1')
    assert record['task']['coordinate_coverage'] == 1
    assert record['positions']['1']['plddt'] == 80
    with pytest.raises(ValueError, match='mapping'):
        cif_features(p, 'ACGA', predicted=False, provider='fixture', version='1')
    # ModelCIF (AF3) can omit the canonical sequence column and instead declare
    # the polymer through entity_poly_seq. Modified chemistry must still fail.
    modelcif = text.replace('_entity_poly.pdbx_seq_one_letter_code_can\n1 polyribonucleotide ACGU',
                           '1 polyribonucleotide')
    modelcif += 'loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n'
    modelcif += ''.join(f'1 {i} {base}\n' for i, base in enumerate('ACGU', 1))
    p.write_text(modelcif)
    assert cif_features(p, 'ACGU', predicted=True, provider='AF3', version='test')['task']['coordinate_coverage'] == 1
    p.write_text(modelcif.replace('1 4 U', '1 4 DU'))
    with pytest.raises(ValueError, match='mapping'):
        cif_features(p, 'ACGU', predicted=False, provider='fixture', version='1')


def save(path, data):
    path.write_text(json.dumps(data))
    return path


def fixture_bundle(tmp_path):
    wt = "ACGU"
    variants = [{"variant_id": "M01", "sequence": "GCGU", "mutation": "A1G"},
                {"variant_id": "M02", "sequence": "AUGU", "mutation": "C2U"}]
    sources = {s: empty_source("missing") for s in ("msa", "structure", "literature")}
    sources["structure"] = {"status": "available", "task": {"temperature": 37},
        "positions": {"1": {"value": .1}, "2": {"value": .2}},
        "candidates": {digest(v["sequence"]): {"changed_positions": [i+1]} for i, v in enumerate(variants)},
        "candidate_specific": True, "provenance": [{"id": "test", "tool": "fixture", "version": "1", "sha256": "test"}]}
    data = {"schema": SCHEMA, "assay": "synthetic", "wt_sha256": digest(wt),
            "source_status": "admitted", "sources": sources}
    data["sha256"] = digest(data)
    return wt, variants, save(tmp_path/'bundle.json', data), data


def test_render_only_current_chunk(tmp_path):
    wt, variants, p, _ = fixture_bundle(tmp_path)
    view = FrozenEvidence(p, assay="synthetic", wt_sequence=wt, variants=variants, sources=["structure"])
    text = view.render(["M01"])
    assert 'M01' in text and 'M02' not in text and '"2":' not in text


def test_bundle_corruption_and_label_rejected(tmp_path):
    _, _, _, data = fixture_bundle(tmp_path)
    data["sources"]["structure"]["task"]["temperature"] = 38
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(data)
    data["experimental_score"] = 1
    data["sha256"] = digest({k: v for k, v in data.items() if k != "sha256"})
    with pytest.raises(ValueError, match="Forbidden"):
        validate_bundle(data)


def test_wrong_wt_and_missing_candidate_rejected(tmp_path):
    wt, variants, p, _ = fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="WT"):
        FrozenEvidence(p, assay="synthetic", wt_sequence="ACGA", variants=variants, sources=["structure"])
    variants[0]["sequence"] = "UUUU"
    with pytest.raises(ValueError, match="candidate"):
        FrozenEvidence(p, assay="synthetic", wt_sequence=wt, variants=variants, sources=["structure"])


def test_msa_requires_review_and_exact_mapping(tmp_path):
    p = tmp_path/'msa.fa'
    p.write_text('>wt\nACGU\n>natural1\nACGA\n>natural2\nACGG\n')
    variants = [{"sequence": "GCGU"}]
    with pytest.raises(ValueError, match="natural homologs"):
        msa_features("ACGU", variants, p, {"origin": "candidate_library", "reviewed": True})
    value = msa_features("ACGU", variants, p, {"origin": "natural_homologs", "reviewed": True})
    assert value["positions"]["1"]["support"] == "low"
    assert value["candidates"][digest("GCGU")]["changed_positions"] == [1]
    with pytest.raises(ValueError, match="exact WT"):
        msa_features("UUUU", variants, p, {"origin": "natural_homologs", "reviewed": True})


def campaign_fixture(tmp_path, budget=100000):
    wt, variants, p, _ = fixture_bundle(tmp_path)
    sample = save(tmp_path/'sample.json', {"assay": "synthetic", "variants": [{**v, "score": i} for i, v in enumerate(variants)]})
    context = save(tmp_path/'context.json', {"assay": "synthetic", "wt_sequence": wt, "metadata": {"DMS_ID": "synthetic"}})
    model = {"model": "test", "thinking_level": "medium", "max_tokens": 256, "timeout_s": 30}
    cells = []
    for condition in ("cr", "structure"):
        cfg = {"method": "cr", "backend": "direct", **model, "sample_file": str(sample),
               "context_file": str(context), "output_dir": str(tmp_path/condition), "token_budget": budget,
               "group_size": 2, "anchor_count": 0}
        if condition == "structure":
            cfg.update(evidence_mode="bundle", evidence_bundle_file=str(p), evidence_sources=["structure"])
        cp = save(tmp_path/f'{condition}.json', cfg)
        cells.append({"assay": "synthetic", "n": 2, "condition": condition, "family": "fixture",
                      "status": "eligible", "source_status": "admitted", "config": str(cp)})
    return save(tmp_path/'campaign.json', {"schema": "evagent-campaign-v1", "output_dir": str(tmp_path/'campaign'),
                 "token_budget": budget, "model_config": model, "cells": cells})


def fake_provider(request):
    ids = request["metadata"].get("expected_ids")
    text = json.dumps({"variants": [{"variant_id": v, "predicted_score": float(i)} for i, v in enumerate(ids)]}) if ids else "OK"
    return {"final_text": text, "finish_reason": "STOP",
            "usage_summary": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "model_calls": 1}}


def test_campaign_dry_run_no_writes_or_calls(tmp_path):
    path = campaign_fixture(tmp_path)
    result = run_campaign(path, dry_run=True)
    assert result["eligible"] == 2 and result["writes"] == 0
    assert not (tmp_path/'campaign').exists()


def test_campaign_accounting_and_restart(tmp_path):
    path = campaign_fixture(tmp_path)
    calls = []
    def transport(request):
        calls.append(request)
        return fake_provider(request)
    result = run_campaign(path, transport=transport)
    assert result["status"] == "complete"
    assert result["complete_cells"] == 2
    assert len(calls) == result["ledger"]["provider_attempts"] == 3
    again = run_campaign(path, transport=transport)
    assert len(calls) == 3 and again["ledger"]["reported_tokens"] == 6


def test_campaign_budget_stops_before_next_call(tmp_path):
    path = campaign_fixture(tmp_path, budget=2000)
    calls = []
    def transport(request):
        calls.append(request)
        value = fake_provider(request)
        value['usage_summary']['total_tokens'] = 1900
        return value
    result = run_campaign(path, transport=transport)
    assert result["status"] == "paused"
    assert len(calls) == 1 and result["ledger"]["reported_tokens"] == 1900
    assert result["statuses"]["not_run"] == 1


def test_campaign_unknown_response_never_retried(tmp_path):
    path = campaign_fixture(tmp_path)
    calls = []
    def transport(request):
        calls.append(request)
        return fake_provider(request) if len(calls) == 1 else {"final_text": "unclassified"}
    result = run_campaign(path, transport=transport)
    assert result["status"] == "paused" and len(calls) == 2
    run_campaign(path, transport=transport)
    assert len(calls) == 2


def test_campaign_freezes_all_inputs(tmp_path):
    path = campaign_fixture(tmp_path)
    run_campaign(path, transport=fake_provider)
    config = json.loads((tmp_path/'cr.json').read_text())
    config['max_repairs'] = 0
    save(tmp_path/'cr.json', config)
    with pytest.raises(ValueError, match="changed"):
        run_campaign(path, transport=fake_provider)


def test_campaign_rejects_config_change_after_preflight(tmp_path):
    path = campaign_fixture(tmp_path)
    calls = []
    def transport(request):
        calls.append(request)
        if len(calls) == 1:
            config_path = tmp_path/'cr.json'
            config = json.loads(config_path.read_text())
            config['max_repairs'] = 0
            save(config_path, config)
        return fake_provider(request)
    result = run_campaign(path, transport=transport)
    assert result['status'] == 'paused'
    assert len(calls) == 1  # only connectivity; no changed cell submitted
    assert result['ledger']['paused']


def test_prior_calibration_cost_is_carried_once(tmp_path):
    from evagent.integrated.bundles import file_hash
    from evagent.integrated.gateway import RequestGateway
    g = RequestGateway(model_config={'model': 'test', 'thinking_level': 'medium'},
        run_dir=tmp_path/'prior', fingerprint='prior', token_budget=10000, transport=fake_provider)
    g.generate(system='connectivity', user='OK', run_name='prior')
    p = g.path
    g.close()
    campaign = campaign_fixture(tmp_path)
    raw = json.loads(campaign.read_text())
    raw['prior_ledgers'] = [{'path': str(p), 'sha256': file_hash(p)}]
    save(campaign, raw)
    result = run_campaign(campaign, transport=fake_provider)
    assert result['ledger']['reported_tokens'] == 8
    assert result['ledger']['current_run_reported_tokens'] == 6
    assert result['ledger']['prior_reported_tokens'] == 2
    assert result['ledger']['provider_attempts'] == 4
    assert run_campaign(campaign, transport=fake_provider)['ledger']['reported_tokens'] == 8


def test_unknown_prior_ledger_cannot_start_new_campaign(tmp_path):
    from evagent.integrated.bundles import file_hash
    from evagent.integrated.campaign import read_prior_ledgers
    from evagent.integrated.gateway import ExecutionPaused, RequestGateway
    g = RequestGateway(model_config={'model': 'test', 'thinking_level': 'medium'},
        run_dir=tmp_path/'prior', fingerprint='prior', token_budget=10000, transport=lambda _: {})
    with pytest.raises(ExecutionPaused):
        g.generate(system='connectivity', user='OK', run_name='prior')
    p = g.path
    g.close()
    with pytest.raises(ValueError, match='Unknown prior'):
        read_prior_ledgers([{'path': str(p), 'sha256': file_hash(p)}], tmp_path)
