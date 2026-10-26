"""Synthetic cache integrity and missing-component tests; no model calls."""
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / 'tools/offline_ablation'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('platinum_ablation_test_runner', DIRECTORY / 'platinum.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_missing_msa_preserves_all_components_without_subset_ranking():
    scores = dict(a=-.2, b=.3, c=.3)
    features = dict(independent_msa_scores=dict(a=2., b=None, c=-2.), full_msa_coverage=False)
    result = runner.fusion_artifact(list(scores), scores, features,
                                   dict(source_response='old.json', source_response_sha256='sha'))
    assert result['status'] == 'incomplete_MSA'
    assert result['missing_candidate_ids'] == ['b']
    assert [r['msa_score'] for r in result['rows']] == [2., None, -2.]
    assert [r['gemini_rank_percentile'] for r in result['rows']] == [0., .75, .75]
    assert all(r['msa_rank_percentile'] is None and r['final_rank'] is None for r in result['rows'])
    assert result['new_calls'] == 0


def test_structure_score_fusion_is_separate_from_sequence_cache_fusion():
    ids = ['a', 'b']
    features = dict(independent_msa_scores=dict(a=1., b=0.), full_msa_coverage=True)
    origin = dict(source_response='structure.json', source_response_sha256='sha')
    sequence = runner.fusion_artifact(ids, dict(a=1., b=0.), features, origin)
    structure = runner.fusion_artifact(ids, dict(a=0., b=1.), features, origin,
                                       condition='postprocessing_of_structure_scores')
    assert sequence['condition'] == 'postprocessing_only'
    assert structure['condition'] == 'postprocessing_of_structure_scores'
    assert structure['new_calls'] == 0
    assert sequence['rows'][0]['fusion_value'] != structure['rows'][0]['fusion_value']
    assert structure['rows'][0]['gemini_score'] == 0
    assert structure['rows'][0]['msa_score'] == 1


def test_structure_prompt_only_contains_relevant_evidence():
    text = runner.prompt_user(dict(user='base unchanged'), dict(structure_sites={}))
    assert text.startswith('base unchanged\n')
    assert 'MSA probabilities' not in text
    assert 'unavailable, never zero' in text


def test_prepare_refuses_to_overwrite_existing_results(tmp_path):
    p = tmp_path / 'old.json'
    p.write_text('preserved')
    with pytest.raises(ValueError, match='new empty output'):
        runner.prepare(tmp_path)
    assert p.read_text() == 'preserved'


def test_credentials_reject_ambiguous_or_public_file(tmp_path):
    path = tmp_path / 'private.env'
    path.write_text("GEMINI_API_KEY='synthetic-token'\n")
    path.chmod(0o600)
    assert runner.credential_value(path) == 'synthetic-token'
    path.write_text('GEMINI_API_KEY=one\nGEMINI_API_KEY=two\n')
    with pytest.raises(ValueError, match='ambiguous'):
        runner.credential_value(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match='permissions'):
        runner.credential_value(path)


def cache_fixture(root):
    plan = dict(model_config=dict(model='synthetic-gemini', generation_seed=42), fingerprint='freeze')
    request = dict(assay_id='synthetic_assay', condition='msa_prompt', expected_ids=['M001', 'M002'],
                   system='system', user='actual new prompt')
    request['request_hash'] = runner.digest(request)
    unit = {k: request[k] for k in ('request_hash', 'assay_id', 'condition')}
    wire = dict(model_config=plan['model_config'], fingerprint='freeze',
                messages=[dict(role='system', content=request['system']), dict(role='user', content=request['user'])],
                metadata=dict(request_hash=unit['request_hash'], assay_id=unit['assay_id'], condition=unit['condition']))
    wire_hash = runner.digest(wire)
    text = json.dumps(dict(variants=[dict(variant_id='M001', predicted_score=.2),
                                    dict(variant_id='M002', predicted_score=-.3)]))
    raw = dict(final_text=text, response_metadata=dict(model_version='synthetic-gemini-001'))
    attempt = dict(request_sha256=wire_hash, final_text=text,
                   attempt_prompt_sha256=hashlib.sha256(request['user'].encode()).hexdigest())
    result = dict(request_hash=unit['request_hash'], assay_id=unit['assay_id'], condition=unit['condition'],
                  expected_ids=request['expected_ids'], model_config=plan['model_config'], fingerprint='freeze',
                  status='ok', value=dict(M001=.2, M002=-.3), attempts=[attempt])
    runner.save(root / 'requests' / (unit['request_hash'] + '.json'), request)
    result_path = root / 'results' / (unit['request_hash'] + '.json')
    runner.save(result_path, result)
    ledger = root / 'provider/request_ledger.sqlite'
    ledger.parent.mkdir()
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE attempts(request_sha256 TEXT,request_json TEXT,response_json TEXT,status TEXT)')
        db.execute('INSERT INTO attempts VALUES (?,?,?,?)', (wire_hash, json.dumps(wire), json.dumps(raw), 'ok'))
    return plan, unit, result_path, ledger


def test_changed_prompt_cache_validates_actual_wire_and_raw_scores(tmp_path):
    plan, unit, _, _ = cache_fixture(tmp_path)
    assert runner.validated_result(tmp_path, unit, plan)['value'] == dict(M001=.2, M002=-.3)


@pytest.mark.parametrize('field', ['request_hash', 'assay_id', 'condition', 'expected_ids', 'model_config', 'value'])
def test_changed_prompt_cache_rejects_tampered_result(tmp_path, field):
    plan, unit, path, _ = cache_fixture(tmp_path)
    result = runner.load(path)
    result[field] = dict(M001=.4, M002=-.3) if field == 'value' else 'changed'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        runner.validated_result(tmp_path, unit, plan)


def test_changed_prompt_cache_rejects_altered_actual_provider_request(tmp_path):
    plan, unit, _, ledger = cache_fixture(tmp_path)
    with sqlite3.connect(ledger) as db:
        wire = json.loads(db.execute('SELECT request_json FROM attempts').fetchone()[0])
        wire['messages'][1]['content'] = 'old prompt'
        db.execute('UPDATE attempts SET request_json=?', (json.dumps(wire),))
    with pytest.raises(ValueError, match='provider request differs'):
        runner.validated_result(tmp_path, unit, plan)


@pytest.mark.parametrize('partner,incomplete', [(True, False), (True, True), (False, False)])
def test_ligand_geometry_uses_only_annotated_copies_and_preserves_missing(tmp_path, partner, incomplete):
    from test_offline_ablation_features import FrozenFeatureTests
    from ligand_features import LigandFeatureBuilder
    builder, task, observed, _ = FrozenFeatureTests().fixture(tmp_path, incomplete=incomplete)
    builder.__class__ = LigandFeatureBuilder
    path = Path(observed['structure_source']['path'])
    lines = path.read_text().splitlines()[:-1]
    if partner:
        lines += ['HETATM  201   C1 LIG B  10       4.000   0.000   0.000  1.00 10.00           C',
                  'HETATM  202   C1 LIG D  11       6.000   0.000   0.000  1.00 10.00           C']
    lines += ['HETATM  203   C1 COF C  12       0.100   0.000   0.000  1.00 10.00           C', 'END']
    path.write_text('\n'.join(lines) + '\n')
    task.update(wt_chains=observed['wt_chains'][:1],
                structure_source=dict(path=str(path), sha256=runner.file_hash(path)),
                binding_partner=dict(pdb_ligand_id='LIG'))
    features, mapping = builder.build_ligand(task)
    block = runner.render_evidence(task, ['one', 'two'], features, 'msa_prompt_structure')
    row = block['structure_sites']['one'][0]
    assert row['site_id'] == block['msa_sites']['one'][0]['site_id']
    assert len(mapping['sites']) == 1
    if partner:
        assert row['min_partner_heavy_distance_A'] == 1
        assert row['partner_contact_count_5A'] == 2
        assert row['partner_chain_ids'] == ['B', 'D']
        assert len(features['structure_definition']['partner_instances']) == 2
        assert row['sasa_status'] == ('missing_WT_heavy_atoms' if incomplete else 'ok')
    else:
        assert row['partner_status'] == 'missing_annotated_ligand'
        assert all(row[k] is None for k in runner.render_evidence.__globals__['STRUCTURE_FIELDS'][:4])
    if incomplete:
        assert row['sasa_bound_A2'] is None and row['delta_sasa_partner_A2'] is None
