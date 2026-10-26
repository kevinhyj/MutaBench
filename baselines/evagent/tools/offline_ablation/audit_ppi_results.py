"""Audit frozen PPI artifacts and report unavailable components without imputation."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    freeze = read(output / 'freeze.json')
    for relative, expected in freeze['artifact_hashes'].items():
        assert sha(output / relative) == expected, relative
    manifest = read(output / 'execution_snapshot_manifest.json')
    core_path = next(path for path in manifest if path.endswith('/evagent/offline_ablation.py'))
    core_snapshot = manifest[core_path]['snapshot']
    assert sha(core_snapshot) == manifest[core_path]['sha256']
    spec = importlib.util.spec_from_file_location('frozen_ppi_audit_core', core_snapshot)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    origins = read(output / 'cache_sources.json')
    original_paths = {}
    for origin in origins:
        for name in ('source_task', 'source_request', 'source_response'):
            original_paths[origin[name]] = origin[name + '_sha256']
    for path, expected in original_paths.items():
        assert sha(path) == expected, path
    input_snapshot = read(output / 'input_snapshot.json')
    for entry in input_snapshot['inputs'].values():
        assert sha(entry['path']) == entry['sha256']
    root = Path(input_snapshot['inputs']['abagym']['path']).parent
    removal = read(root / 'field_removal_report.json')
    for entry in removal['datasets'].values():
        assert sha(entry['source']) == entry['source_sha256']
        assert sha(entry['output']) == entry['output_sha256']
    historical = read(output / 'historical_metrics_provenance.json')
    for entry in historical['sources'].values():
        assert sha(entry['path']) == entry['sha256']
    with (output / 'per_assay_metrics.csv').open() as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 88 * 7
    assert len({row['group_id'] for row in metrics}) == 88
    assert len({(row['group_id'], row['method']) for row in metrics}) == len(metrics)
    inventory = read(output / 'assay_inventory.json')
    reviewed = {row['group_id'] for row in inventory if row['inference_status'] == 'prepared'}
    with (output / 'rankings_and_components.csv').open() as handle:
        predictions = list(csv.DictReader(handle))
    assert all(row['group_id'] in reviewed for row in predictions)
    assert all(row['status'] != 'complete' for row in metrics if row['group_id'] not in reviewed)
    fusion_inputs = []
    for gid in sorted(reviewed):
        task = read(output / 'prepared' / f'{gid}.json')
        features = read(output / 'features' / f'{gid}.json')
        ids = [row['id'] for row in task['candidates']]
        gemini, sources = {}, {}
        for origin in (row for row in origins if row['group_id'] == gid):
            cached = read(output / 'cached_sequence' / f"{origin['unit_id']}.json")
            gemini.update(cached['scores'])
            sources.update({cid: origin['source_response'] for cid in cached['scores']})
        core.validate_scores(gemini, ids, bounded=True)
        msa = features['independent_msa_scores']
        complete = all(value is not None for value in msa.values())
        fused = {row['candidate_id']: row for row in core.fuse_rankings(ids, gemini, msa)} if complete else {}
        ranks = core.midranks([gemini[cid] for cid in ids])
        for index, cid in enumerate(ids):
            row = dict(group_id=gid, candidate_id=cid, gemini_score=gemini[cid], msa_score=msa[cid],
                       gemini_rank_percentile=(ranks[index] - 1) / (len(ids) - 1),
                       msa_rank_percentile=None, fusion_value=None, final_rank=None,
                       assay_full_MSA_coverage=complete, cache_source=sources[cid],
                       status='complete' if complete else 'assay_MSA_incomplete_no_subset_fusion')
            row.update(fused.get(cid, {}))
            fusion_inputs.append(row)
    with (output / 'fusion_input_coverage.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fusion_inputs[0]))
        writer.writeheader()
        writer.writerows(fusion_inputs)
    attempts = [read(path) for path in (output / 'attempts').glob('*/*.json')
                if not path.name.endswith('.started.json')]
    plan = read(output / 'plan.json')
    common_path = next(path for path in manifest if path.endswith('/complex_sequence_all_assays_N250_N10_seed42_20260928/common.py'))
    common_spec = importlib.util.spec_from_file_location('frozen_ppi_audit_parser', manifest[common_path]['snapshot'])
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    config = read(output / 'config.json')
    response_provenance = []
    for unit in plan:
        response_path = output / 'responses' / (unit['id'] + '.json')
        if not response_path.exists():
            continue
        response = read(response_path)
        assert response['request_sha256'] == unit['request_sha256']
        assert response['unit_id'] == unit['id']
        if response.get('status') != 'ok':
            continue
        core.validate_scores(response['scores'], unit['expected_ids'], bounded=True)
        reused = response.get('reused_corresponding_new_prompt_response')
        if reused:
            assert sha(reused['path']) == reused['sha256']
            original_new_response = read(reused['path'])
            assert original_new_response['request_sha256'] == reused['request_sha256'] == unit['request_sha256']
            assert original_new_response['scores'] == response['scores']
            assert sha(original_new_response['actual_prompt_path']) == unit['request_sha256']
        raw_path = Path(response['raw_attempt'])
        assert raw_path.resolve().is_relative_to(output / 'attempts')
        raw = read(raw_path)
        assert raw['http_status'] == 200
        assert raw['request_sha256'] == unit['request_sha256']
        assert raw['body']['modelVersion'] == response['model']
        assert response['model'] == config['requested_model'] or response['model'].startswith(config['requested_model'] + '-')
        parsed = common.parse(raw['body'], unit['expected_ids'])
        recovery = response.get('format_recovery')
        if recovery:
            assert parsed['status'] == raw['status'] == 'invalid_prediction'
            snapshot = Path(recovery['normalizer_snapshot'])
            assert snapshot.resolve().is_relative_to(output / 'execution_code_snapshot')
            assert sha(snapshot) == recovery['normalizer_sha256']
            repair_spec = importlib.util.spec_from_file_location('frozen_ppi_identifier_normalizer', snapshot)
            repair = importlib.util.module_from_spec(repair_spec)
            repair_spec.loader.exec_module(repair)
            normalized = repair.normalize_body(raw['body'], unit['expected_ids'])
            assert normalized['scores'] == response['scores']
            assert normalized['identifier_mappings'] == recovery['identifier_mappings']
            assert sha(recovery['original_failed_response']) == recovery['original_failed_response_sha256']
            for earlier_path in sorted(raw_path.parent.glob('*.json')):
                if earlier_path.name.endswith('.started.json') or int(earlier_path.stem) >= raw['attempt']:
                    continue
                earlier = read(earlier_path)
                if earlier.get('http_status') != 200:
                    continue
                try:
                    repair.normalize_body(earlier['body'], unit['expected_ids'])
                except ValueError:
                    continue
                raise AssertionError('Identifier recovery did not select the earliest eligible response')
        else:
            assert raw['status'] == parsed['status'] == 'ok' and parsed['scores'] == response['scores']
        response_provenance.append(dict(unit_id=unit['id'], request_sha256=unit['request_sha256'],
            response_path=str(response_path), response_sha256=sha(response_path),
            raw_attempt=str(raw_path), raw_attempt_sha256=sha(raw_path),
            reported_model=response['model'], endpoint=config['endpoint'], proxy=config.get('proxy'),
            identifier_prefix_recovery=bool(recovery), normalizer_sha256=recovery.get('normalizer_sha256') if recovery else None,
            reused_matching_new_prompt_response=bool(reused)))
    write(output / 'new_response_provenance.json', response_provenance)
    attempted_ids = {row['unit_id'] for row in attempts}
    complete_counts = Counter(row['method'] for row in metrics if row['status'] == 'complete')
    pending_methods = [method for method, expected in (
        ('msa_prompt', 21), ('gemini_structure', 21), ('msa_prompt_structure', 21),
        ('msa_gemini_rank5050_structure', 12)) if complete_counts[method] < expected]
    authentication_blocked = bool(attempts) and all(row['http_status'] == 403 and
        row['body'].get('error', {}).get('type') == 'gateway_error' and
        'API token required' in row['body'].get('error', {}).get('message', '') for row in attempts)
    quota_blocked = bool(attempts) and all(row['http_status'] == 402 and
        row['body'].get('error', {}).get('code') == 'insufficient_quota' for row in attempts)
    blocker = ('Current gateway credential rejected with HTTP 403: API token required (sk-*).'
               if authentication_blocked else
               'Authorized credential reached the gateway but HTTP 402 insufficient_quota prevents scoring.'
               if quota_blocked else None)
    write(output / 'completion_status.json', dict(
        total_assays=88, reviewed_assays=21, assays_outside_reviewed_scope=67,
        A=dict(cached_chunks=210, complete_fusion_assays=12, incomplete_MSA_assays=9),
        B=dict(serialized_reviewed_assays=21, new_scoring_complete_assays=complete_counts['msa_prompt']),
        C=dict(serialized_reviewed_assays=21, new_scoring_complete_assays=complete_counts['gemini_structure']),
        new_scoring=dict(planned_units=len(plan), actual_attempts=len(attempts),
                         attempted_units=len(attempted_ids), never_attempted_units=len(plan)-len(attempted_ids),
                         http_status_counts=dict(Counter(row['http_status'] for row in attempts)),
                         successful_units=sum(row['status']=='ok' for row in attempts)),
        blocker=blocker,
        blocker_kind='authentication' if authentication_blocked else 'quota' if quota_blocked else None,
        pending_conditions=pending_methods,
        do_not_retry_with_the_same_rejected_credential=authentication_blocked,
        do_not_retry_until_quota_available=quota_blocked))
    write(output / 'result_audit.json', dict(status='PASS',
        frozen_artifacts_checked=len(freeze['artifact_hashes']),
        unchanged_original_cache_request_task_files=len(original_paths),
        original_historical_metric_files_unchanged=len(historical['sources']),
        successful_new_responses_reparsed_and_verified=len(response_provenance),
        original_canonical_inputs_unchanged=True, new_inputs_unchanged=True,
        per_assay_rows=len(metrics), inventory_assays=88, reviewed_prediction_rows=len(predictions),
        reviewed_fusion_input_rows=len(fusion_inputs), unreviewed_candidate_rankings_generated=False,
        partial_MSA_not_imputed=True, auditor_sha256=sha(__file__)))
    print(json.dumps(read(output / 'result_audit.json'), indent=2))


if __name__ == '__main__':
    main()
