"""Keep successful matching new-prompt scores; enlarge only truncated output envelopes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-output-tokens', type=int, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if read(source / 'run_state.json')['status'] in ('running', 'stopping'):
        raise ValueError('Source scoring has not finished')
    model = read(source / 'google_model_metadata.json')
    if model['http_status'] != 200 or not 16384 < args.max_output_tokens <= model['metadata']['outputTokenLimit']:
        raise ValueError('Output limit exceeds the queried model limit')
    freeze = read(source / 'freeze.json')
    for relative, expected in freeze['artifact_hashes'].items():
        if sha(source / relative) != expected:
            raise ValueError('Source frozen inputs changed')
    output.mkdir(exist_ok=False)
    for relative in freeze['artifact_hashes']:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    shutil.copytree(source / 'execution_code_snapshot', output / 'execution_code_snapshot')
    plan = read(output / 'plan.json')
    reused, retry, preserved = [], [], []
    for unit in plan:
        previous_path = source / 'responses' / (unit['id'] + '.json')
        response = read(previous_path)
        if response['request_sha256'] != unit['request_sha256'] or response['unit_id'] != unit['id']:
            raise ValueError('Response and frozen request differ')
        target_path = output / 'responses' / previous_path.name
        if response['status'] == 'ok':
            copied = copy.deepcopy(response)
            original = dict(path=str(previous_path), sha256=sha(previous_path),
                            request_sha256=unit['request_sha256'], match='exact_new_prompt_candidates_order_and_generation_configuration')
            copied['reused_corresponding_new_prompt_response'] = original
            shutil.copytree(source / 'attempts' / unit['id'], output / 'attempts' / unit['id'])
            copied['raw_attempt'] = str(output / 'attempts' / unit['id'] / Path(response['raw_attempt']).name)
            if copied.get('format_recovery'):
                original_snapshot = Path(copied['format_recovery']['normalizer_snapshot'])
                copied['format_recovery']['normalizer_snapshot'] = str(output / 'execution_code_snapshot' / original_snapshot.name)
            write(target_path, copied)
            reused.append(dict(unit_id=unit['id'], **original))
            continue
        raw = read(response['raw_attempt'])
        candidates = raw.get('body', {}).get('candidates') or []
        truncated = (response['status'] == 'incomplete_response' and raw.get('http_status') == 200
                     and len(candidates) == 1 and candidates[0].get('finishReason') == 'MAX_TOKENS')
        if not truncated:
            # Do not turn refusals or unknown outcomes into a new route or protocol.
            copied = copy.deepcopy(response)
            shutil.copytree(source / 'attempts' / unit['id'], output / 'attempts' / unit['id'])
            copied['raw_attempt'] = str(output / 'attempts' / unit['id'] / Path(response['raw_attempt']).name)
            write(target_path, copied)
            preserved.append(unit['id'])
            continue
        request_path = output / unit['request_path']
        original_request = read(request_path)
        request = copy.deepcopy(original_request)
        request['generationConfig']['maxOutputTokens'] = args.max_output_tokens
        assert request['contents'] == original_request['contents']
        unchanged = copy.deepcopy(request)
        unchanged['generationConfig']['maxOutputTokens'] = original_request['generationConfig']['maxOutputTokens']
        assert unchanged == original_request
        write(request_path, request)
        old_hash = unit['request_sha256']
        unit.update(request_sha256=sha(request_path), semantic_request_sha256=digest(request),
                    prior_request_sha256=old_hash, completion_output_limit=args.max_output_tokens)
        retry.append(dict(unit_id=unit['id'], original_request=str(source / unit['request_path']),
                          original_request_sha256=old_hash, request_path=unit['request_path'],
                          request_sha256=unit['request_sha256'],
                          original_failed_response=str(previous_path), original_failed_response_sha256=sha(previous_path),
                          original_output_limit=original_request['generationConfig']['maxOutputTokens'],
                          new_output_limit=args.max_output_tokens, reason='MAX_TOKENS', scientific_prompt_unchanged=True))
    config = read(output / 'config.json')
    config['output_limit_completion_policy'] = dict(original_limit=16384, retry_limit=args.max_output_tokens,
        selector='prior MAX_TOKENS only; no labels or evaluation metrics used',
        generation_seed_temperature_and_MEDIUM_unchanged=True, per_unit_actual_configuration='requests/*.json')
    write(output / 'config.json', config)
    write(output / 'plan.json', plan)
    write(output / 'completion_protocol_amendment.json', dict(source_output=str(source),
        builder_path=str(Path(__file__).resolve()), builder_sha256=sha(__file__),
        reused_matching_new_prompt_responses=reused, retry_units=retry, other_failures_preserved=preserved,
        previous_outputs_unchanged=True, labels_read=False, N=250, CR=25, seed=42,
        model_limit_source=str(source / 'google_model_metadata.json'), model_limit_source_sha256=sha(source / 'google_model_metadata.json')))
    code_hashes = {**freeze['code_hashes'], str(Path(__file__).resolve()): sha(__file__)}
    artifact_hashes = {str(path.relative_to(output)): sha(path) for path in output.rglob('*')
                       if path.is_file() and 'execution_code_snapshot' not in path.parts}
    write(output / 'freeze.json', dict(artifact_hashes=artifact_hashes, code_hashes=code_hashes))
    print(json.dumps(dict(reused_corresponding_new_prompt_units=len(reused),
                         completion_retry_units=len(retry), other_failures_preserved=len(preserved),
                         new_output_limit=args.max_output_tokens, output=str(output))))


if __name__ == '__main__':
    main()
