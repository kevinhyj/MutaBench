"""Recover only omitted literal v_ prefixes in completed, label-free model output."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def normalize_body(body, ids):
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r'v_[0-9a-f]{20}', cid) for cid in ids):
        raise ValueError('Expected IDs do not follow the frozen prefix convention')
    candidates = body.get('candidates') or []
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('Cannot normalize an incomplete or refused response')
    text = ''.join(part.get('text', '') for part in candidates[0].get('content', {}).get('parts', [])
                   if not part.get('thought'))
    expected = set(ids)
    for match in reversed(list(re.finditer(r'\{\s*"variants"\s*:', text))):
        try:
            obj, end = json.JSONDecoder().raw_decode(text[match.start():])
            if text[match.start() + end:].strip().strip('`').strip():
                continue
            rows = obj['variants']
            if len(rows) != len(ids):
                continue
            scores, mappings = {}, []
            for row in rows:
                original, value = row['variant_id'], row['predicted_score']
                if not isinstance(original, str):
                    raise ValueError('Invalid identifier')
                canonical = original
                if original not in expected:
                    if not re.fullmatch(r'[0-9a-f]{20}', original) or 'v_' + original not in expected:
                        raise ValueError('No exact unique prefix-only identity mapping')
                    canonical = 'v_' + original
                    mappings.append(dict(original_id=original, canonical_id=canonical))
                if canonical in scores or type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
                    raise ValueError('Duplicate, missing or invalid prediction')
                scores[canonical] = value
            if set(scores) != expected or not mappings:
                continue
            return dict(scores=scores, identifier_mappings=mappings,
                        rule='Only prepend literal v_ to an exact 20-character hexadecimal ID in the frozen candidate set; scores unchanged')
        except (ValueError, TypeError, KeyError):
            continue
    raise ValueError('Response cannot be uniquely normalized by the prefix-only rule')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    read = lambda path: json.loads(Path(path).read_text())
    config = read(output / 'config.json')
    state = read(output / 'run_state.json')
    if state['status'] in ('running', 'stopping'):
        raise ValueError('Wait for all in-flight requests before recovering formats')
    for relative, expected in read(output / 'freeze.json')['artifact_hashes'].items():
        if sha(output / relative) != expected:
            raise ValueError('Frozen scientific input changed')
    code_snapshot = output / 'execution_code_snapshot' / (sha(__file__) + '.py')
    if not code_snapshot.exists():
        with code_snapshot.open('xb') as handle:
            handle.write(Path(__file__).read_bytes())
    recovered, unavailable = [], []
    units = {unit['id']: unit for unit in read(output / 'plan.json')}
    for unit_id, unit in units.items():
        path = output / 'responses' / (unit_id + '.json')
        if not path.exists():
            continue
        previous = read(path)
        if previous.get('status') != 'invalid_prediction':
            continue
        chosen = None
        for attempt in sorted((output / 'attempts' / unit_id).glob('*.json')):
            if attempt.name.endswith('.started.json'):
                continue
            raw = read(attempt)
            if raw.get('http_status') != 200 or raw.get('request_sha256') != unit['request_sha256']:
                continue
            model = raw.get('body', {}).get('modelVersion', '')
            if not (model == config['requested_model'] or model.startswith(config['requested_model'] + '-')):
                continue
            try:
                normalized = normalize_body(raw['body'], unit['expected_ids'])
            except ValueError:
                continue
            chosen = (attempt, raw, normalized)
            break  # Earliest eligible output, independent of all evaluation labels.
        if chosen is None:
            unavailable.append(unit_id)
            continue
        attempt, raw, normalized = chosen
        archive = output / 'identifier_format_failures' / path.name
        archive.parent.mkdir(exist_ok=True)
        if archive.exists():
            raise ValueError('Never overwrite an original format-failure record')
        path.rename(archive)
        response = dict(unit_id=unit_id, group_id=unit['group_id'], condition=unit['condition'],
            status='ok', scores=normalized['scores'], request_sha256=unit['request_sha256'],
            model=raw['body']['modelVersion'], attempt=raw['attempt'], usage=raw['body'].get('usageMetadata'),
            actual_prompt_path=str(output / unit['request_path']), raw_attempt=str(attempt), finished=time.time(),
            format_recovery=dict(rule=normalized['rule'], identifier_mappings=normalized['identifier_mappings'],
                original_failed_response=str(archive), original_failed_response_sha256=sha(archive),
                normalizer_snapshot=str(code_snapshot), normalizer_sha256=sha(code_snapshot),
                selected_attempt_policy='earliest eligible completed response; no labels read',
                raw_response_and_numeric_scores_unchanged=True))
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        recovered.append(dict(unit_id=unit_id, **response['format_recovery']))
    report = dict(recovered_units=recovered, unavailable_units=unavailable, new_calls=0,
                  evaluation_labels_read=False, scientific_prompt_and_generation_configuration_unchanged=True)
    (output / 'identifier_format_recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(recovered_units=len(recovered), unavailable_units=len(unavailable), new_calls=0)))


if __name__ == '__main__':
    main()
