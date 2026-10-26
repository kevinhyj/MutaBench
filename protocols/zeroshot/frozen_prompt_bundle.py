"""Portable, standard-library-only reader for exact frozen SYSTEM/USER messages.

Copy this file with a bundle as reader.py. No model SDK, credentials or labels.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def message_sha256(messages):
    return sha_bytes(json.dumps(messages, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'), allow_nan=False).encode('utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(bundle):
    bundle = Path(bundle).resolve()
    raw = (bundle / 'MANIFEST.json').read_bytes()
    require(sha_bytes(raw) == (bundle / 'MANIFEST.sha256').read_text().strip(), 'Manifest checksum changed')
    manifest = json.loads(raw)
    for relative, expected in manifest['files'].items():
        p = (bundle / relative).resolve()
        require(p.is_relative_to(bundle), 'File outside bundle')
        require(sha_bytes(p.read_bytes()) == expected, 'Frozen file changed: ' + relative)
    queue = json.loads((bundle / 'QUEUE.json').read_text())
    requests = [json.loads(line) for line in (bundle / 'REQUESTS.jsonl').read_text().splitlines()]
    require(len(queue) == len(requests) == manifest['queue_units'], 'Request count changed')
    indexed = {q['logical_id']: q for q in queue}
    require(len(indexed) == len(queue), 'Duplicate logical ID')
    require(len({r['request_id'] for r in requests}) == len(requests), 'Duplicate request ID')
    groups = {}
    for r in requests:
        q = indexed[r['request_id']]
        expected = [{'role':'system', 'content':q['template']['system']},
                    {'role':'user', 'content':q['template']['user']}]
        require(r['messages'] == expected, 'Messages differ from QUEUE')
        require(message_sha256(expected) == r['messages_sha256'], 'Message digest mismatch')
        require(sha_bytes(expected[0]['content'].encode()) == r['system_sha256'], 'SYSTEM digest mismatch')
        require(sha_bytes(expected[1]['content'].encode()) == r['user_sha256'], 'USER digest mismatch')
        require((bundle / 'SYSTEM.txt').read_bytes() == expected[0]['content'].encode(), 'SYSTEM file mismatch')
        require((bundle / r['user_path']).read_bytes() == expected[1]['content'].encode(), 'USER text file mismatch')
        require(json.loads((bundle / r['request_path']).read_text()) == r, 'Per-request JSON mismatch')
        require(r['expected_ids'] == q['template']['expected_ids'], 'Expected IDs changed')
        user = json.loads(expected[1]['content'])
        ids = [c['variant_id'] for c in user['candidates']]
        require(ids == r['expected_ids'] == user['output_contract']['expected_variant_ids'], 'Candidate ID/order mismatch')
        require(len(ids) == len(set(ids)) == 10, 'Frozen N10 changed')
        require(user['task']['assay_id'] == r['canonical_assay_id'], 'Canonical assay mismatch')
        require((r['domain'], r['assay'], r['arm']) == (q['domain'], q['assay'], q['arm']), 'Request identity mismatch')
        base = {k:v for k,v in user.items() if k != 'evidence'}
        groups.setdefault((r['domain'],r['assay']), []).append((r['arm'],base,ids))
    require(len(groups) == manifest['scope']['included_assays'], 'Assay count mismatch')
    for group in groups.values():
        require(len(group) == 4 and {g[0] for g in group} == set(manifest['arms']), 'Missing or duplicate arm')
        require(all(g[1:] == group[0][1:] for g in group), 'Four-arm base or candidate order mismatch')
    return manifest, requests


def load_requests(bundle, domain=None, arm=None):
    """Verify first, then return exact messages. Never parse/re-render USER for sending."""
    manifest, rows = verify(bundle)
    require(domain is None or domain in manifest['scope']['domains'], 'Unknown domain')
    require(arm is None or arm in manifest['arms'], 'Unknown arm')
    return [r for r in rows if (domain is None or r['domain'] == domain) and (arm is None or r['arm'] == arm)]


def check_actual_messages(request, actual_messages):
    """Call at a provider boundary after translating its native roles back to these roles."""
    require(actual_messages == request['messages'], 'Provider-visible message text/roles/order differ from frozen prompt')
    require(message_sha256(actual_messages) == request['messages_sha256'], 'Actual messages digest mismatch')
    return request['messages_sha256']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--domain', choices=['protein_dna','protein_ligand'])
    p.add_argument('--arm', choices=['sequence','msa_prompt','structure','msa_prompt_structure'])
    a = p.parse_args()
    rows = load_requests(a.bundle, a.domain, a.arm)
    print(json.dumps(dict(status='PASS', selected_requests=len(rows),
                          assays=len({(r['domain'],r['assay']) for r in rows}),
                          model_calls=0, labels_read=False)))


if __name__ == '__main__':
    main()
