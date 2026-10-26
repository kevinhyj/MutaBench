"""Download the pinned HF dataset release and verify every distributable artifact."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def verify_artifacts(directory, tables_only=False):
    release=json.loads((directory/'RELEASE.json').read_text())
    if release.get('schema')!='mutabench-data-release-v1':raise ValueError('Unknown release schema')
    records=release['manifests']+release['tables']+([] if tables_only else release['archives'])
    for record in records:
        relative=Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe artifact path')
        path=directory/relative
        if not path.is_file() or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Artifact checksum mismatch: '+str(relative))
    return release


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tables-only',action='store_true',help='Fetch query/label tables and manifests without large archives')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    binding=json.loads((root/'benchmark/data_release.json').read_text())
    if binding.get('status')!='available' or not binding.get('repo_id') or not binding.get('revision'):
        parser.error('The pinned dataset release is not available in this code revision')
    if args.output.exists():parser.error('Choose a fresh output directory')
    from huggingface_hub import snapshot_download
    patterns=['README.md','SOURCES.md','RELEASE.json','PLAN.json','FILES.jsonl','tables/*']
    if not args.tables_only:patterns.append('archives/*')
    snapshot_download(repo_id=binding['repo_id'],repo_type='dataset',revision=binding['revision'],
                      local_dir=args.output,allow_patterns=patterns)
    if sha(args.output/'RELEASE.json')!=binding['release_manifest_sha256']:
        raise ValueError('Pinned release manifest checksum mismatch')
    release=verify_artifacts(args.output,args.tables_only)
    print(json.dumps(dict(status='PASS',repo_id=binding['repo_id'],revision=binding['revision'],
                         archives=0 if args.tables_only else len(release['archives']),tables=release['table_summary'])))


if __name__=='__main__':main()
