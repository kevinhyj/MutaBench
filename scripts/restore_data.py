"""Restore verified data archives into a fresh pipeline-relative data tree."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tarfile

from download_data import sha, verify_artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Restore only into a fresh output directory')
    if not hasattr(tarfile,'data_filter'):parser.error('Use Python 3.12+ for safe tar extraction')
    release=verify_artifacts(args.download)
    records=[json.loads(line) for line in (args.download/'FILES.jsonl').read_text().splitlines()]
    expected={record['path']:record for record in records}
    if len(expected)!=len(records):raise ValueError('Duplicate archive path in manifest')
    args.output.mkdir(parents=True)
    count=0
    for index,record in enumerate(release['archives']):
        with tarfile.open(args.download/record['path'],'r:gz') as archive:
            members=archive.getmembers()
            names=[member.name for member in members]
            wanted={name for name,row in expected.items() if row['archive']==index}
            if len(set(names))!=len(names) or set(names)!=wanted:raise ValueError('Archive membership mismatch')
            for member in members:
                name=Path(member.name)
                if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.islnk()):
                    raise ValueError('Unsupported or unsafe archive member')
                if member.islnk():
                    target=Path(member.linkname)
                    if target.is_absolute() or '..' in target.parts or member.linkname not in wanted:
                        raise ValueError('Unsafe archive hardlink')
                    if expected[member.name]['sha256']!=expected[member.linkname]['sha256']:
                        raise ValueError('Hardlink checksum identities differ')
                elif member.size!=expected[member.name]['bytes']:raise ValueError('Archive size mismatch')
            archive.extractall(args.output,filter='data')
        for name in wanted:
            path=args.output/name;item=expected[name]
            if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
                raise ValueError('Restored file differs from frozen input: '+name)
            count+=1
    print(json.dumps(dict(status='PASS',files_verified=count,output=str(args.output),model_calls=0)))


if __name__=='__main__':main()
