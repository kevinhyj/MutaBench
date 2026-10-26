"""Run the finite queue endpoint by endpoint and evaluate in separate processes."""
import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    args=p.parse_args()
    root=args.root
    queue=json.loads((root/'queue.json').read_text())
    endpoints=list(dict.fromkeys(u['endpoint'] for u in queue))
    for ep in endpoints:
        pending=sum(not (root/'units'/f'{u["unit_id"]}.json').exists() for u in queue if u['endpoint']==ep)
        if pending:
            print(f'Running endpoint {ep}: {pending} remaining units',flush=True)
            subprocess.run([sys.executable,'-I','-m','evagent.admet.campaign','run','--root',str(root),'--max-units',str(pending)],check=True)
        subprocess.run([sys.executable,'-I','-m','evagent.admet.report','--root',str(root),'--data-root',str(args.data_root)],check=True)
    records=[json.loads(p.read_text()) for p in (root/'units').glob('*.json')]
    state='complete' if len(records)==len(queue) and all(r['status']=='ok' for r in records) else 'finished_with_invalid_units'
    receipt={'state':state,'utc':datetime.now(timezone.utc).isoformat(),'planned_units':len(queue),'finished_units':len(records),'unit_statuses':dict(Counter(r['status'] for r in records))}
    (root/'DONE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status=json.loads((root/'STATUS.json').read_text())
    status.update(state=state, updated_utc=receipt['utc'])
    (root/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
