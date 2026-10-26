"""Explicit one-time launch; refuses active V4 and existing launch receipts."""
import datetime,fcntl,json,os,subprocess,sys
from pathlib import Path
import probe
project=Path(__file__).resolve().parents[3]
source=project/'experiments/cr_ncrna_evidence_b_repro_v4'
root=project/'experiments/cr_ncrna_simple_structure_v5'
launch_lease=Path(__file__).with_name('launch.lock').open('a+')
fcntl.flock(launch_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (root/'launch.json').exists():raise SystemExit('Launch receipt exists; inspect progress/process before any manual recovery')
if not (root/'campaign.json').exists():probe.freeze(root,source)
probe.run(root/'campaign.json',dry=True)
if probe.r.load(source/'progress.json')['status']!='complete':raise SystemExit('V4 is not complete')
if (root/'provider/PAUSE').exists():raise SystemExit('Operator pause')
log=(root/'run.log').open('ab',buffering=0)
command=[sys.executable,'-u',str(Path(__file__).with_name('probe.py')),'run','--root',str(root)]
p=subprocess.Popen(command,cwd=project,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
probe.r.save(root/'launch.json',{'pid':p.pid,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'campaign_sha256':probe.r.file_hash(root/'campaign.json')})
print(json.dumps({'status':'launched','pid':p.pid,'root':str(root)}))
