"""Native GEMME Java/R stages in a pinned, isolated CPU container."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from .common import ROOT, sha, load, freeze, mutations

IMAGE_DIGEST='sha256:a35099f7dee9411842e3e04318f4a9b435541bce16aff1c9ef5baa507baf5e7d'


def predict(item, input_path, out):
    if item['modality']!='protein':
        raise ValueError('GEMME requires protein alignment')
    image=ROOT/'assets/gemme.sif'
    image_card=load(ROOT/'assets/gemme_manifest.json')
    if image_card['image_sha256']!=sha(image) or IMAGE_DIGEST not in image_card['image_source']:
        raise ValueError('GEMME container identity changed')
    evidence=item['evidence']['msa'];msa=Path(evidence['path'])
    if sha(msa)!=evidence['sha256']:
        raise ValueError('MSA hash mismatch')
    sequences=[]
    for line in msa.read_text().splitlines():
        if line.startswith('>'):sequences.append('')
        else:sequences[-1]+=line.strip().upper().replace('.','-')
    wt=item['base']
    if sequences[0]!=wt or any(len(s)!=len(wt) for s in sequences):
        raise ValueError('GEMME adapter requires full-length aligned query coordinates')
    ambiguous=sum(bool(set(s)-set('ACDEFGHIKLMNPQRSTVWY-')) for s in sequences[1:])
    sequences=[sequences[0]]+[s for s in sequences[1:] if not set(s)-set('ACDEFGHIKLMNPQRSTVWY-')]
    if len(sequences)>20000:
        raise ValueError('More than 20000 MSA rows; explicit subsampling protocol required')
    out=Path(out).resolve();work=out/'native';work.mkdir(parents=True,exist_ok=False)
    # Names carry full query coordinates; only headers are normalized.
    (work/'alignment.fasta').write_text(''.join(f'>{"AMFR" if i==0 else "H"+str(i)}/1-{len(wt)}\n{s}\n' for i,s in enumerate(sequences)))
    (work/'AMFR.fasta').write_text('>AMFR\n'+wt+'\n')
    names=['ALA','CYS','ASP','GLU','PHE','GLY','HIS','ILE','LYS','LEU','MET','ASN','PRO','GLN','ARG','SER','THR','VAL','TRP','TYR']
    aa=dict(zip('ACDEFGHIKLMNPQRSTVWY',names))
    # This dummy coordinate file is upstream's JET interface, not a predicted structure.
    (work/'AMFR.pdb').write_text(''.join('ATOM%7d  CA  %s A%4d      43.524  70.381  46.465  1.00   0.0\n'%(i,aa[a],i) for i,a in enumerate(wt,1)))
    mapping={}
    for row in item['candidates']:
        edits=mutations(wt,row['sequence'])
        if not edits:
            raise ValueError('Explicit WT candidate is unsupported by native selected-mutant path')
        mapping[row['id']]=','.join(f'{a}{i+1}{b}' for i,a,b in edits)
    (work/'mutants.txt').write_text('\n'.join(mapping.values())+'\n')
    prefix=['singularity','exec','--cleanenv','--containall','--no-home',
            '--bind',str(work)+':/work','--pwd','/work',
            '--env','OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2',str(image)]
    allowed=sorted(os.sched_getaffinity(0))[:2]
    def call(command,logname):
        with (work/logname).open('w') as log:
            subprocess.run(prefix+command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,
                           preexec_fn=lambda:os.sched_setaffinity(0,allowed))
    # The image has no Python2. These are the exact Java and R stages called by
    # gemmeAnal.py; preserve native algorithms and keep intermediates for inspection.
    call(['/bin/sh','-c','cp /opt/GEMME/default.conf default.conf'], 'prepare.log')
    cfg=(work/'default.conf').read_text().replace('results\t\t5000','results\t\t20000')
    (work/'default.conf').write_text(cfg)
    shutil.copyfile(work/'alignment.fasta',work/'AMFR_A.fasta')
    call(['java','-Xmx9126m','-XX:ActiveProcessorCount=2','-cp','/opt/JET2:/opt/JET2/jet/extLibs/vecmath.jar',
          'jet.JET','-c','default.conf','-i','AMFR.pdb','-o','/work','-p','J','-r','input',
          '-f','AMFR_A.fasta','-d','chain','-n','1'], 'jet.log')
    jet=work/'AMFR/AMFR_jet.res'
    if jet.exists():shutil.copyfile(jet,work/'AMFR_jet.res')
    if not (work/'AMFR_jet.res').exists():
        raise ValueError('JET did not produce a conservation file')
    call(['Rscript','--save','/opt/GEMME/computePred.R','AMFR','alignment.fasta','FALSE','mutants.txt'], 'gemme.log')
    import pandas as pd
    path=work/'AMFR_normPred_evolCombi.txt'
    table=pd.read_csv(path,sep=r'\s+',index_col=0)
    if list(table.columns)!=['x'] or table.index.has_duplicates:
        raise ValueError('Unrecognized or duplicate native GEMME predictions')
    scores=table['x'].to_dict()
    if set(scores)!=set(mapping.values()):
        raise ValueError('GEMME candidate mismatch; no imputation or row-order guess')
    provenance={'image':'elodielaine/gemme','image_digest':IMAGE_DIGEST,'image_sha256':sha(image),
        'input_sha256':sha(input_path),'msa_sha256':sha(msa),'msa_rows':len(sequences),
        'ambiguous_homologs_excluded':ambiguous,'prepared_msa_sha256':sha(work/'alignment.fasta'),
        'nseqs':20000,'jet_iterations':1,'native_internal_rng':'JET native random sampling is not exposed as seed; data sampling remains seed42',
        'method':'native normalized evolCombi selected-mutant output; no sign fitting',
        'wrapper':'Java/R stages from gemmeAnal.py, because container lacks Python2',
        'dummy_pdb':'upstream JET input interface only; not structural evidence',
        'labels_read':False,'cpu_affinity':allowed,
        'intermediates':str(work), 'native_output_sha256':sha(path)}
    freeze(work/'provenance.json',provenance)
    return [float(scores[mapping[r['id']]]) for r in item['candidates']],0.0,provenance
