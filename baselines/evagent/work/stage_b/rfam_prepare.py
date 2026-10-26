"""Freeze natural Rfam seed homologs, then map complete WT with Infernal."""
import json
import subprocess
from pathlib import Path

import httpx
from evagent.integrated.bundles import file_hash
from evagent.integrated.evidence_tools import canonical, read_alignment
from evagent.integrated.gateway import digest

ROOT = Path('/data/yanjie_huang/EVAgent/experiments/cr_ncrna_evidence_b_v1/assets')
CMALIGN = Path('/data/yanjie_huang/EVAgent/work/stage_b_tools/infernal-1.1.5-linux-intel-gcc/binaries/cmalign')


def main():
    registry = json.loads((ROOT/'registry.json').read_text())
    with httpx.Client(timeout=90, follow_redirects=True, trust_env=False) as client:
        for row in registry['ncrna_tasks']:
            family = {'yeast_tRNA': 'RF00005', 'mammalian_CPEB3': 'RF00622'}.get(row['audit']['family'])
            if not family or row['audit']['status'] != 'admitted':
                continue
            folder = ROOT/'evidence'/row['assay']
            raw = folder/'raw'
            urls = {'seed.sto': f'https://rfam.org/family/{family}/alignment',
                    'family.cm': f'https://rfam.org/family/{family}/cm'}
            for name, url in urls.items():
                p = raw/name
                if not p.exists():
                    response = client.get(url)
                    response.raise_for_status()
                    p.write_bytes(response.content)
            records = read_alignment(raw/'seed.sto')
            task = json.loads((ROOT/'tasks'/f"{row['assay']}.json").read_text())
            wt = canonical(task['wt_sequence'])
            homologs = [(name, seq.replace('-', '')) for name, seq in records
                        if not set(seq)-set('ACGU-')]
            if len(homologs) < 2:
                raise ValueError('No credible natural homologs')
            (raw/'align_input.fa').write_text('>WT_query\n'+wt+'\n'+''.join(f'>{n}\n{s}\n' for n, s in homologs))
            args = [str(CMALIGN), '--cpu', '2', '--notrunc', '-o', str(raw/'mapped.sto'),
                    str(raw/'family.cm'), str(raw/'align_input.fa')]
            with (raw/'cmalign.log').open('w') as log:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
            aligned = read_alignment(raw/'mapped.sto')
            query = dict(aligned)['WT_query']
            assert query.replace('-', '') == wt
            columns = [i for i, c in enumerate(query) if c != '-']
            a3m = '>WT_query\n'+wt+'\n'+''.join(f'>{n}\n'+''.join(s[i] for i in columns)+'\n'
                    for n, s in aligned if n != 'WT_query')
            (raw/'af3_rna_msa.a3m').write_text(a3m)
            audit = {'origin': 'natural_homologs', 'reviewed': True, 'family': family,
                     'source_urls': urls, 'seed_sha256': file_hash(raw/'seed.sto'),
                     'cm_sha256': file_hash(raw/'family.cm'), 'cmalign_sha256': file_hash(CMALIGN),
                     'infernal_version': '1.1.5', 'parameters': ['--cpu', '2', '--notrunc'],
                     'excluded_ambiguous_seed_records': len(records)-len(homologs),
                     'note': 'Natural Rfam seed only plus WT query; no benchmark mutant library. Broad tRNA family is not isoacceptor-specific.'}
            (folder/'msa_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
            job_path = folder/'af3_job.json'
            if job_path.exists():
                job = json.loads(job_path.read_text())
                p = ROOT/'af3_inputs'/f"{job['job']}.json"
                prediction = json.loads(p.read_text())
                prediction['sequences'][0]['rna']['unpairedMsa'] = a3m
                p.write_text(json.dumps(prediction, indent=2)+'\n')
                job.update(msa_mode='Rfam_seed_Infernal', msa_sha256=file_hash(raw/'af3_rna_msa.a3m'),
                           msa_family=family, input_sha256=file_hash(p))
                job_path.write_text(json.dumps(job, indent=2)+'\n')
            print(json.dumps({'assay': row['assay'], 'family': family, 'natural_records': len(homologs),
                              'mapped_wt_length': len(wt), 'mapping_sha256': digest(query)}), flush=True)


if __name__ == '__main__':
    main()
