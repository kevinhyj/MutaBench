"""Collect public, frozen WT structures and homolog alignments before inference."""
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import httpx
from evagent.integrated.bundles import file_hash
from evagent.integrated.evidence_tools import canonical, read_alignment
from evagent.integrated.gateway import digest
from evagent.integrated.structure_tools import cif_features

ROOT = Path('/data/yanjie_huang/EVAgent/experiments/cr_ncrna_evidence_b_v1/assets')
REF = Path('/data/yanjie_huang/RG-LLM/data/RNAGym/fitness/reference_sheet_final.csv')


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2)+'\n')


def normalize(s):
    return re.sub('[^a-z0-9]', '', s.lower())


def main():
    with REF.open(encoding='utf-8-sig') as f:
        metadata = {r['DMS_ID']: r for r in csv.DictReader(f)}
    denied_titles = {normalize(r['TITLE']) for r in metadata.values()}
    registry = json.loads((ROOT/'registry.json').read_text())
    archive = ROOT/'public/fitness_MSAs.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=45, trust_env=False, follow_redirects=True) as client:
        if not archive.exists():
            response = client.get('https://marks.hms.harvard.edu/rnagym/fitness_prediction/fitness_MSAs.zip')
            response.raise_for_status()
            archive.write_bytes(response.content)
        z = zipfile.ZipFile(io.BytesIO(archive.read_bytes()))
        for row in registry['ncrna_tasks']:
            if row['audit']['status'] != 'admitted':
                continue
            a = row['assay']
            task = json.loads((ROOT/'tasks'/f'{a}.json').read_text())
            wt = canonical(task['wt_sequence'])
            folder = ROOT/'evidence'/a
            raw = folder/'raw'
            raw.mkdir(parents=True, exist_ok=True)
            receipt = {'assay': a, 'msa': {'status': 'unavailable', 'reason': 'No assay-specific official alignment'},
                       'pdb': {'status': 'unavailable', 'reason': 'No verified exact WT structure'}, 'searches': []}
            found = [n for n in z.namelist() if n.lower().endswith('/'+a.lower()+'.a2m')]
            if found:
                p = raw/'official_msa.a2m'
                p.write_bytes(z.read(found[0]))
                alignment = read_alignment(p)
                exact = any(s.replace('-', '') == wt for _, s in alignment)
                receipt['msa'] = {'status': 'available' if exact else 'needs_query_alignment',
                    'file': str(p), 'sha256': file_hash(p), 'archive_sha256': file_hash(archive),
                    'entry': found[0], 'sequences': len(alignment), 'exact_wt': exact,
                    'header_examples': [name for name, _ in alignment[:4]]}
            query = {'query': {'type': 'terminal', 'service': 'sequence', 'parameters': {
                'evalue_cutoff': .01, 'identity_cutoff': 1.0, 'sequence_type': 'rna', 'value': wt}},
                'return_type': 'polymer_entity', 'request_options': {'results_content_type': ['experimental'], 'paginate': {'start': 0, 'rows': 50}}}
            try:
                response = client.get('https://search.rcsb.org/rcsbsearch/v2/query', params={'json': json.dumps(query)})
                if response.status_code == 204:
                    candidates = []
                else:
                    response.raise_for_status()
                    candidates = response.json().get('result_set', [])
                pdb_ids = sorted({x['identifier'].split('_')[0] for x in candidates})
                for pdb in pdb_ids:
                    entry = client.get('https://data.rcsb.org/rest/v1/core/entry/'+pdb)
                    entry.raise_for_status()
                    data = entry.json()
                    citation = data.get('rcsb_primary_citation', {})
                    if normalize(citation.get('title', '')) in denied_titles:
                        receipt['searches'].append({'pdb': pdb, 'status': 'excluded_target_experiment'})
                        continue
                    cif = client.get('https://files.rcsb.org/download/'+pdb+'.cif')
                    cif.raise_for_status()
                    p = raw/(pdb+'.cif')
                    p.write_bytes(cif.content)
                    try:
                        features = cif_features(p, task['wt_sequence'], predicted=False, provider='RCSB-PDB',
                                                version=data.get('rcsb_accession_info', {}).get('revision_date', 'frozen'),
                                                artifact='raw/'+p.name)
                        features['task']['pdb_id'] = pdb
                        features['task']['experimental_method'] = [x['method'] for x in data.get('exptl', [])]
                        save(raw/'tertiary.json', features)
                        receipt['pdb'] = {'status': 'available', 'pdb': pdb, 'file': str(raw/'tertiary.json')}
                        break
                    except ValueError as e:
                        receipt['searches'].append({'pdb': pdb, 'status': 'mapping_rejected', 'reason': str(e)})
                receipt['pdb']['search_hits'] = len(pdb_ids)
            except Exception as e:
                receipt['pdb'] = {'status': 'retrieval_failed', 'error': type(e).__name__}
            save(folder/'public_receipt.json', receipt)
            if receipt['pdb']['status'] != 'available':
                name = digest(wt)[:20]
                # RNA MSA is added only after its WT mapping and origin are validated.
                prediction = {'name': name, 'dialect': 'alphafold3', 'version': 3, 'modelSeeds': [1],
                              'sequences': [{'rna': {'id': 'A', 'sequence': wt, 'unpairedMsa': ''}}]}
                save(ROOT/'af3_inputs'/f'{name}.json', prediction)
                save(folder/'af3_job.json', {'job': name, 'wt_sha256': digest(task['wt_sequence']), 'msa_mode': 'none_pending_review'})
            print(json.dumps({'assay': a, 'msa': receipt['msa']['status'], 'pdb': receipt['pdb']['status']}), flush=True)


if __name__ == '__main__':
    main()
