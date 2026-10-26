"""Build reviewed evidence/configuration artifacts, without any model requests."""
import csv
import json
import shutil
from pathlib import Path

from evagent.integrated.bundles import file_hash
from evagent.integrated.evidence_tools import prepare_bundle
from evagent.integrated.structure_tools import cif_features

ROOT = Path('/data/yanjie_huang/EVAgent/experiments/cr_ncrna_evidence_b_v1')
ASSETS = ROOT/'assets'
BACKGROUND = {
    'Ribozyme': {'id': 'Doudna_Cech_2002_background', 'date': '2002-07-11',
        'doi': '10.1038/418222a', 'title': 'The chemical repertoire of natural ribozymes',
        'url': 'https://pubmed.ncbi.nlm.nih.gov/12110898/',
        'excerpt': 'some RNA molecules, called ribozymes, have intrinsic enzyme-like activity',
        'summary': 'RNA itself can catalyze chemical reactions without protein cofactors. The review describes several classes of catalytic RNA and emphasizes that RNA can have functions beyond storing genetic information.',
        'applicability': 'General catalytic-RNA background only; no assay-specific variants, kinetics, measurements, sites or mechanisms are inferred from this abstract.'},
    'tRNA': {'id': 'Shi_Moore_2000_background', 'date': '2000-08-01',
        'doi': '10.1017/S1355838200000364',
        'title': 'The crystal structure of yeast phenylalanine tRNA at 1.93 A resolution: a classic structure revisited.',
        'url': 'https://rnajournal.cshlp.org/content/6/8/1091.abstract',
        'excerpt': 'Several new divalent cation binding sites have been identified',
        'summary': 'A refined yeast phenylalanine tRNA structure showed that loop backbone conformations, bound divalent cations and hydration details can differ from earlier structural models.',
        'applicability': 'Background on structural uncertainty in tRNA. This does not establish the structure of the assayed construct or predict effects across different tRNA isoacceptors.'},
    'Aptamer': {'id': 'Ellington_Szostak_1990_background', 'date': '1990-08-30',
        'doi': '10.1038/346818a0', 'title': 'In vitro selection of RNA molecules that bind specific ligands',
        'url': 'https://www.nature.com/articles/346818a0',
        'excerpt': 'RNA molecules that bind specifically to a variety of organic dyes',
        'summary': 'RNA selected from initially random sequences can form binding sites for specific small ligands.',
        'applicability': 'General aptamer background only. The study concerns organic dyes, not the protein targets in this assay; no target-specific binding contacts or mutation effects are inferred.'},
}


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(obj, indent=2, allow_nan=False)+'\n'
    if p.exists() and p.read_text() != content:
        raise ValueError(f'Refusing to replace frozen artifact: {p}')
    p.write_text(content)


def main():
    registry = json.loads((ASSETS/'registry.json').read_text())
    with Path('/data/yanjie_huang/RG-LLM/data/RNAGym/fitness/reference_sheet_final.csv').open(encoding='utf-8-sig') as f:
        denied_titles = [r['TITLE'] for r in csv.DictReader(f)]
    model = {'model': 'gemini-3.8-flash', 'thinking_level': 'medium', 'temperature': 0.0,
             'max_tokens': 8192, 'timeout_s': 900, 'api_key_env': 'GEMINI_API_KEY',
             'proxy': 'http://127.0.0.1:17892'}
    cells, coverage = [], []
    calibration = None
    for row in registry['ncrna_tasks']:
        a = row['assay']
        source_status = row['audit']['status']
        available = []
        if source_status == 'admitted':
            folder = ASSETS/'evidence'/a
            raw = folder/'raw'
            raw.mkdir(parents=True, exist_ok=True)
            task = json.loads((ASSETS/'tasks'/f'{a}.json').read_text())
            shutil.copyfile(ASSETS/'secondary'/f'{a}.json', raw/'secondary.json')
            record = {**BACKGROUND[row['rna_type']], 'background_only': True, 'target_experiment': False,
                      'direct_variant_effects': False, 'benchmark_derived': False}
            save(raw/'literature.json', {'assay': a, 'reviewed': True, 'records': [record],
                'review': {'date': '2026-09-10', 'scope': 'abstract only; no target experiment, supplement or mutation query',
                          'source_precedes_all_assays': True, 'base_model_exposure_unknown': True}})
            config = {'task_file': str(ASSETS/'tasks'/f'{a}.json'), 'source_audit': row['audit'],
                'bundle_file': str(folder/'bundle.json'), 'secondary_file': 'raw/secondary.json',
                'literature_file': 'raw/literature.json', 'denied_titles': denied_titles}
            audit_path = folder/'msa_audit.json'
            if audit_path.exists():
                config.update(msa_file='raw/mapped.sto', msa_audit=json.loads(audit_path.read_text()))
            tertiary = raw/'tertiary.json'
            public = json.loads((folder/'public_receipt.json').read_text())
            if public['pdb']['status'] == 'available':
                pdb = public['pdb']['pdb']
                if pdb != '9V7N':
                    raise ValueError('New PDB entry requires a fresh independent-source review')
                features = cif_features(raw/f'{pdb}.cif', task['wt_sequence'], predicted=False,
                    provider='RCSB-PDB', version='frozen-2026-09-10', artifact=f'raw/{pdb}.cif')
                features['task'].update(pdb_id=pdb, construct_context='Observed WT region within the deposited precursor; outside-region contacts omitted')
                # The accepted Azoarcus entry is an independently measured cryo-EM
                # structure, not a structure inferred from benchmark mutation effects.
                features['task']['structure_source_doi'] = '10.1038/s41467-026-70747-y'
                save(raw/'verified_pdb.json', features)
                tertiary = raw/'verified_pdb.json'
            if (folder/'af3_job.json').exists():
                job = json.loads((folder/'af3_job.json').read_text())
                name = job['job']
                output = ASSETS/'af3_outputs'/name
                receipt = output/'receipt.json'
                if not receipt.exists():
                    raise ValueError(f'AF3 job has no terminal receipt: {a}')
                receipt_value = json.loads(receipt.read_text())
                if receipt_value['status'] == 'complete':
                    for relative, expected in receipt_value['outputs'].items():
                        if file_hash(output/relative) != expected:
                            raise ValueError('AF3 raw artifact hash changed')
                    p = output/name/f'{name}_model.cif'
                    shutil.copyfile(p, raw/'af3_model.cif')
                    shutil.copyfile(output/name/f'{name}_summary_confidences.json', raw/'af3_confidence.json')
                    shutil.copyfile(receipt, raw/'af3_receipt.json')
                    features = cif_features(raw/'af3_model.cif', task['wt_sequence'], predicted=True,
                        provider='AlphaFold3', version=receipt_value['image'], artifact='raw/af3_model.cif')
                    features['task'].update(msa_dependency='none; no credible natural alignment available' if job.get('msa_mode') == 'none_pending_review' else job.get('msa_mode', 'none'),
                        msa_family=job.get('msa_family'), model_seed=1,
                        selected_model='AF3 own structural ranking among 5 samples; benchmark labels never used')
                    features['provenance'].extend([{'id': key, 'tool': 'AlphaFold3', 'version': receipt_value['image'],
                        'sha256': file_hash(raw/key), 'artifact': 'raw/'+key} for key in ('af3_confidence.json', 'af3_receipt.json')])
                    save(raw/'af3_tertiary.json', features)
                    tertiary = raw/'af3_tertiary.json'
            if tertiary.exists():
                config['structure_file'] = str(tertiary.relative_to(folder))
            save(folder/'prepare.json', config)
            prepare_bundle(config, base=folder, output=folder/'bundle.json')
            bundle = json.loads((folder/'bundle.json').read_text())
            available = [s for s, v in bundle['sources'].items() if v['status'] == 'available']
            if len(available) == 3 and calibration is None:
                calibration = a
        coverage.append({**row, 'available_evidence': available})
        for n in (50, 250):
            for condition in ('cr', 'msa', 'structure', 'literature', 'all'):
                cell = {'assay': a, 'n': n, 'condition': condition, 'family': row['audit']['family'], 'source_status': source_status}
                if row['sample_sizes'][str(n)] != 'present':
                    cell.update(status='insufficient_samples', reason='Requested 250 candidates; Pitt_2010 has only 162 valid candidates')
                elif source_status != 'admitted':
                    cell.update(status=source_status, reason=row['audit']['reason'])
                elif condition not in ('cr', 'all') and condition not in available:
                    cell.update(status='evidence_unavailable', reason=f'No credible frozen {condition} evidence')
                else:
                    cfg = {'method': 'cr', 'backend': 'direct', **model, 'protocol': 'integrated-cr-v2',
                        'prompt_version': 'v2', 'max_repairs': 1, 'group_size': 25, 'anchor_count': 4,
                        'seed': 1, 'memory_mode': 'none', 'evidence_mode': 'none', 'token_budget': 50_000_000,
                        'sample_file': str(ASSETS/'samples'/a/f'n{n}.json'),
                        'context_file': str(ASSETS/'contexts'/f'{a}.json'),
                        'output_dir': str(ROOT/'results'/a/f'n{n}'/condition)}
                    if condition != 'cr':
                        cfg.update(evidence_mode='bundle', evidence_bundle_file=str(ASSETS/'evidence'/a/'bundle.json'),
                                   evidence_sources=['msa', 'structure', 'literature'] if condition == 'all' else [condition])
                    path = ROOT/'configs'/a/f'n{n}_{condition}.json'
                    save(path, cfg)
                    cell.update(status='eligible', config=str(path))
                cells.append(cell)
    save(ROOT/'coverage.json', {'total_tasks': 70, 'ncrna_tasks': 31, 'matrix_slots': 310,
        'potential_results_after_sample_size_check': 305, 'tasks': coverage,
        'splicing_tasks_separate': registry['splicing_tasks_separate']})
    if calibration is None:
        raise ValueError('No complete five-condition calibration assay')
    save(ROOT/'campaign.json', {'schema': 'evagent-campaign-v1', 'output_dir': str(ROOT/'campaign'),
        'token_budget': 50_000_000, 'model_config': model, 'cells': cells, 'calibration_assay': calibration})
    save(ROOT/'assets_manifest.json', {'sha256': {str(p.relative_to(ASSETS)): file_hash(p)
        for p in sorted(ASSETS.rglob('*')) if p.is_file()}})
    print(json.dumps({'cells': len(cells), 'eligible': sum(c['status']=='eligible' for c in cells), 'calibration': calibration}))


if __name__ == '__main__':
    main()
