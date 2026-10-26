"""Materialize the complete ncRNA registry and admitted label-free worker inputs.

Source audit metadata is kept outside model-facing evidence. No source papers'
experimental results are imported. Historical *samples*, never predictions, are reused.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/yanjie_huang/EVAgent')
OUT = ROOT / 'experiments/cr_ncrna_evidence_b_v1/assets'
REF = Path('/data/yanjie_huang/RG-LLM/data/RNAGym/fitness/reference_sheet_final.csv')


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and json.loads(p.read_text()) != value:
        raise ValueError(f'Frozen input changed: {p}')
    p.write_text(json.dumps(value, indent=2)+'\n')


def audit(row):
    a = row['DMS_ID']
    excluded = {'Roberts_2023_HDV_ribozyme',
                'Kobori_2015_ribozyme_j12', 'Kobori_2015_ribozyme_p4'}
    if a in excluded:
        return dict(status='source_excluded', family=a, reason='Reference identifies HDV-derived construct; coverage statistics only')
    if row['RNA_TYPE'] == 'tRNA':
        return dict(status='admitted', family='yeast_tRNA', reason='Yeast cellular tRNA; source explicitly identified in reference assay description', source='RNAGym reference metadata')
    if row['RNA_TYPE'] == 'Aptamer':
        return dict(status='admitted', family='Tome_aptamers', reason='Synthetic selected RNA binding nonpathogen protein targets', source='RNAGym reference metadata')
    if 'cpeb3' in a.lower() or 'cepeb3' in a.lower() or a == 'Beck_2022_ribozyme':
        return dict(status='admitted', family='mammalian_CPEB3', reason='Mammalian CPEB3 intronic RNA, distinguished from viral HDV source', source='https://pmc.ncbi.nlm.nih.gov/articles/PMC2890282/')
    if a.startswith('Janzen_'):
        return dict(status='admitted', family='Janzen_aminoacylation', reason='Synthetic in-vitro-selected aminoacylation RNAs', source='https://www.nature.com/articles/s41467-022-31387-0')
    if a.startswith('Zhang_2024_'):
        return dict(status='admitted', family='human_TS_like', reason='Human genomic RNA; reference identifies LINE1/OR4K15 origin', source='RNAGym reference metadata')
    if a.startswith('McRae_'):
        return dict(status='admitted', family='synthetic_TPR', reason='Artificial in-vitro-evolved RNA polymerase ribozyme; not a pathogen polymerase', source='https://pure.au.dk/ws/files/453154360/mcrae-et-al-2024-cryo-em-structure-and-functional-landscape-of-an-rna-polymerase-ribozyme.pdf')
    if a == 'Pitt_2010_ribozyme':
        return dict(status='admitted', family='synthetic_classII_ligase', reason='In-vitro-selected class II RNA ligase; reference assay description incorrectly calls it a hammerhead', source='https://pmc.ncbi.nlm.nih.gov/articles/PMC3392653/')
    if a == 'Peri_2022_ribozyme':
        return dict(status='admitted', family='Azoarcus_groupI', reason='Azoarcus group I intron, environmental nonpathogen RNA', source='https://pmc.ncbi.nlm.nih.gov/articles/PMC8890501/')
    if a == 'Soo_2021_ribozyme':
        return dict(status='source_unresolved', family='Tetrahymena_groupI', reason='Source is Tetrahymena, but reference claims in-vitro activity while primary study uses an antibiotic-selection reporter; task-context review required', source='https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1009353')
    if a == 'Kobori_2016_osa_ribozyme':
        return dict(status='admitted', family='rice_twister', reason='Oryza sativa Osa-1-4 RNA; provenance subject to PDB organism verification', source='https://www.rcsb.org/structure/4OJI')
    return dict(status='source_unresolved', family=a, reason='Exact construct provenance requires review before any sequence-level prediction')


def main():
    with REF.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    registry, separate = [], []
    for r in rows:
        if r['RNA_TYPE'] == 'mRNA-splicing':
            separate.append(r['DMS_ID'])
        if r['RNA_TYPE'] not in {'Ribozyme', 'tRNA', 'Aptamer'}:
            continue
        a = r['DMS_ID']
        entry = {'assay': a, 'rna_type': r['RNA_TYPE'], 'audit': audit(r), 'sample_sizes': {}}
        worker = {'assay': a, 'wt_sequence': r['RAW_CONSTRUCT_SEQ'], 'variants': []}
        metadata = {k: r[k] for k in ['DMS_ID', 'RNA_TYPE', 'RNA_TYPE_GRANULAR', 'ASSAY_DESCRIPTION']}
        if a == 'Pitt_2010_ribozyme':
            metadata['ASSAY_DESCRIPTION'] = 'In-vitro ligation activity of synthetic class II RNA ligase variants.'
        seen = set()
        for n in (50, 250):
            p = ROOT/'experiments/cr_fullseq_b25_v1/samples'/a/f'n{n}_b1.json'
            entry['sample_sizes'][str(n)] = 'present' if p.is_file() else 'insufficient_samples'
            if not p.is_file() or entry['audit']['status'] != 'admitted':
                continue
            sample = json.loads(p.read_text())
            if len(sample['variants']) != n:
                raise ValueError('Historical frozen sample length mismatch')
            sample['assay'] = a
            save(OUT/'samples'/a/f'n{n}.json', sample)
            for index, v in enumerate(sample['variants'], 1):
                s = v.get('sequence', v.get('seq'))
                if s not in seen:
                    worker['variants'].append({'variant_id': f'M{index:02d}', 'sequence': s,
                                               'mutation': v.get('mutation', v.get('id'))})
                    seen.add(s)
        registry.append(entry)
        if entry['audit']['status'] == 'admitted':
            save(OUT/'tasks'/f'{a}.json', worker)
            save(OUT/'contexts'/f'{a}.json', {'assay': a, 'wt_sequence': worker['wt_sequence'], 'metadata': metadata})
    if len(registry) != 31:
        raise ValueError('Unexpected ncRNA registry size')
    save(OUT/'registry.json', {'reference_sha256': hashlib.sha256(REF.read_bytes()).hexdigest(),
                             'total_reference_tasks': len(rows), 'ncrna_tasks': registry,
                             'splicing_tasks_separate': separate})
    print(json.dumps({'registry': len(registry), 'admitted': sum(r['audit']['status']=='admitted' for r in registry)}))


if __name__ == '__main__':
    main()
