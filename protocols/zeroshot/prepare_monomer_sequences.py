"""Prepare an explicit Protein/RNA scope, including independently frozen extensions.

Historical queries stay byte-identical. Source cleaning may inspect source labels
for eligibility, but no outcome is copied to the predictor. Evaluation references
are kept separately. This command never submits inference or creates structures.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil

from prepare_singlechain import OLD, ROOT, HERE, resolve
from sampling import singlechain_select
from shared_protocol import load, save, sha, require
from prepare_shared_prompts import add_assay, finish, safe_name
from msa_missingness import load_and_repair_protein_packet
from rna_confidence import repair_packet
from rna_msa_missingness import load_and_repair as repair_rna_msa


def replay_query(cleaner, item, domain, query):
    item = dict(item, source_csv=dict(item['source_csv']))
    item['source_csv']['path'] = str(resolve(item['source_csv']['path']))
    wt, rows, exclusions = cleaner.source_rows(item, domain)
    selected = singlechain_select(rows, n=250, seed=42)
    require(wt == query['wt_sequence'] and len(selected) == len(query['candidates']), 'Frozen cohort mismatch')
    for i, (row, candidate) in enumerate(zip(selected, query['candidates']), 1):
        require(set(candidate) == {'id', 'mutation', 'sequence', 'changed_positions'}, 'Unexpected candidate fields')
        require(candidate['id'] == f'M{i:03d}' and all(candidate[k] == row[k]
                for k in ('mutation', 'sequence', 'changed_positions')), 'Sampling/order mismatch')
    return dict(n=len(selected), pool_n=len(rows), exclusions=exclusions,
                selected_order_match=True, source_csv=item['source_csv'])


def prepare(scope_path, version, *, here=HERE, old=OLD):
    safe_name(version)
    scope_path = Path(scope_path).resolve()
    scope = load(scope_path)
    entries = scope['assays']
    keys = [(r['domain'], r['assay']) for r in entries]
    require(entries and len(set(keys)) == len(keys), 'Empty/duplicate explicit scope')
    require(all(d in {'protein', 'rna'} for d, _ in keys), 'Sequence domain required')
    out, ev = here/'inputs'/version, here/'evaluator'/version
    require(not out.exists() and not ev.exists(), 'New version required')
    spec = importlib.util.spec_from_file_location('frozen_singlechain_cleaner', old/'prepare.py')
    cleaner = importlib.util.module_from_spec(spec); spec.loader.exec_module(cleaner)
    inventory, plan = load(old/'INVENTORY.json'), load(old/'PLAN.json')
    indexed = {(d, r['assay']): r for d in ('protein', 'rna') for r in inventory[d]}
    names, queue, refs, replay, coverage = [], [], [], [], []
    provenance = {str(p): sha(p) for p in (scope_path, old/'PLAN.json', old/'INVENTORY.json',
                  old/'prepare.py', Path(__file__), HERE/'sampling.py')}
    for entry in entries:
        domain, aid = entry['domain'], safe_name(entry['assay'])
        packets = {}
        if 'extension_manifest' in entry:
            mp = (scope_path.parent/entry['extension_manifest']['path']).resolve()
            require(sha(mp) == entry['extension_manifest']['sha256'], 'Extension manifest changed')
            ext = load(mp); require((ext['domain'], ext['assay']) == (domain, aid), 'Extension identity mismatch')
            qp = Path(ext['query_path']); require(sha(qp) == ext['query_sha256'], 'Extension query changed')
            query = load(qp)
            item = dict(assay=aid, wt_sequence=query['wt_sequence'], source_csv=ext['source_csv'])
            require(sha(old/'prepare.py') == ext['cleaning_sha256'], 'Extension cleaning version mismatch')
            rp = here/'evaluator'/ext['version']/'LABEL_REFERENCES.json'
            matches = [r for r in load(rp) if (r['domain'], r['assay']) == (domain, aid)]
            require(len(matches) == 1 and matches[0]['query_sha256'] == sha(qp), 'Extension evaluator mismatch')
            ref = dict(matches[0]); provenance.update({str(mp): sha(mp), str(rp): sha(rp)})
            for kind in ('msa', 'structure'):
                require(ext[kind+'_status'] == 'not_prepared', 'Extension evidence requires a hashed source adapter')
                packets[kind] = dict(status='unavailable', reason='not_prepared_in_frozen_extension')
        else:
            require((domain, aid) in indexed, 'Unknown assay in scope')
            item = indexed[(domain, aid)]
            qp = old/'queries'/domain/aid/'query.json'
            require(sha(qp) == plan['input_hashes'][domain+'/'+aid]['query'], 'Query changed')
            query = load(qp)
            lp = qp.parent/'evaluation_labels.json'
            ref = dict(domain=domain, assay=aid, label_path=str(lp), label_sha256=sha(lp))
            for kind in ('msa', 'structure'):
                src = old/'evidence'/domain/aid/(kind+'.json')
                require(sha(src) == plan['input_hashes'][domain+'/'+aid][kind], 'Evidence changed')
                packet = load(src); provenance[str(src)] = sha(src)
                if domain == 'protein' and kind == 'msa':
                    packet = load_and_repair_protein_packet(query, item, packet, resolve)
                if domain == 'rna' and kind == 'msa':
                    packet = repair_rna_msa(query, packet, resolve)
                    if packet.get('source_profile'):
                        profile_path=resolve(packet['source_profile']['path'])
                        provenance[str(profile_path)]=sha(profile_path)
                if domain == 'rna' and kind == 'structure':
                    packet = repair_packet(query, item, packet,
                        pdb_output=out/'structures'/domain/aid/'mapped_WT.pdb')
                packets[kind] = packet
        require((query['domain'], query['assay']) == (domain, aid), 'Query identity mismatch')
        context_keys = ({'DMS_id','UniProt_ID','molecule_name','selection_assay','selection_type','source_organism'}
                        if domain == 'protein' else {'ASSAY_DESCRIPTION','AUTHOR','DMS_ID','RNA_TYPE','RNA_TYPE_GRANULAR','TITLE','YEAR'})
        context = query.get('assay_context') or {}
        require(context and set(context) <= context_keys, 'Unknown assay context fields; inspect before exposing to predictor')
        require(all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in context.values()), 'Non-scalar assay metadata')
        audit = replay_query(cleaner, item, domain, query)
        for kind, packet in packets.items():
            save(out/'evidence'/domain/aid/(kind+'.json'), packet)
        dest = out/'queries'/domain/aid/'query.json'; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(qp, dest)
        name, units = add_assay(out, domain, aid, dict(query=query, **packets))
        names.append(name); queue.extend(units)
        ref.update(query_path=str(dest.resolve()), query_sha256=sha(dest)); refs.append(ref)
        provenance[str(qp)] = sha(qp)
        provenance[audit['source_csv']['path']] = audit['source_csv']['sha256']
        replay.append(dict(domain=domain, assay=aid, **audit))
        coverage.append(dict(domain=domain, assay=aid, n=audit['n'],
            cohort='N250' if audit['n'] == 250 else 'short_query',
            evidence={k: v.get('status') for k, v in packets.items()}))
        print(f'prepared {domain}/{aid}: {len(units)} requests', flush=True)
    save(out/'SCOPE.json', scope)
    save(out/'COVERAGE.json', coverage)
    save(out/'SAMPLING_REPLAY.json', dict(seed=42, candidate_resampling=False, assays=replay,
         labels_used_for_source_eligibility_only=True, model_calls=0))
    manifest = finish(out, version, names, queue, provenance)
    save(ev/'LABEL_REFERENCES.json', refs)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    m = prepare(args.scope, args.version)
    print({k: m[k] for k in ('version', 'domains', 'queue_units', 'fully_ready')})
