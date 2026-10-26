"""Recover RNA confidence from verified AF3 atoms, never exported PDB B-factors.

Geometry and candidate selection remain those of the frozen input. Only new
version directories may receive the derived PDB and repaired evidence.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

VERSION = 'rna-cif-confidence-v1'
ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_MARKER = 'WT structure evidence; no mutant structure is supplied:\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    p = Path(path)
    if p.is_file():
        return p
    return Path(str(p).replace('/data/yanjie_huang/EVAgent/', str(ROOT) + '/'))


def _blank(value):
    return '' if value in {'?', '.', ' '} else value


def pdb_atoms(path):
    """Use explicit identities; ambiguous/multiple models are never merged."""
    atoms = {}
    models = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith('MODEL '):
            models += 1
        if not line.startswith('ATOM  '):
            continue
        key = (line[21], int(line[22:26]), line[26].strip(),
               line[17:20].strip(), line[12:16].strip().replace('*', "'"),
               line[16].strip())
        if key in atoms:
            raise ValueError('duplicate_pdb_atom')
        xyz = tuple(float(line[i:i+8]) for i in (30, 38, 46))
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError('nonfinite_pdb_coordinates')
        atoms[key] = (xyz, line)
    if models > 1 or not atoms:
        raise ValueError('ambiguous_or_empty_pdb_model')
    return atoms


def anchors(atoms, wt):
    residues = {}
    for key in atoms:
        chain, pos, ins, base, atom, alt = key
        if base in 'ACGU' and len(base) == 1 and atom in {"C4'", "C1'", 'P'} and alt in {'', 'A'}:
            slot = residues.setdefault((chain, pos, ins, base), {})
            if atom in slot:
                raise ValueError('ambiguous_anchor_altloc')
            slot[atom] = key
    chains = {}
    for key in sorted(residues):
        chains.setdefault(key[0], []).append(key)
    matches = [keys for keys in chains.values() if ''.join(k[3] for k in keys) == wt]
    if len(matches) != 1:
        raise ValueError('no_unique_exact_wt_chain')
    return [next(residues[k][a] for a in ("C4'", "C1'", 'P') if a in residues[k])
            for k in matches[0]]


def locate_cif(item, pdb):
    """Known source layouts only; snapshot hashes take precedence."""
    snapshot_path = pdb.parent / 'snapshot.json'
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        provenance = snapshot['sources']['structure']['provenance']
        sources = [p for p in provenance if p.get('id') == 'tertiary' and p.get('tool') == 'AlphaFold3']
        if len(sources) != 1 or not sources[0].get('sha256'):
            raise ValueError('unverified_cif_provenance')
        source = sources[0]
        cif = pdb.parent / source['artifact']
        if sha(cif) != source['sha256']:
            raise ValueError('cif_provenance_hash_mismatch')
        return cif, {'snapshot_path': str(snapshot_path), 'snapshot_sha256': sha(snapshot_path),
                     'expected_cif_sha256': source['sha256']}
    if not item.get('structure_method', '').startswith('AF3 WT prediction'):
        raise ValueError('unverified_prediction_method')
    matches = sorted(pdb.parent.glob('*_model.cif'))
    if len(matches) != 1:
        raise ValueError('missing_or_ambiguous_source_cif')
    return matches[0], {'structure_method': item['structure_method']}


def cif_confidence(cif, atoms):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    data = MMCIF2Dict(str(cif))
    if 'AlphaFold' not in data.get('_software.name', []):
        raise ValueError('cif_not_alphafold')
    n = len(data['_atom_site.id'])

    def col(name, fallback=None, default=None):
        values = data.get('_atom_site.' + name)
        if values is None and fallback:
            values = data.get('_atom_site.' + fallback)
        if values is None:
            if default is None:
                raise ValueError('missing_cif_column:' + name)
            values = [default] * n
        if len(values) != n:
            raise ValueError('cif_column_length_mismatch')
        return values

    if len(set(col('pdbx_PDB_model_num', default='1'))) != 1:
        raise ValueError('ambiguous_cif_models')
    rows = zip(col('group_PDB'), col('auth_asym_id', 'label_asym_id'),
               col('auth_seq_id', 'label_seq_id'), col('pdbx_PDB_ins_code', default='?'),
               col('auth_comp_id', 'label_comp_id'), col('auth_atom_id', 'label_atom_id'),
               col('label_alt_id', default='.'), col('Cartn_x'), col('Cartn_y'),
               col('Cartn_z'), col('B_iso_or_equiv'))
    source = {}
    for group, chain, pos, ins, base, atom, alt, x, y, z, b in rows:
        if group != 'ATOM':
            continue
        key = (chain, int(pos), _blank(ins), base, atom.replace('*', "'"), _blank(alt))
        if key in source:
            raise ValueError('duplicate_cif_atom')
        xyz, confidence = (float(x), float(y), float(z)), float(b)
        if not all(math.isfinite(v) for v in (*xyz, confidence)) or not 0 <= confidence <= 100:
            raise ValueError('invalid_cif_confidence_or_coordinates')
        source[key] = (xyz, confidence)
    values = {}
    for key, (xyz, _) in atoms.items():
        if key not in source:
            raise ValueError('unmapped_pdb_atom')
        original_xyz, confidence = source[key]
        # PDB and AF3 CIF coordinates are rounded to 0.001 angstrom.
        if max(abs(a-b) for a, b in zip(xyz, original_xyz)) > 0.0011:
            raise ValueError('cif_pdb_coordinate_mismatch')
        values[key] = confidence
    if not any(values.values()):
        raise ValueError('all_zero_source_confidence')
    return values


def repair_packet(query, item, packet, pdb_output=None):
    """Fail closed for unverifiable confidence; keep verified legacy geometry."""
    result = copy.deepcopy(packet)
    if packet.get('assay') != query['assay'] or item['assay'] != query['assay']:
        raise ValueError('assay_mismatch')
    if packet.get('kind') != item['structure_kind']:
        raise ValueError('structure_kind_mismatch')
    if packet.get('source', {}).get('sha256') != item['structure']['sha256']:
        raise ValueError('packet_coordinate_source_mismatch')
    pdb = resolve(item['structure']['path'])
    if sha(pdb) != item['structure']['sha256']:
        raise ValueError('pdb_source_hash_mismatch')
    atoms = pdb_atoms(pdb)
    nodes = anchors(atoms, query['wt_sequence'])
    values = {}
    provenance = {'version': VERSION, 'coordinate_path': str(pdb), 'coordinate_sha256': sha(pdb),
                  'status': 'unavailable', 'reason': None, 'matched_atoms': 0}
    if item['structure_kind'] == 'predicted_WT':
        try:
            cif, origin = locate_cif(item, pdb)
            provenance.update(origin, cif_path=str(cif), cif_sha256=sha(cif))
            values = cif_confidence(cif, atoms)
            provenance.update(status='verified', matched_atoms=len(values),
                              field='_atom_site.B_iso_or_equiv', coordinate_tolerance_a=0.0011)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            provenance['reason'] = str(exc)
    else:
        provenance['reason'] = 'experimental_b_factor_is_not_plddt'
    candidates = query['candidates']
    if set(result['site_evidence']) != {c['id'] for c in candidates}:
        raise ValueError('packet_candidate_mismatch')
    for candidate in candidates:
        sites = result['site_evidence'][candidate['id']]
        if [s['position'] for s in sites] != candidate['changed_positions']:
            raise ValueError('packet_position_mismatch')
        for site in sites:
            if not 1 <= site['position'] <= len(nodes):
                raise ValueError('packet_position_out_of_range')
            key = nodes[site['position'] - 1]
            if key[4] != site['anchor_atom']:
                raise ValueError('packet_anchor_mismatch')
            site['plddt'] = round(values[key], 2) if values else None
    if values and pdb_output is not None:
        target = Path(pdb_output)
        if target.resolve() == pdb.resolve():
            raise ValueError('cannot_overwrite_original_pdb')
        target.parent.mkdir(parents=True, exist_ok=True)
        replacements = {line: line[:60] + f'{values[key]:6.2f}' + line[66:]
                        for key, (_, line) in atoms.items()}
        with target.open('x') as stream:
            stream.write('\n'.join(replacements.get(line, line) for line in pdb.read_text().splitlines()) + '\n')
        result['source'] = {'path': str(target), 'sha256': sha(target)}
    result['schema'] = 'pg-rna-seed42-structure-evidence-v2'
    result['confidence_provenance'] = provenance
    result['definition'] = (packet['definition'] + ' pLDDT is the original AF3 anchor-atom confidence '
                            'after identity and coordinate verification against CIF; null means unavailable '
                            'confidence, not zero or low confidence. Experimental B-factors are not pLDDT.')
    return result


def rebuild_structure_unit(unit, packet):
    """Replace the structured evidence block and bind the new request identity."""
    result = copy.deepcopy(unit)
    template = result['template']
    user = template['user']
    if user.count(STRUCTURE_MARKER) != 1:
        raise ValueError('expected_one_structure_block')
    prefix, tail = user.split(STRUCTURE_MARKER)
    old_json, suffix = tail.split('\n\n', 1)
    old_block = json.loads(old_json)
    ids = template['expected_ids']
    if set(old_block['site_evidence']) != set(ids):
        raise ValueError('request_structure_ids_mismatch')
    block = {'kind': packet['kind'], 'definition': packet['definition'],
             'confidence_status': packet['confidence_provenance']['status'],
             'confidence_unavailable_reason': packet['confidence_provenance']['reason'],
             'site_evidence': {cid: packet['site_evidence'][cid] for cid in ids}}
    template['user'] = prefix + STRUCTURE_MARKER + json.dumps(
        block, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n\n' + suffix
    result.pop('logical_id', None)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    result['logical_id'] = hashlib.sha256(payload.encode()).hexdigest()
    return result
