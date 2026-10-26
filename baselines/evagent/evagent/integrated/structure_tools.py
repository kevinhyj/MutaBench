"""Conservative WT mmCIF mapping; geometry is never a mutation-effect label."""
from __future__ import annotations

import math
import re
from pathlib import Path

from .bundles import file_hash
from .evidence_tools import canonical
from .gateway import digest


def cif_features(path: Path, wt_sequence: str, *, predicted: bool,
                 provider: str, version: str, artifact: str | None = None) -> dict:
    import gemmi
    block = gemmi.cif.read_file(str(path)).sole_block()
    wt = canonical(wt_sequence)
    mapping = {}
    modified = {row[0] for row in block.find(['_entity_poly.entity_id', '_entity_poly.pdbx_seq_one_letter_code'])
                if any(x not in {'A', 'C', 'G', 'U'} for x in re.findall(r'\(([^)]+)\)', gemmi.cif.as_string(row[1])))}
    sequences = {row[0]: re.sub(r'\s+', '', gemmi.cif.as_string(row[1]))
                 for row in block.find(['_entity_poly.entity_id', '_entity_poly.pdbx_seq_one_letter_code_can'])}
    monomers = {}
    for row in block.find(['_entity_poly_seq.entity_id', '_entity_poly_seq.num', '_entity_poly_seq.mon_id']):
        entity, index, monomer = row[0], int(row[1]), row[2]
        chain = monomers.setdefault(entity, {})
        if monomer not in {'A', 'C', 'G', 'U'} or index in chain:
            modified.add(entity)
        chain[index] = monomer
    for entity, chain in monomers.items():
        if sorted(chain) != list(range(1, len(chain)+1)):
            modified.add(entity)
        if entity not in modified:
            seq = ''.join(chain[i] for i in sorted(chain))
            if entity in sequences and seq != sequences[entity]:
                modified.add(entity)
            sequences[entity] = seq
    for row in block.find(['_entity_poly.entity_id', '_entity_poly.type']):
        if row[0] in modified or row[0] not in sequences:
            continue
        if 'polyribonucleotide' not in gemmi.cif.as_string(row[1]):
            continue
        seq = sequences[row[0]]
        if seq.count(wt) == 1:
            offset = seq.index(wt)
            mapping[row[0]] = {i+offset+1: i+1 for i in range(len(wt))}
        elif wt.count(seq) == 1 and len(seq) >= .8*len(wt):
            offset = wt.index(seq)
            mapping[row[0]] = {i+1: i+offset+1 for i in range(len(seq))}
    if not mapping:
        raise ValueError('No unambiguous exact WT/construct mapping with >=80pct coverage')
    chains = {}
    labels = {}
    fields = ['label_entity_id', 'label_asym_id', 'label_seq_id', 'label_atom_id',
              'Cartn_x', 'Cartn_y', 'Cartn_z', 'B_iso_or_equiv', 'pdbx_PDB_model_num']
    for row in block.find(['_atom_site.'+name for name in fields]):
        if row[0] not in mapping or row[2] in {'.', '?'} or row[8] != '1':
            continue
        pos = mapping[row[0]].get(int(row[2]))
        if pos is None or gemmi.cif.as_string(row[3]) not in {"C4'", 'P'}:
            continue
        atom = gemmi.cif.as_string(row[3])
        chain = chains.setdefault(row[1], {})
        if pos not in chain or atom == "C4'":
            chain[pos] = (tuple(float(row[i]) for i in (4, 5, 6)), float(row[7]))
            labels.setdefault(row[1], {})[pos] = int(row[2])
    if not chains:
        raise ValueError('No mapped RNA backbone coordinates')
    chain_id = sorted(chains, key=lambda c: (-len(chains[c]), c))[0]
    points = chains[chain_id]
    if len(points) < .8*len(wt):
        raise ValueError('Observed coordinates cover less than 80pct of WT')
    positions = {}
    for i, (xyz, confidence) in points.items():
        contacts = [j for j, (other, _) in points.items() if abs(i-j) > 3 and math.dist(xyz, other) <= 12]
        item = {'geometric_contact_degree_12A': len(contacts), 'contact_positions': sorted(contacts),
                'pdb_label_seq_id': labels[chain_id][i]}
        if predicted:
            item.update(plddt=round(confidence, 3), structural_support='low' if confidence < 70 else 'adequate')
        positions[str(i)] = item
    provenance = {'id': 'tertiary', 'sha256': file_hash(path), 'tool': provider, 'version': version}
    if artifact:
        provenance['artifact'] = artifact
    return {'wt_sha256': digest(wt_sequence), 'task': {'kind': 'predicted_WT' if predicted else 'experimental_WT_construct',
            'chain': chain_id, 'coordinate_coverage': len(points)/len(wt), 'mapped_positions': len(points),
            'contact_definition': "C4-prime/P <=12 A, sequence separation >3; not base-pair assignments",
            'limitations': 'One static WT conformation; geometric contacts and model confidence do not establish functional effects.'},
            'positions': positions, 'provenance': [provenance]}
