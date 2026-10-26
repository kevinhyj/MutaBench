"""RNA-specific structural annotations mapped to checked WT coordinates."""

from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path
from typing import Protocol

from ..structure_tools import cif_features
from ..tasks import file_hash


class DSSRAdapter(Protocol):
    """Optional licensed implementation; deliberately absent from the frozen tool whitelist."""

    def annotate(self, coordinate_file: Path, wt_sequence: str) -> dict: ...


def annotate_cif(path: Path, wt: str, output: Path, *, predicted: bool):
    import barnaba as bb
    import gemmi
    import mdtraj as md
    from scipy.spatial.distance import cdist

    checked = cif_features(path, wt, predicted=predicted, provider="frozen-WT", version="v1")
    chain = checked["task"]["chain"]
    mapping = {int(v["pdb_label_seq_id"]): int(k) for k, v in checked["positions"].items()}
    block = gemmi.cif.read_file(str(path)).sole_block()
    fields = [
        "label_asym_id",
        "label_seq_id",
        "label_atom_id",
        "label_comp_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "type_symbol",
        "pdbx_PDB_model_num",
    ]
    lines, atom_points, used = [], {}, set()
    for row in block.find(["_atom_site." + x for x in fields]):
        if row[7] in {"H", "D"}:
            continue
        if row[0] != chain or row[1] in {".", "?"} or row[8] != "1" or int(row[1]) not in mapping:
            continue
        pos = mapping[int(row[1])]
        atom = gemmi.cif.as_string(row[2])
        base = row[3]
        if base not in "ACGU" or base != wt[pos - 1].replace("T", "U"):
            raise ValueError("Atom residue differs from mapped WT")
        if (pos, atom) in used:
            continue
        used.add((pos, atom))
        xyz = tuple(float(row[i]) for i in (4, 5, 6))
        if not all(math.isfinite(x) for x in xyz):
            raise ValueError("Nonfinite coordinate")
        atom_points.setdefault(pos, []).append(xyz)
        lines.append(
            f"ATOM  {len(lines) + 1:5d} {atom:>4} {base:>3} A{pos:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {row[7]:>2}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\nTER\nEND\n")
    stack, pair, residues = bb.annotate(str(output))
    traj = md.load(str(output))
    topology_positions = [r.resSeq for r in traj.topology.residues]
    annotations = []
    # Barnaba filters residues lacking base atoms; its identifiers include resSeq.
    position_map = []
    for name in residues:
        parts = str(name).split("_")
        position_map.append(int(parts[1]))
    for kind, frames in (("stacking", stack), ("base_pair", pair)):
        for indices, label in zip(frames[0][0], frames[0][1]):
            annotations.append(
                {
                    "kind": kind,
                    "positions": [position_map[int(i)] for i in indices],
                    "classification": str(label),
                }
            )
    angles, angle_residues = bb.backbone_angles(str(output))
    torsions = {
        int(str(r).split("_")[1]): [
            float(x) if math.isfinite(float(x)) else None for x in angles[0, i]
        ]
        for i, r in enumerate(angle_residues)
    }
    positions = {}
    for pos, xyzs in atom_points.items():
        distances = []
        for other, points in atom_points.items():
            if abs(other - pos) <= 1:
                continue
            distance = float(cdist(xyzs, points).min())
            if distance <= 5:
                distances.append(
                    {"position": other, "minimum_heavy_atom_distance_A": round(distance, 3)}
                )
        positions[str(pos)] = {
            "contacts_within_5A": distances,
            "backbone_angles_radians": torsions.get(pos),
            **checked["positions"].get(str(pos), {}),
        }
    return {
        "status": "available",
        "annotations": annotations,
        "positions": positions,
        "mapping": {
            "chain": chain,
            "label_seq_to_WT": {str(k): v for k, v in mapping.items()},
            "pdb_resSeq": topology_positions,
        },
        "provenance": [
            {"tool": n, "version": importlib.metadata.version(n)}
            for n in ("barnaba", "mdtraj", "gemmi")
        ]
        + [{"input_sha256": file_hash(path), "mapped_pdb_sha256": file_hash(output)}],
        "limitations": [
            "WT conformation only; no mutant tertiary structures or functional effects inferred.",
            "Backbone torsion order alpha,beta,gamma,delta,epsilon,zeta,chi; radians. Missing angles are null.",
            "Annotations inherit original coordinate and model uncertainty.",
        ],
    }
