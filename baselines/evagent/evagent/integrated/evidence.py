"""Build leak-resistant literature and structural evidence for an assay.

The module deliberately uses a strict metadata allowlist. Fitness values,
candidate labels, source-paper results, and free-form notes never enter an
evidence bundle. Benchmark calls use frozen local assets and do no live
retrieval, making the effective input reproducible.
"""

# Adapted from ypxia/src/rnagym_deep_agent/evidence.py; see provenance manifest.

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

EVIDENCE_PROTOCOL = "assay-evidence-v2"
EVIDENCE_MODES: dict[str, tuple[str, ...]] = {
    "none": (),
    "skill": ("skill",),
    "literature": ("literature",),
    "structure": ("structure",),
    "curated": ("skill", "literature", "structure"),
}
_PDB_RE = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")
_SAFE_METADATA_FIELDS = (
    "DMS_ID",
    "RNA_TYPE",
    "RNA_TYPE_GRANULAR",
    "ASSAY_DESCRIPTION",
    "PDB_STRUCTURE",
    "RFAM_LINK",
)
_SAFE_LITERATURE_FIELDS = (
    "id", "title", "doi", "pmid", "url", "tags", "agent_guidance", "limitations",
)
_MISSING_MARKERS = {"", "-", "na", "n/a", "none", "null"}
_WATER = {"HOH", "DOD", "WAT"}
_NUCLEOTIDES = {
    "A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DU",
    "PSU", "H2U", "5MC", "5MU", "1MA", "2MG", "M2G", "7MG",
    "OMC", "OMG", "YYG",
}
_BASE_MAP = {
    "A": "A", "C": "C", "G": "G", "U": "U", "I": "G",
    "DA": "A", "DC": "C", "DG": "G", "DT": "U", "DU": "U",
    "PSU": "U", "H2U": "U", "5MC": "C", "5MU": "U", "1MA": "A",
    "2MG": "G", "M2G": "G", "7MG": "G", "OMC": "C", "OMG": "G",
    "YYG": "G",
}


class EvidenceBundle(dict[str, Any]):
    """JSON-compatible typed evidence mapping with a detached export method.

    All evidence is frozen text or numeric structural summaries. This is not
    a native image, sequence-embedding, or live-retrieval backend.
    """

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self, ensure_ascii=False))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _clean_optional(value: str | None) -> str | None:
    text = str(value or "").strip()
    return None if text.lower() in _MISSING_MARKERS else text


def reference_sheet_path(rg_llm_root: Path) -> Path:
    return rg_llm_root / "data" / "RNAGym" / "fitness" / "reference_sheet_final.csv"


def load_assay_metadata(rg_llm_root: Path, assay: str) -> dict[str, str | None]:
    """Load only fields approved for model-facing assay evidence."""

    path = reference_sheet_path(rg_llm_root)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("DMS_ID") != assay:
                continue
            safe = {field: _clean_optional(raw.get(field)) for field in _SAFE_METADATA_FIELDS}
            safe["DMS_ID"] = assay
            return safe
    raise KeyError(f"Unknown assay in RNAGym reference sheet: {assay}")


def _load_construct_sequence(rg_llm_root: Path, assay: str) -> str:
    """Load the public WT construct only for PDB sequence matching.

    The sequence is already present in the benchmark prompt. It is never copied
    into the evidence bundle, keeping the bundle compact and preventing an
    accidental second source of candidate information.
    """

    with reference_sheet_path(rg_llm_root).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for raw in csv.DictReader(handle):
            if raw.get("DMS_ID") == assay:
                sequence = re.sub(r"[^ACGTU]", "", raw.get("RAW_CONSTRUCT_SEQ", "").upper())
                if not sequence:
                    raise ValueError(f"Missing construct sequence for assay: {assay}")
                return sequence.replace("T", "U")
    raise KeyError(f"Unknown assay in RNAGym reference sheet: {assay}")


def _joined_records(lines: list[str], record: str, start: int = 10) -> str | None:
    pieces = [line[start:].strip() for line in lines if line.startswith(record)]
    text = " ".join(piece for piece in pieces if piece)
    return " ".join(text.split()) or None


def parse_pdb_summary(path: Path) -> dict[str, Any]:
    """Reduce a PDB file to a compact, auditable model-facing summary."""

    data = path.read_bytes()
    lines = data.decode("utf-8", errors="replace").splitlines()
    chain_lengths: dict[str, int] = {}
    for line in lines:
        if not line.startswith("SEQRES"):
            continue
        chain = line[11:12].strip() or "_"
        try:
            chain_lengths[chain] = int(line[13:17].strip())
        except ValueError:
            continue

    coordinate_residues: set[tuple[str, str, str]] = set()
    hetero_names: set[str] = set()
    chains: set[str] = set()
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        residue = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        residue_id = (line[22:26].strip() + line[26:27].strip()) or "?"
        chains.add(chain)
        coordinate_residues.add((chain, residue_id, residue))
        if line.startswith("HETATM") and residue not in _WATER | _NUCLEOTIDES:
            hetero_names.add(residue)

    resolution = None
    for line in lines:
        if line.startswith("REMARK   2 RESOLUTION."):
            match = re.search(r"RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS", line)
            if match:
                resolution = float(match.group(1))
                break
    model_records = sum(line.startswith("MODEL ") for line in lines)
    return {
        "pdb_id": path.stem.upper(),
        "path": str(path),
        "sha256": _sha256_bytes(data),
        "title": _joined_records(lines, "TITLE "),
        "experimental_method": _joined_records(lines, "EXPDTA"),
        "resolution_angstrom": resolution,
        "model_count": model_records or 1,
        "chains": sorted(chains),
        "seqres_chain_lengths": dict(sorted(chain_lengths.items())),
        "coordinate_residue_count": len(coordinate_residues),
        "nonpolymer_or_unrecognized_residues": sorted(hetero_names),
        "interpretation_warning": (
            "Experimental structure may represent a homolog, truncation, ligand state, "
            "or only part of the assayed construct; transfer only covered features."
        ),
    }


def _seqres_sequences(lines: list[str]) -> dict[str, str]:
    residues: dict[str, list[str]] = {}
    for line in lines:
        if line.startswith("SEQRES"):
            chain = line[11:12].strip() or "_"
            residues.setdefault(chain, []).extend(line[19:70].split())
    return {
        chain: "".join(_BASE_MAP.get(name.upper(), "N") for name in names)
        for chain, names in residues.items()
    }


def _exact_construct_map(
    construct: str, chain_sequences: dict[str, str]
) -> tuple[str | None, dict[int, int], str]:
    """Return a conservative SEQRES-index to construct-index map."""

    candidates: list[tuple[int, str, dict[int, int], str]] = []
    for chain, sequence in chain_sequences.items():
        if sequence == construct:
            mapping = {index: index for index in range(len(sequence))}
            candidates.append((len(mapping), chain, mapping, "exact_full_construct"))
        elif sequence and sequence in construct:
            offset = construct.index(sequence)
            mapping = {index: offset + index for index in range(len(sequence))}
            candidates.append((len(mapping), chain, mapping, "exact_chain_subsequence"))
        elif construct and construct in sequence:
            offset = sequence.index(construct)
            mapping = {offset + index: index for index in range(len(construct))}
            candidates.append((len(mapping), chain, mapping, "exact_construct_subsequence"))
    if not candidates:
        return None, {}, "explicit_homolog_unmapped"
    _, chain, mapping, status = max(candidates, key=lambda item: (item[0], item[1]))
    return chain, mapping, status


def _alignment_observed_to_reference(reference: str, observed: str) -> dict[int, int]:
    """Needleman-Wunsch index map used only to account for unresolved residues."""

    rows, cols = len(reference) + 1, len(observed) + 1
    gap = -2
    score = [[0] * cols for _ in range(rows)]
    trace = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        score[i][0], trace[i][0] = i * gap, "U"
    for j in range(1, cols):
        score[0][j], trace[0][j] = j * gap, "L"
    for i in range(1, rows):
        for j in range(1, cols):
            choices = (
                (score[i - 1][j - 1] + (2 if reference[i - 1] == observed[j - 1] else -1), "D"),
                (score[i - 1][j] + gap, "U"),
                (score[i][j - 1] + gap, "L"),
            )
            score[i][j], trace[i][j] = max(choices, key=lambda item: item[0])
    i, j = len(reference), len(observed)
    mapping: dict[int, int] = {}
    while i or j:
        direction = trace[i][j]
        if direction == "D":
            if reference[i - 1] == observed[j - 1]:
                mapping[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif direction == "U":
            i -= 1
        else:
            j -= 1
    return mapping


def _coordinate_data(
    lines: list[str], chain: str
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[float, float, float]]]]:
    residues: dict[tuple[str, str], dict[str, Any]] = {}
    ligands: dict[str, list[tuple[float, float, float]]] = {}
    saw_model = False
    for line in lines:
        if line.startswith("MODEL "):
            if saw_model:
                break
            saw_model = True
            continue
        if saw_model and line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        residue_name = line[17:20].strip().upper()
        atom_name = line[12:16].strip().upper().replace("*", "'")
        atom_chain = line[21:22].strip() or "_"
        if atom_chain == chain and residue_name in _NUCLEOTIDES:
            key = (line[22:26].strip(), line[26:27].strip())
            item = residues.setdefault(
                key, {"base": _BASE_MAP.get(residue_name, "N"), "atoms": {}}
            )
            item["atoms"].setdefault(atom_name, xyz)
        elif line.startswith("HETATM") and residue_name not in _WATER | _NUCLEOTIDES:
            ligands.setdefault(residue_name, []).append(xyz)
    return list(residues.values()), ligands


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _position_features(
    path: Path, construct: str
) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    seqres = _seqres_sequences(lines)
    chain, seqres_to_construct, status = _exact_construct_map(construct, seqres)
    result: dict[str, Any] = {
        "status": status,
        "construct_length": len(construct),
        "matched_chain": chain,
        "mapped_seqres_positions": len(seqres_to_construct),
    }
    if chain is None:
        result["position_transfer_allowed"] = False
        return result

    residues, ligands = _coordinate_data(lines, chain)
    observed = "".join(item["base"] for item in residues)
    observed_to_seqres = _alignment_observed_to_reference(seqres[chain], observed)
    mapped: dict[int, dict[str, Any]] = {}
    for observed_index, item in enumerate(residues):
        seqres_index = observed_to_seqres.get(observed_index)
        if seqres_index is None or seqres_index not in seqres_to_construct:
            continue
        mapped[seqres_to_construct[seqres_index] + 1] = item

    representatives: dict[int, tuple[float, float, float]] = {}
    for position, item in mapped.items():
        atoms = item["atoms"]
        point = atoms.get("C4'") or atoms.get("P")
        if point is None and atoms:
            values = list(atoms.values())
            point = tuple(sum(atom[axis] for atom in values) / len(values) for axis in range(3))
        if point is not None:
            representatives[position] = point

    degrees = {position: 0 for position in representatives}
    long_range_pairs: list[tuple[int, int]] = []
    positions = sorted(representatives)
    for left_index, left in enumerate(positions):
        for right in positions[left_index + 1 :]:
            if right - left <= 3:
                continue
            if _distance(representatives[left], representatives[right]) <= 12.0:
                degrees[left] += 1
                degrees[right] += 1
                long_range_pairs.append((left, right))

    ligand_positions: dict[str, list[int]] = {}
    for ligand, ligand_atoms in ligands.items():
        nearby: list[int] = []
        for position, item in mapped.items():
            if any(
                _distance(atom, ligand_atom) <= 6.0
                for atom in item["atoms"].values()
                for ligand_atom in ligand_atoms
            ):
                nearby.append(position)
        if nearby:
            ligand_positions[ligand] = sorted(nearby)

    result.update(
        {
            "position_transfer_allowed": True,
            "mapped_coordinate_positions": len(mapped),
            "contact_definition": "C4-prime/P representative distance <=12 A; sequence separation >3",
            "high_contact_positions": [
                {"position": position, "contact_degree": degree}
                for position, degree in sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[:12]
                if degree > 0
            ],
            "long_range_contact_count": len(long_range_pairs),
            "ligand_proximal_positions_6A": ligand_positions,
            "feature_warning": "Geometric contacts are structural descriptors, not base-pair or fitness labels.",
        }
    )
    return result


def _pdb_id(metadata: dict[str, str | None]) -> str | None:
    text = metadata.get("PDB_STRUCTURE") or ""
    match = _PDB_RE.search(text)
    return match.group(1).upper() if match else None


def _structure_path(project_root: Path, pdb_id: str) -> Path | None:
    root = project_root / "knowledge" / "structures"
    candidates = (
        root / "fitness_assays_3D_structures" / f"{pdb_id.lower()}.pdb",
        root / "rcsb" / f"{pdb_id.lower()}.pdb",
    )
    return next((path for path in candidates if path.is_file()), None)


def load_literature_pack(project_root: Path) -> dict[str, Any]:
    path = project_root / "knowledge" / "literature" / "fitness_evidence_v1.json"
    pack = json.loads(path.read_text(encoding="utf-8"))
    if pack.get("pack_id") != "rna-fitness-general-literature-v1":
        raise ValueError(f"Unexpected literature evidence pack: {path}")
    return pack


def _literature_for(metadata: dict[str, str | None], pack: dict[str, Any]) -> list[dict[str, Any]]:
    rna_type = (metadata.get("RNA_TYPE") or "").lower()
    wanted = {"all", "benchmark", "dms"}
    if "coding" in rna_type:
        wanted.add("coding")
    else:
        wanted.update({"functional_rna", "secondary_structure", "chemical_probing"})
    selected = []
    for item in pack.get("items", []):
        if wanted.intersection(str(tag) for tag in item.get("tags", [])):
            selected.append({key: item[key] for key in _SAFE_LITERATURE_FIELDS if key in item})
    return selected


def build_evidence_bundle(
    *, assay: str, rg_llm_root: Path, project_root: Path | None = None,
    mode: str = "curated",
) -> EvidenceBundle:
    """Freeze the selected local modalities; disabled mode performs no reads."""
    if mode not in EVIDENCE_MODES:
        raise ValueError(f"Unknown evidence mode: {mode}")
    active = EVIDENCE_MODES[mode]
    if not active:
        disabled = EvidenceBundle({
            "schema_version": 1, "protocol": EVIDENCE_PROTOCOL,
            "assay": assay, "mode": mode, "representation": "frozen_text",
            "assay_context": {}, "fitness_skill": None, "literature": [],
            "structure": None, "source_files": {}, "unavailable_evidence": [],
            "unavailable_by_modality": {},
        })
        disabled["bundle_sha256"] = _canonical_hash(disabled)
        return disabled
    project_root = Path(project_root) if project_root is not None else Path(__file__).parent
    metadata = load_assay_metadata(rg_llm_root, assay)
    pack_path = project_root / "knowledge" / "literature" / "fitness_evidence_v1.json"
    skill_path = project_root / "knowledge" / "fitness_prediction_skill.md"
    source_path = project_root / "knowledge" / "structures" / "SOURCE.json"
    structure = None
    unavailable: list[str] = []
    unavailable_by_modality: dict[str, list[str]] = {name: [] for name in active}

    def missing(modality: str, reason: str) -> None:
        unavailable.append(reason)
        unavailable_by_modality[modality].append(reason)

    sources: dict[str, Any] = {
        "reference_sheet": {
            "path": str(reference_sheet_path(rg_llm_root)),
            "sha256": _sha256_bytes(reference_sheet_path(rg_llm_root).read_bytes()),
            "allowlisted_fields": list(_SAFE_METADATA_FIELDS),
        },
    }
    literature: list[dict[str, Any]] = []
    if "literature" in active:
        if pack_path.is_file():
            literature = _literature_for(metadata, load_literature_pack(project_root))
            sources["literature_pack"] = {
                "path": str(pack_path), "sha256": _sha256_bytes(pack_path.read_bytes()),
            }
        else:
            missing("literature", "Frozen literature pack is not available locally")
    skill = None
    if "skill" in active:
        if skill_path.is_file():
            skill = {"path": str(skill_path), "sha256": _sha256_bytes(skill_path.read_bytes())}
        else:
            missing("skill", "Frozen fitness skill is not available locally")
    pdb_id = _pdb_id(metadata)
    if "structure" in active and source_path.is_file():
        sources["structure_manifest"] = {
            "path": str(source_path), "sha256": _sha256_bytes(source_path.read_bytes()),
        }
    elif "structure" in active:
        missing("structure", "Frozen structure source manifest is not available locally")
    if "structure" in active and pdb_id:
        pdb_path = _structure_path(project_root, pdb_id)
        if pdb_path is None:
            missing("structure", f"Explicit PDB {pdb_id} is not available locally")
        else:
            structure = parse_pdb_summary(pdb_path)
            structure["mapping_basis"] = "Explicit PDB ID in RNAGym fitness reference sheet"
            try:
                construct = _load_construct_sequence(rg_llm_root, assay)
            except ValueError:
                structure["construct_mapping"] = {
                    "status": "construct_unavailable", "position_transfer_allowed": False,
                }
                missing("structure", "Public construct sequence unavailable for structure mapping")
            else:
                structure["construct_mapping"] = _position_features(pdb_path, construct)
    elif "structure" in active:
        missing("structure", "No explicit assay-to-PDB mapping in RNAGym reference sheet")
    if "structure" in active and metadata.get("RFAM_LINK") is None:
        missing("structure", "No explicit Rfam family mapping in RNAGym reference sheet")
    if "structure" in active:
        missing("structure", "ViennaRNA ensemble features unavailable: no frozen prediction asset")

    bundle = EvidenceBundle({
        "schema_version": 1,
        "protocol": EVIDENCE_PROTOCOL,
        "mode": mode,
        "representation": "frozen_text",
        "assay": assay,
        "assay_context": {
            "rna_type": metadata.get("RNA_TYPE"),
            "rna_type_granular": metadata.get("RNA_TYPE_GRANULAR"),
            "assay_description": metadata.get("ASSAY_DESCRIPTION"),
            "rfam_url": metadata.get("RFAM_LINK"),
            "explicit_pdb_id": pdb_id,
        },
        "fitness_skill": skill,
        "literature": literature,
        "structure": structure,
        "unavailable_evidence": unavailable,
        "unavailable_by_modality": unavailable_by_modality,
        "source_files": sources,
        "leakage_controls": [
            "fitness-value and free-text annotation columns are not read into the bundle",
            "candidate labels and rankings are never loaded by this module",
            "exact assay source papers are excluded from the curated literature pack",
            "benchmark calls perform no live retrieval",
        ],
    })
    bundle["bundle_sha256"] = _canonical_hash(bundle)
    return bundle


def render_evidence_context(
    bundle: dict[str, Any], *, modalities: tuple[str, ...] | None = None
) -> str:
    """Render a bounded evidence block for one benchmark prompt."""

    active = EVIDENCE_MODES[bundle.get("mode", "curated")] if modalities is None else modalities
    unknown = set(active) - {"skill", "literature", "structure"}
    if unknown:
        raise ValueError(f"Unknown evidence modalities: {sorted(unknown)}")
    if not active:
        return ""
    context = bundle["assay_context"]
    lines = [
        f"<assay_evidence protocol=\"{EVIDENCE_PROTOCOL}\">",
        "Use this frozen evidence as uncertain mechanistic context, never as candidate labels.",
        f"Active modalities: {', '.join(active)}.",
        f"RNA class: {context.get('rna_type')} ({context.get('rna_type_granular')}).",
        f"Measured phenotype: {context.get('assay_description')}.",
    ]
    if "structure" in active and context.get("rfam_url"):
        lines.append(f"Family evidence pointer: {context['rfam_url']} (family mapping, not a fitness label).")
    structure = bundle.get("structure")
    if "structure" in active and structure:
        mapping = structure.get("construct_mapping", {})
        lines.extend(
            [
                "Experimental structure summary:",
                f"- {structure['pdb_id']}: {structure.get('title')}",
                f"- Method={structure.get('experimental_method')}; resolution={structure.get('resolution_angstrom')}; "
                f"models={structure.get('model_count')}; chain lengths={structure.get('seqres_chain_lengths')}",
                f"- Other resolved residues/ligands={structure.get('nonpolymer_or_unrecognized_residues')}",
                f"- Construct mapping={mapping.get('status')}; matched chain={mapping.get('matched_chain')}; "
                f"position transfer allowed={mapping.get('position_transfer_allowed')}",
                f"- Caution: {structure.get('interpretation_warning')}",
            ]
        )
        if mapping.get("position_transfer_allowed"):
            lines.extend(
                [
                    f"- High-contact construct positions={mapping.get('high_contact_positions')}",
                    f"- Ligand-proximal construct positions (6 A)={mapping.get('ligand_proximal_positions_6A')}",
                    f"- Geometry definition/limit: {mapping.get('contact_definition')}; "
                    f"{mapping.get('feature_warning')}",
                ]
            )
        else:
            lines.append("- No position-level annotations are transferred from this homolog structure.")
    elif "structure" in active:
        lines.append("Experimental structure summary: unavailable for this assay.")
    if "literature" in active:
        lines.append("General literature guidance:")
        for item in bundle.get("literature", []):
            lines.append(f"- [{item['id']}] {item['agent_guidance']} Limitation: {item['limitations']}")
    if "skill" in active and bundle.get("fitness_skill"):
        lines.append(
            "Apply the fitness skill hierarchy: phenotype first; mutation audit; experimental/family evidence "
            "before predicted structure; coding consequences for mRNA-coding assays; explicit uncertainty."
        )
    if "unavailable_by_modality" in bundle:
        unavailable = [
            item for modality in active
            for item in bundle["unavailable_by_modality"].get(modality, [])
        ]
    else:
        unavailable = bundle.get("unavailable_evidence", [])
    if unavailable:
        lines.append("Unavailable frozen evidence: " + "; ".join(unavailable))
    lines.extend(
        [
            "Do not assume lower folding energy means higher fitness. Validate final ID coverage and uniqueness.",
            "</assay_evidence>",
        ]
    )
    return "\n".join(lines)


def write_evidence_bundle(bundle: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
