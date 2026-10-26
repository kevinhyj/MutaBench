"""Actual local structure calculations against immutable, source-admitted inputs."""

from __future__ import annotations

import fcntl
import importlib.metadata
import itertools
import json
import math
import tempfile
from pathlib import Path

from ..gateway import ExecutionPaused, digest
from ..structure_tools import cif_features
from ..tasks import file_hash
from .contracts import save

TOOL_PARAMETERS = {
    "temperature_c": 37.0,
    "minimum_predicted_plddt": 70,
    "max_candidates": 25,
    "max_selected_positions": 12,
    "max_joint_substitutions": 12,
    "pair_rows_per_candidate": 8,
    "contact_rows_per_candidate": 12,
    "annotation_rows_per_candidate": 12,
    "ensemble_gamma": 1.0,
}
EXPERIMENTAL_KINDS = {"experimental", "experimental_WT", "experimental_WT_construct"}
TORSIONS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]


def canonical(sequence):
    sequence = sequence.upper().replace("T", "U")
    if not sequence or set(sequence) - set("ACGU"):
        raise ValueError("Structure tools require an unambiguous complete RNA sequence")
    return sequence


def changed(wt, mutant):
    if len(wt) != len(mutant):
        return None
    return [i for i, (a, b) in enumerate(zip(wt, mutant), 1) if a != b]


def coordinate_asset(snapshot, snapshot_path):
    """Resolve only a hashed snapshot asset, never a model-supplied path or URL."""
    tertiary = snapshot["sources"]["structure"].get("task", {}).get("tertiary_structure", {})
    kind = tertiary.get("kind")
    if kind not in EXPERIMENTAL_KINDS | {"predicted_WT"}:
        return {"status": "unavailable", "reason": "missing_or_unknown_WT_structure_kind"}
    root = Path(snapshot_path).resolve().parent
    cifs = [
        a
        for a in snapshot.get("assets", [])
        if Path(a["path"]).suffix.lower() in {".cif", ".mmcif"}
    ]
    if len(cifs) != 1:
        return {"status": "unavailable", "reason": "missing_or_ambiguous_frozen_coordinate_asset"}
    asset = cifs[0]
    path = (root / asset["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != asset["sha256"]:
        raise ExecutionPaused("Frozen structure asset missing, changed, or outside snapshot root")
    return {
        "status": "available",
        "path": str(path),
        "sha256": asset["sha256"],
        "kind": kind,
        "predicted": kind == "predicted_WT",
        "msa_dependency": tertiary.get("msa_dependency", "not_recorded"),
    }


def preflight(task, snapshot, snapshot_path):
    wt = canonical(task["wt_sequence"])
    for v in task["variants"]:
        canonical(v["sequence"])
    asset = coordinate_asset(snapshot, snapshot_path)
    result = {
        "wt_length": len(wt),
        "candidates": len(task["variants"]),
        "secondary_supported": sum(len(v["sequence"]) == len(wt) for v in task["variants"]),
        "geometry_status": asset["status"],
        "geometry_reason": asset.get("reason"),
        "geometry_asset_sha256": asset.get("sha256"),
        "eligible_geometry_positions": [],
        "geometry_kind": asset.get("kind"),
    }
    if asset["status"] == "available":
        try:
            data = cif_features(
                Path(asset["path"]),
                wt,
                predicted=asset["predicted"],
                provider="Gemmi",
                version="active-v1",
            )
        except ValueError as exc:
            return {
                **result,
                "geometry_status": "unavailable",
                "geometry_reason": "WT_mapping_failed: " + str(exc),
            }
        result["mapped_geometry_positions"] = len(data["positions"])
        result["eligible_geometry_positions"] = [
            int(p)
            for p, r in data["positions"].items()
            if not asset["predicted"]
            or (
                isinstance(r.get("plddt"), (int, float))
                and r["plddt"] >= TOOL_PARAMETERS["minimum_predicted_plddt"]
            )
        ]
        if not result["eligible_geometry_positions"]:
            result.update(
                geometry_status="withheld",
                geometry_reason="no_positions_pass_frozen_confidence_threshold",
            )
    return result


class ComputationCache:
    """A shared, verified cache. Hits still produce a tool-call journal entry."""

    def __init__(self, root):
        self.root = Path(root)
        self.signature = {
            "source_sha256": file_hash(Path(__file__)),
            "parameters": TOOL_PARAMETERS,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("ViennaRNA", "gemmi", "barnaba", "mdtraj", "numpy", "scipy")
            },
            "geometry_adapter_sha256": file_hash(Path(__file__).with_name("structure.py")),
            "mapping_sha256": file_hash(Path(__file__).parents[1] / "structure_tools.py"),
        }
        self.events = []

    def get(self, operation, inputs, compute):
        key = digest({"operation": operation, "inputs": inputs, "runtime": self.signature})
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (key + ".json")
        with (self.root / (key + ".lock")).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            hit = path.exists()
            if hit:
                try:
                    value = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    raise ExecutionPaused("Structure computation cache is unreadable") from exc
                if (
                    value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"})
                    or value.get("key") != key
                ):
                    raise ExecutionPaused("Structure computation cache integrity failure")
            else:
                try:
                    output = compute()
                except Exception as exc:
                    raise ExecutionPaused(
                        "Structure computation failed: " + type(exc).__name__
                    ) from exc
                value = {
                    "key": key,
                    "operation": operation,
                    "inputs": inputs,
                    "runtime": self.signature,
                    "value": output,
                }
                value["sha256"] = digest(value)
                save(path, value)
            self.events.append(
                {
                    "computation_key": key,
                    "operation": operation,
                    "cache_hit": hit,
                    "result_sha256": value["sha256"],
                    "input_sequence_sha256": digest(inputs["sequence"])
                    if "sequence" in inputs
                    else None,
                }
            )
            return value["value"]


class StructureAnalysis:
    def __init__(self, task, snapshot, snapshot_path, cache_root):
        self.task, self.snapshot = task, snapshot
        self.snapshot_path = Path(snapshot_path)
        self.wt = canonical(task["wt_sequence"])
        self.variants = {v["variant_id"]: v for v in task["variants"]}
        self.cache = ComputationCache(cache_root)
        self.coverage = preflight(task, snapshot, snapshot_path)

    def fold(self, sequence):
        sequence = canonical(sequence)

        def compute():
            import numpy as np
            import RNA

            md = RNA.md()
            md.temperature = TOOL_PARAMETERS["temperature_c"]
            fc = RNA.fold_compound(sequence, md)
            dot, energy = fc.mfe()
            fc.exp_params_rescale(energy)
            _, ensemble = fc.pf()
            matrix = np.asarray(fc.bpp())[1:, 1:]
            matrix = matrix + matrix.T
            centroid, centroid_distance = fc.centroid()
            mea, accuracy = fc.MEA(TOOL_PARAMETERS["ensemble_gamma"])
            return {
                "dot_bracket": dot,
                "mfe_kcal_mol": float(energy),
                "ensemble_kcal_mol": float(ensemble),
                "ensemble_diversity": float(fc.mean_bp_distance()),
                "centroid": centroid,
                "centroid_distance": float(centroid_distance),
                "mea": mea,
                "mea_expected_accuracy": float(accuracy),
                "pair_probabilities": matrix.tolist(),
            }

        return self.cache.get("ViennaRNA.complete_sequence", {"sequence": sequence}, compute)

    def _selection(self, ids, positions):
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= TOOL_PARAMETERS["max_candidates"]
            or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
            or set(ids) - self.variants.keys()
        ):
            raise ValueError("Select distinct candidates from the current group")
        if positions is not None and (
            not isinstance(positions, list)
            or not 1 <= len(positions) <= TOOL_PARAMETERS["max_selected_positions"]
            or any(type(i) is not int or not 1 <= i <= len(self.wt) for i in positions)
            or len(set(positions)) != len(positions)
        ):
            raise ValueError("Positions must be distinct 1-based full WT coordinates")
        return ids

    def secondary(self, ids, *, mode="pairing", positions=None):
        self._selection(ids, positions)
        if mode not in {"pairing", "ensemble", "joint_changes"}:
            raise ValueError("Unknown secondary analysis mode")
        import numpy as np
        import RNA

        wt = self.fold(self.wt)
        a = np.asarray(wt["pair_probabilities"])
        cards = []
        for vid in ids:
            mutant = canonical(self.variants[vid]["sequence"])
            sites = changed(self.wt, mutant)
            card = {
                "variant_id": vid,
                "status": "available",
                "coordinate_system": "1-based full WT",
                "wt_sequence_sha256": digest(self.wt),
                "candidate_sequence_sha256": digest(mutant),
            }
            if sites is None:
                cards.append(
                    {
                        **card,
                        "status": "unavailable",
                        "reason": "indel_coordinate_mapping_not_supported",
                    }
                )
                continue
            selected = sorted(positions if positions is not None else sites)
            folded = self.fold(mutant)
            b = np.asarray(folded["pair_probabilities"])
            card.update(
                changed_positions=sites,
                selected_positions=selected,
                delta_mfe_kcal_mol=folded["mfe_kcal_mol"] - wt["mfe_kcal_mol"],
                delta_ensemble_kcal_mol=folded["ensemble_kcal_mol"] - wt["ensemble_kcal_mol"],
            )
            if mode == "pairing":
                indices = sorted(set(selected))
                pairs = {
                    (min(i, j), max(i, j))
                    for i in indices
                    for j in range(1, len(self.wt) + 1)
                    if i != j
                }
                changes = [
                    {
                        "positions": [i, j],
                        "wt_probability": float(a[i - 1, j - 1]),
                        "candidate_probability": float(b[i - 1, j - 1]),
                        "delta_probability": float(b[i - 1, j - 1] - a[i - 1, j - 1]),
                    }
                    for i, j in pairs
                    if max(a[i - 1, j - 1], b[i - 1, j - 1]) >= 0.01
                ]
                changes.sort(key=lambda x: (-abs(x["delta_probability"]), x["positions"]))
                limit = TOOL_PARAMETERS["pair_rows_per_candidate"]
                card.update(
                    pair_changes=changes[:limit],
                    omitted_pair_rows=max(0, len(changes) - limit),
                    pair_probability_floor=0.01,
                    unpaired_probability=[
                        {
                            "position": i,
                            "wt": max(0.0, 1 - float(a[i - 1].sum())),
                            "candidate": max(0.0, 1 - float(b[i - 1].sum())),
                        }
                        for i in indices
                    ],
                    mfe_pair_distance=int(
                        RNA.bp_distance(wt["dot_bracket"], folded["dot_bracket"])
                    ),
                    mean_pair_probability_change=float(abs(a - b).sum() / len(self.wt)),
                )
            elif mode == "ensemble":
                card.update(
                    wt={k: v for k, v in wt.items() if k != "pair_probabilities"},
                    candidate={k: v for k, v in folded.items() if k != "pair_probabilities"},
                    centroid_pair_distance=int(RNA.bp_distance(wt["centroid"], folded["centroid"])),
                    mea_pair_distance=int(RNA.bp_distance(wt["mea"], folded["mea"])),
                )
            elif not 2 <= len(sites) <= TOOL_PARAMETERS["max_joint_substitutions"]:
                card.update(
                    status="unavailable", reason="joint_analysis_requires_2_to_12_substitutions"
                )
            else:
                singles = []
                for i in sites:
                    sequence = self.wt[: i - 1] + mutant[i - 1] + self.wt[i:]
                    one = self.fold(sequence)
                    singles.append(
                        {
                            "position": i,
                            "sequence_sha256": digest(sequence),
                            "delta_mfe_kcal_mol": one["mfe_kcal_mol"] - wt["mfe_kcal_mol"],
                            "delta_ensemble_kcal_mol": one["ensemble_kcal_mol"]
                            - wt["ensemble_kcal_mol"],
                        }
                    )
                card.update(
                    single_change_counterfactuals=singles,
                    nonadditivity_mfe_kcal_mol=card["delta_mfe_kcal_mol"]
                    - sum(s["delta_mfe_kcal_mol"] for s in singles),
                    nonadditivity_ensemble_kcal_mol=card["delta_ensemble_kcal_mol"]
                    - sum(s["delta_ensemble_kcal_mol"] for s in singles),
                    mutated_site_pairing=[
                        {
                            "positions": [i, j],
                            "wt_probability": float(a[i - 1, j - 1]),
                            "candidate_probability": float(b[i - 1, j - 1]),
                        }
                        for i, j in itertools.combinations(sites, 2)
                    ],
                )
            card["limitations"] = [
                "Thermodynamic structure changes are not measured function or functional epistasis.",
                "Full sequences, fixed 37 C, equilibrium and pseudoknot-free model; no partners or kinetics.",
            ]
            cards.append(card)
        return cards

    def geometry(self):
        asset = coordinate_asset(
            self.snapshot, self.snapshot_path
        )  # Recheck bytes even on cache hits.
        if self.coverage["geometry_status"] == "unavailable" or asset["status"] != "available":
            return {"status": "unavailable", "reason": self.coverage.get("geometry_reason")}

        def compute():
            import gemmi

            from .structure import annotate_cif

            with tempfile.TemporaryDirectory(dir=self.cache.root, prefix="mapped-wt-") as folder:
                pdb = Path(folder) / "WT.pdb"
                result = annotate_cif(
                    Path(asset["path"]), self.wt, pdb, predicted=asset["predicted"]
                )
                atoms = {}
                for chain in gemmi.read_structure(str(pdb))[0]:
                    for residue in chain:
                        points = [
                            [a.pos.x, a.pos.y, a.pos.z]
                            for a in residue
                            if not a.element.is_hydrogen
                        ]
                        atoms[str(residue.seqid.num)] = points
                result["heavy_atoms"] = atoms
                result["coordinate_asset_sha256"] = asset["sha256"]
                result["geometry_kind"] = asset["kind"]
                result["msa_dependency"] = asset["msa_dependency"]
                return result

        return self.cache.get(
            "Barnaba.Gemmi.WT_geometry",
            {"asset_sha256": asset["sha256"], "wt": self.wt, "kind": asset["kind"]},
            compute,
        )

    def tertiary(self, ids, *, mode, positions=None, radius_A=5.0):
        self._selection(ids, positions)
        if mode not in {"contacts", "annotations"}:
            raise ValueError("Unknown tertiary analysis")
        if (
            type(radius_A) not in (float, int)
            or not math.isfinite(radius_A)
            or not 3 <= radius_A <= 12
        ):
            raise ValueError("Contact radius must be between 3 and 12 Angstrom")
        geometry = self.geometry()
        allowed = set(self.coverage["eligible_geometry_positions"])
        cards = []
        for vid in ids:
            sites = changed(self.wt, canonical(self.variants[vid]["sequence"]))
            queried = positions if positions is not None else sites
            selected = sorted(set(queried or []) & allowed)
            card = {
                "variant_id": vid,
                "status": "available",
                "changed_positions": sites,
                "selected_positions": sorted(queried or []),
                "usable_positions": selected,
                "withheld_positions": sorted(set(queried or []) - allowed),
                "geometry_kind": geometry.get("geometry_kind"),
                "coordinate_asset_sha256": geometry.get("coordinate_asset_sha256"),
                "msa_dependency": geometry.get("msa_dependency"),
                "limitations": [
                    "Static WT geometry, never a prediction of mutant 3D coordinates or function.",
                    "Predicted geometry requires pLDDT >=70 at both endpoints; local confidence does not establish domain orientation.",
                ],
            }
            if geometry["status"] != "available" or sites is None or not selected:
                card.update(
                    status="unavailable",
                    reason=(
                        geometry.get("reason")
                        or (
                            "indel_coordinate_mapping_not_supported"
                            if sites is None
                            else "no_selected_positions_pass_mapping_and_confidence"
                        )
                    ),
                )
                cards.append(card)
                continue
            card["site_confidence"] = {
                str(i): geometry["positions"][str(i)].get("plddt") for i in selected
            }
            if mode == "contacts":
                from scipy.spatial.distance import cdist

                def distance(i, j):
                    return float(
                        cdist(
                            geometry["heavy_atoms"][str(i)], geometry["heavy_atoms"][str(j)]
                        ).min()
                    )

                contacts = [
                    {"positions": [i, j], "minimum_heavy_atom_distance_A": distance(i, j)}
                    for i in selected
                    for j in sorted(allowed)
                    if abs(i - j) > 1
                ]
                contacts = [c for c in contacts if c["minimum_heavy_atom_distance_A"] <= radius_A]
                contacts.sort(key=lambda c: (c["minimum_heavy_atom_distance_A"], c["positions"]))
                limit = TOOL_PARAMETERS["contact_rows_per_candidate"]
                joint = sorted(set(sites) & allowed)
                pairs = list(itertools.combinations(joint, 2))
                card.update(
                    radius_A=radius_A,
                    contacts=contacts[:limit],
                    omitted_contacts=max(0, len(contacts) - limit),
                    contact_definition="minimum heavy-atom distance; immediate sequence neighbors excluded",
                    mutated_site_distances=[
                        {"positions": [i, j], "minimum_heavy_atom_distance_A": distance(i, j)}
                        for i, j in pairs[:66]
                    ],
                    omitted_mutated_site_pairs=max(0, len(pairs) - 66),
                )
            else:
                annotations = [
                    a
                    for a in geometry["annotations"]
                    if set(a["positions"]) & set(selected) and set(a["positions"]) <= allowed
                ]
                torsions = {}
                for i in selected:
                    values = geometry["positions"][str(i)].get("backbone_angles_radians")
                    if values is not None:
                        # alpha depends on i-1; epsilon/zeta on i+1. Mask untrusted neighbors.
                        torsions[str(i)] = {
                            name: v
                            if (
                                {i - 1, i}
                                if name == "alpha"
                                else {i, i + 1}
                                if name in {"epsilon", "zeta"}
                                else {i}
                            )
                            <= allowed
                            else None
                            for name, v in zip(TORSIONS, values)
                        }
                limit = TOOL_PARAMETERS["annotation_rows_per_candidate"]
                card.update(
                    annotations=annotations[:limit],
                    omitted_annotations=max(0, len(annotations) - limit),
                    backbone_torsions_radians=torsions,
                    annotation_definition="Barnaba base-pair and stacking classifications; missing angles are null",
                )
            cards.append(card)
        return cards
