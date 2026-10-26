"""Offline preflight: rejoin every frozen label to the original measured source."""
from __future__ import annotations

import argparse
from pathlib import Path

from evagent.fewshot.data import load_landscape, sha256_file
from evagent.fewshot.runner import load, save
from evagent.integrated.gateway import public_model_config

from .workflow import build_task, task_requests


def audit(root):
    cohort = load(root / "COHORT.json")
    model = public_model_config(load(root / "model_config.json"))
    results = []
    for entry in cohort["assays"]:
        assay = entry["metadata"]["assay_id"]
        assert entry["public"] is True
        assert sha256_file(entry["source"]) == entry["expected_source_sha256"]
        landscape = load_landscape(entry["source"], entry["metadata"], entry["columns"])
        measured = {r["sequence"]: r["fitness"] for r in landscape.records}
        folder = root / "inputs" / assay
        public = load(folder / "public.json")
        oracle, truth = load(folder / "oracle.json"), load(folder / "evaluation.json")
        assert len(public["test"]) == 250
        assert len(public["pool"]) == 250 if entry["multi_round_eligible"] else len(public["pool"]) >= 32
        assert set(oracle).isdisjoint(truth)
        assert not ({r["sequence"] for r in public["test"]} & {r["sequence"] for r in public["pool"]})
        for r in public["test"]:
            assert truth[r["variant_id"]] == measured[r["sequence"]]
        for r in public["pool"]:
            assert oracle[r["variant_id"]] == measured[r["sequence"]]
        task = build_task(root, assay, "static", "anchored32", selected=public["random_order"][:32])
        static = task_requests(task, model)
        assert len(static) == 10 and all(len(r["support_ids"]) == 32 for r in static)
        for k in (4, 8, 16, 32):
            t = build_task(root, assay, "round/random", str(k), selected=public["random_order"][:k])
            reqs = task_requests(t, model)
            tests = [r for r in reqs if r["role"] == "test"]
            assert [r["expected_ids"] for r in tests] == [r["expected_ids"] for r in static]
            if k == 32:
                assert [r["request_hash"] for r in tests] == [r["request_hash"] for r in static]
        for split in entry["extrapolation"]:
            if not split["eligible"]:
                continue
            p = load(folder / split["split"] / "public.json")
            labels = load(folder / split["split"] / "evaluation.json")
            low = {1} if split["split"] == "1_to_2" else {1, 2}
            assert all(s["mutation_order"] in low and s["fitness"] == measured[s["sequence"]] for s in p["support"])
            assert all(labels[r["variant_id"]] == measured[r["sequence"]] for r in p["test"])
            assert all(r["mutation_order"] == 2 if low == {1} else r["mutation_order"] >= 3 for r in p["test"])
            assert not ({r["sequence"] for r in p["test"]} & {r["sequence"] for r in p["support"]})
        results.append({"assay": assay, "verified_original_measured_labels": len(truth) + len(oracle),
                        "source_sha256": entry["expected_source_sha256"], "passed": True})
        print(assay, "verified", flush=True)
    value = {"passed": True, "assays": results, "frozen_test_size": 250,
             "paid_calls": 0, "label_rule": "Only original experimental measurements; hidden labels are evaluator-only"}
    save(root / "preflight.json", value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
