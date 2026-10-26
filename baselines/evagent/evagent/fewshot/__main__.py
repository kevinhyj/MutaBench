"""Offline-first few-shot campaign commands. Generation requires explicit --live."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    DEFAULT_SEEDS,
    SPLITS,
    NotEligibleError,
    freeze_episode,
    load_landscape,
    write_episode,
)
from .runner import execute, freeze_plan, load, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    prepare = subs.add_parser("prepare", help="freeze one source landscape without model calls")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--metadata", type=Path, required=True)
    prepare.add_argument("--columns", type=Path, help="JSON mapping sequence/fitness/mutation/mutation_order to source columns")
    prepare.add_argument("--output", type=Path, required=True)
    for action in ("plan", "run"):
        p = subs.add_parser(action)
        p.add_argument("--episodes", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--model-config", type=Path, required=True)
        p.add_argument("--conditions", nargs='+', help="Registered condition names; omitted keeps the original eight-arm protocol")
        if action == "run":
            p.add_argument("--live", action="store_true")
            p.add_argument("--token-budget", type=int, required=True)
            p.add_argument("--max-units", type=int)
            p.add_argument("--workers", type=int, default=64,
                           help="Concurrent CR request workers sharing one ledger and budget (1-64; default 64)")
    baseline = subs.add_parser("baselines")
    baseline.add_argument("--episodes", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    report = subs.add_parser("report")
    report.add_argument("--episodes", type=Path, required=True)
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--baselines-dir", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        landscape = load_landscape(args.source, load(args.metadata), load(args.columns) if args.columns else None)
        index = []
        for split in SPLITS:
            for seed in DEFAULT_SEEDS:
                try:
                    frozen = freeze_episode(landscape, split, seed)
                    identity = frozen["episode"]["episode_id"]
                    paths = write_episode(frozen, args.output / "episodes" / identity)
                    index.append({"episode_id": identity, "status": "prepared", "paths": paths})
                except NotEligibleError as error:
                    index.append({"split": split, "seed": seed, "status": "ineligible", "reason": str(error)})
        save(args.output / (landscape.metadata["assay_id"] + ".preparation.json"), index)
        result = {"prepared": sum(r["status"] == "prepared" for r in index), "model_calls": 0}
    elif args.action == "plan":
        result = freeze_plan(args.episodes, args.run_dir, load(args.model_config), args.conditions)
        result = {k: result[k] for k in ("fingerprint", "planned_initial_requests", "maximum_generation_requests", "cost_estimate")}
        result["model_calls"] = 0
    elif args.action == "run":
        if not args.live:
            parser.error("run requires --live; use plan for a zero-call preflight")
        if args.max_units is not None and args.max_units < 1:
            parser.error("--max-units must be positive")
        if not 1 <= args.workers <= 64:
            parser.error("--workers must be between 1 and 64")
        result = execute(args.episodes, args.run_dir, load(args.model_config), token_budget=args.token_budget,
                         live=True, max_units=args.max_units, workers=args.workers, conditions=args.conditions)
    elif args.action == "baselines":
        from .reporting import run_baselines
        result = run_baselines(args.episodes, args.output)
    else:
        from .reporting import report
        result = report(args.episodes, args.run_dir, args.baselines_dir, args.output, plot=not args.no_plot)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
