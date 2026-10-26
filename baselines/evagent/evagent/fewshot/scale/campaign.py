"""Frozen configuration and bounded, restartable batch scheduler."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from evagent.fewshot.protocol import digest
from evagent.fewshot.runner import GoogleContextCheck, load, save, sha
from evagent.integrated.gateway import public_model_config

from .execution import Executor, Journal, StopCampaign
from .report import report
from .workflow import scan_phase


def source_files():
    package = Path(__file__).resolve().parents[2]
    return sorted(package.rglob("*.py")) + [package.parent / "pyproject.toml"]


def freeze(root):
    paths = sorted((root / "inputs").rglob("*.json")) + [root / n for n in
                ("COHORT.json", "screening.json", "model_config.json")]
    config = {"budget": 200_000_000, "seed": 1, "batch_size": 25,
              "source_hashes": {str(p): sha(p) for p in source_files()},
              "input_hashes": {str(p.relative_to(root)): sha(p) for p in paths},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[3], text=True).strip(),
              "dependencies": {name: importlib.metadata.version(name) for name in
                               ("google-genai", "numpy", "scipy", "langchain-core")},
              "phases": ["pilot_static", "pilot_rounds", "static", "rounds", "extrapolation"],
              "workers": {"initial": 8, "after_20_good": 15, "floor": 2},
              "retry": {"max_total_submissions_per_logical_request_including_repairs": 3,
                        "backoff_seconds": 60, "max_format_repairs": 1},
              "authorization": "User explicitly authorized additional 200M tokens, seed1 and server automatic recovery",
              "budget_scope": "New independent campaign; preserve historical known/unknown costs",
              "model": public_model_config(load(root / "model_config.json"))}
    config["fingerprint"] = digest(config)
    path = root / "FROZEN.json"
    if path.exists() and load(path) != config:
        raise ValueError("Refusing to replace frozen experiment")
    save(path, config)
    return config


def verify(root, config):
    if config["fingerprint"] != digest({k: v for k, v in config.items() if k != "fingerprint"}):
        raise ValueError("Frozen manifest corrupted")
    if set(config["source_hashes"]) != {str(p) for p in source_files()}:
        raise ValueError("Source inventory changed")
    for path, expected in config["source_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Frozen source hash mismatch")
    for path, expected in config["input_hashes"].items():
        if sha(root / path) != expected:
            raise ValueError("Frozen input hash mismatch")
    for name, version in config["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Dependency version changed")


def credentials(project):
    plan = load(project / "experiments/proteingym_cr_msa_remainder_v1/plan.json")
    path = Path(plan["credentials_file"])
    for line in path.read_text().splitlines():
        text = line.strip().removeprefix("export ")
        if text.startswith("GEMINI_API_KEY="):
            values = shlex.split(text.split("=", 1)[1])
            if len(values) == 1:
                os.environ["GEMINI_API_KEY"] = values[0]
                break
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Missing provider credential")
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def run(root):
    root = root.resolve()
    lease = (root / "worker.lock").open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return
    from .watchdog import identity
    process = identity(os.getpid())
    if process is not None:
        save(root / "process.json", {"pid": os.getpid(), "start_ticks": process["start_ticks"],
                                    "command": [x.decode() for x in process["command"]], "started": time.time()})
    journal = None
    state = "starting"
    last_report = last_snapshot = 0
    progress = {}
    def refresh(force=False):
        nonlocal last_report, last_snapshot, progress
        if journal is None:
            return
        now = time.time()
        accounting = journal.summary()
        if force or now - last_report >= 180:
            progress = report(root, accounting, state)
            last_report = now
        if force or now - last_snapshot >= 900:
            journal.snapshot()
            last_snapshot = now
        save(root / "STATUS.json", {"state": state, "time": now, "pid": os.getpid(),
                                    "accounting": accounting, "progress": progress})
    try:
        if (root / "PAUSE").exists() or (root / "STOP.json").exists() or (root / "DONE.json").exists():
            return
        config = load(root / "FROZEN.json")
        verify(root, config)
        journal = Journal(root, config["budget"], config["fingerprint"])
        recovered = journal.recover()
        save(root / "recovery" / f"{int(time.time())}.json", {"recovered_attempt_ids": recovered, "time": time.time()})
        credentials(root.parents[1])
        checker = GoogleContextCheck(config["model"], root / "token_checks")
        executor = Executor(journal, config["model"], checker)
        cohort = load(root / "COHORT.json")
        all_assays = [e["metadata"]["assay_id"] for e in cohort["assays"]]
        for phase in config["phases"]:
            state = phase
            assays = cohort["pilots"] if phase.startswith("pilot_") else all_assays
            kind = phase.removeprefix("pilot_")
            while True:
                journal.check()
                verify(root, config)
                pending, results = scan_phase(root, assays, kind, config["model"])
                refresh()
                if not pending:
                    # Do not expand a pilot whose end-to-end validation failed.
                    if phase.startswith("pilot_") and any(r["status"] != "complete" for r in results):
                        save(root / "STOP.json", {"reason": "pilot_validation_failed", "time": time.time()})
                        raise StopCampaign("pilot_validation_failed")
                    break
                workers = journal.concurrency()
                batch = journal.start_batch(workers)
                # Continuous worker occupancy inside an immutable batch. Drain
                # after 20 initial requests / 50 later requests before adapting.
                batch_size = 20 if workers == 8 and journal.summary()["attempts"] < 20 else 50
                batch_requests = pending[:batch_size]
                save(root / "batches" / f"{batch:06d}.json", {"workers": workers, "phase": phase,
                     "requests": [r["request_hash"] for r in batch_requests], "time": time.time()})
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(executor.run, r, batch) for r in batch_requests]
                    waiting = set(futures)
                    while waiting:
                        _, waiting = wait(waiting, timeout=5)
                        refresh()
                    for future in futures:
                        future.result()
                journal.end_batch(batch)
        state = "complete"
        save(root / "DONE.json", {"time": time.time(), "state": state})
    except StopCampaign as exc:
        state = str(exc)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        state = "integrity_or_configuration_stop"
        save(root / "STOP.json", {"reason": state, "error_type": type(exc).__name__, "time": time.time()})
        # Exception texts may contain paths/provider secrets: record safe types only.
    except Exception as exc:
        state = "unexpected_exit_recoverable"
        save(root / "last_error.json", {"error_type": type(exc).__name__, "time": time.time()})
        raise
    finally:
        refresh(force=True)
        if journal is not None:
            journal.db.close()
        lease.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.root.resolve())
    else:
        run(args.root)
