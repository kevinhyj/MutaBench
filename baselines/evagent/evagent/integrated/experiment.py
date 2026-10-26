"""Unified, versioned experiments with read-only context and one provider ledger."""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evagent.runtime import ExecutionPaused, digest

from .config import ExperimentConfig
from .evidence import EVIDENCE_MODES, build_evidence_bundle, render_evidence_context
from .learning_memory import TaskContext
from .memory import MemoryProvider
from .protocols import build_protocol_tasks, execute_protocol
from .tasks import evaluate_result, file_hash, load_task


def _implementation() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))}
    dependencies = {}
    for name in ("langgraph", "langchain-core", "langgraph-checkpoint-sqlite", "deepagents",
                 "google-genai", "numpy", "scipy", "pydantic"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"source_sha256": digest(files), "source_files": files,
            "python": platform.python_version(), "dependencies": dependencies}


def prepare(config: ExperimentConfig) -> tuple[Any, dict[str, Any], str]:
    values = config.values
    task = load_task(values)
    def optional_path(name):
        return Path(values[name]) if values.get(name) else None

    context = TaskContext(
        rna_type=str(task.metadata.get("RNA_TYPE", "unknown")),
        output_mode="scoring" if values["method"] == "cr" else "ranking",
    )
    memory = MemoryProvider(
        mode=values["memory_mode"], basic_db=optional_path("basic_memory_db"),
        learning_db=optional_path("learning_memory_db"), strategy_file=optional_path("strategy_file"),
    ).snapshot(context=context, forbidden_assays=[task.assay])
    mode = values["evidence_mode"]
    if mode not in {"none", "bundle"} and not values.get("rg_llm_root"):
        raise ValueError("Evidence modes require rg_llm_root with frozen reference metadata")
    evidence = build_evidence_bundle(
        assay=task.assay, rg_llm_root=optional_path("rg_llm_root") or Path("."),
        project_root=optional_path("knowledge_root"), mode="none" if mode == "bundle" else mode,
    )
    evidence_text = "" if mode in {"none", "bundle"} else render_evidence_context(evidence, modalities=EVIDENCE_MODES[mode])
    sections = []
    if memory["text"]:
        sections.append("Frozen long-term memory:\n" + memory["text"])
    if evidence_text:
        sections.append("Frozen textual evidence:\n" + evidence_text)
    context_text = "\n\n".join(sections)
    frozen = None
    if mode == "bundle":
        from .bundles import FrozenEvidence
        frozen = FrozenEvidence(Path(values["evidence_bundle_file"]), assay=task.assay,
                                wt_sequence=task.wt_sequence, variants=task.variants,
                                sources=values["evidence_sources"])
        if not frozen.available:
            raise ValueError("Selected evidence is unavailable; mark this campaign cell ineligible")
        prefix = context_text
        def render_chunk(ids):
            return "\n\n".join(filter(None, [prefix, frozen.render(ids)]))
        context_text = render_chunk
    tasks = []
    if values["method"] != "workflow":
        tasks = build_protocol_tasks(
            values["method"], case=task.case, assay_memory=task.assay_memory,
            context_text=context_text, prompt_version=values["prompt_version"],
            group_size=values["group_size"], anchor_count=values["anchor_count"], seed=values["seed"],
        )
    prepared = [asdict(value) for value in tasks]
    manifest = {
        "schema_version": 1, "config": values, "assay": task.assay,
        "n": len(task.case.ids), "input_hashes": task.input_hashes,
        "implementation": _implementation(), "memory": memory,
        "evidence": frozen.bundle if frozen else evidence.as_dict(),
        "evidence_representation": "text", "context_sha256": digest(
            [x.user for x in tasks] if frozen else context_text),
        "initial_requests": prepared,
        "ground_truth_in_model_input": False,
        "long_term_memory_write_enabled": False,
        "benchmark_tools_enabled": False,
    }
    manifest["fingerprint"] = digest(manifest)
    return task, manifest, context_text


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run_experiment(config: ExperimentConfig, *, dry_run: bool = False, transport=None) -> dict[str, Any]:
    if dry_run:
        return _run_experiment(config, dry_run=True, transport=transport)
    if config.values["token_budget"] is None:
        raise ValueError("Live execution requires an explicit positive token_budget")
    if transport is None and not os.environ.get(config.values["api_key_env"]):
        raise ValueError(f"Missing credential environment variable {config.values['api_key_env']}")
    directory = Path(config.values["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".experiment.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExecutionPaused("Another process owns the experiment output directory") from error
        try:
            return _run_experiment(config, dry_run=False, transport=transport)
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def _run_experiment(config: ExperimentConfig, *, dry_run: bool, transport=None) -> dict[str, Any]:
    """transport is a test-only dependency injection; the CLI never supplies one."""
    task, manifest, context_text = prepare(config)
    values = config.values
    if dry_run:
        return {"status": "dry_run", "method": values["method"], "backend": values["backend"],
                "assay": task.assay, "n": len(task.case.ids), "fingerprint": manifest["fingerprint"],
                "initial_requests": len(manifest["initial_requests"]),
                "prompt_sha256": [digest([row["system"], row["user"]]) for row in manifest["initial_requests"]],
                "workflow_requests": "determined by validated role outputs" if values["method"] == "workflow" else None,
                "memory_mode": values["memory_mode"], "evidence_mode": values["evidence_mode"],
                "unavailable_evidence": manifest["evidence"].get("unavailable_evidence", []),
                "token_budget": values["token_budget"], "api_calls": 0,
                "writes": 0, "output_dir": values["output_dir"]}
    if values["token_budget"] is None:
        raise ValueError("Live execution requires an explicit positive token_budget")
    run_dir = Path(values["output_dir"])
    if run_dir.exists() and any(p.name != ".experiment.lock" for p in run_dir.iterdir()) and not (run_dir / "manifest.json").exists():
        raise ValueError("Output directory already contains unrelated or legacy artifacts")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError("Output directory belongs to a different frozen config/input; choose a new directory")
        result_path = run_dir / "result.json"
        if result_path.exists():
            checksum = run_dir / "result.sha256"
            if not checksum.exists() or checksum.read_text().strip() != file_hash(result_path):
                raise ValueError("Saved result integrity check failed; preserve and inspect artifacts")
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") != manifest["fingerprint"]:
                raise ValueError("Result fingerprint does not match its manifest")
            if saved.get("status") == "ok":
                ranking = saved.get("ranking")
                if not isinstance(ranking, list) or len(ranking) != len(task.case.ids) or set(ranking) != set(task.case.ids):
                    raise ValueError("Saved complete result is not a full permutation")
            if saved.get("status") != "paused":
                return {**saved, "reused_saved_result": True}
    else:
        _save(manifest_path, manifest)
    from .backends import build_generator

    generator = build_generator(values["backend"], model_config=config.model_config(),
                                run_dir=run_dir, fingerprint=manifest["fingerprint"],
                                token_budget=values["token_budget"], transport=transport)
    try:
        if values["method"] == "workflow":
            from .workflow import execute_workflow

            result = execute_workflow(task, generator=generator, config=values,
                                      context_text=context_text, run_dir=run_dir,
                                      fingerprint=manifest["fingerprint"])
        else:
            result = execute_protocol(
                values["method"], case=task.case, generate=generator.generate,
                assay_memory=task.assay_memory, context_text=context_text,
                max_repairs=values["max_repairs"], prompt_version=values["prompt_version"],
                group_size=values["group_size"], anchor_count=values["anchor_count"], seed=values["seed"],
            )
        try:
            result["metrics"] = evaluate_result(Path(values["sample_file"]), task.input_hashes["sample"],
                                               task.case.ids, result)
        except (ValueError, TypeError, KeyError, OSError) as error:
            # Retain paid predictions even if the separate evaluator cannot finish.
            result["prediction_status"] = result["status"]
            result["status"] = "evaluation_error"
            result["metrics"] = {"status": "error", "error_type": type(error).__name__,
                                 "error": str(error)}
    except ExecutionPaused as error:
        result = {"status": "paused", "reason": str(error), "ranking": None, "scores": None,
                  "metrics": {"status": "not_evaluated_incomplete"}}
    finally:
        ledger = generator.summary()
        generator.close()
    result.update({"fingerprint": manifest["fingerprint"], "method": values["method"],
                   "backend": values["backend"], "protocol_id": values["protocol"],
                   "assay": task.assay, "n": len(task.case.ids), "ledger": ledger,
                   "memory_mode": values["memory_mode"], "evidence_mode": values["evidence_mode"]})
    _save(run_dir / "result.json", result)
    (run_dir / "result.sha256").write_text(file_hash(run_dir / "result.json") + "\n")
    return result


def command(path: Path, *, dry_run: bool) -> int:
    try:
        result = run_experiment(ExperimentConfig.load(path), dry_run=dry_run)
        fields = ("status", "method", "backend", "protocol_id", "assay", "n", "fingerprint",
                  "metrics", "ledger", "initial_requests", "memory_mode", "evidence_mode",
                  "api_calls", "writes", "output_dir", "token_budget", "reason")
        print(json.dumps({name: result[name] for name in fields if name in result},
                         ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] in {"ok", "dry_run"} else 2
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)},
                         ensure_ascii=False))
        return 2


def memory_command(args) -> int:
    from .memory import (
        import_legacy_basic_memory,
        initialize_memory,
        inspect_memory,
        write_basic_memory,
    )

    if args.action == "init":
        if args.basic_db is None and args.learning_db is None:
            raise ValueError("Supply --basic-db and/or --learning-db")
        result = initialize_memory(basic_db=args.basic_db, learning_db=args.learning_db)
    elif args.action == "write-basic":
        if args.basic_db is None or args.file is None:
            raise ValueError("write-basic requires --basic-db and --file")
        result = write_basic_memory(args.basic_db, args.file.read_text(encoding="utf-8"))
    elif args.action == "import-basic":
        if args.basic_db is None or args.source_db is None:
            raise ValueError("import-basic requires --basic-db and --source-db")
        result = import_legacy_basic_memory(args.source_db, args.basic_db)
    else:
        result = inspect_memory(basic_db=args.basic_db, learning_db=args.learning_db)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
