"""Make a bounded, hash-verified rsync list; never opens a provider or live DB.

Server: python export_manifest.py --experiment /path/to/fewshot_v1
Client: rsync -a --files-from=files.txt host:/path/to/fewshot_v1/ /local/fewshot_v1/
Client: python export_manifest.py --experiment /local/fewshot_v1 --verify manifest.json

The manifest and files.txt themselves must be copied separately. Run-state and
ledger evidence come exclusively from the newest *completed* progress snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REQUIRED = (
    "process.json", "authorization.json", "model_config.json",
    "llm_full/plan.json", "llm_full/execution_config.json",
)
OPTIONAL = (
    "source_audit.json", "transport_audit.json", "provider_preflight.json",
    "implementation_verification.json", "implementation_verification_preconcurrency.json",
    "pilot_selection.json", "preparation_index.json", "transfer_manifest.json",
    "launch_full.py", "snapshot_progress.py", "finalize_after_run.py",
    "paired_analysis.py", "export_manifest.py",
)
SNAPSHOT_NAME = re.compile(r"\d{8}T\d{6}\.\d{6}Z")
PREDICTION_JSON = re.compile(r"(?:request(?:_\d+)?|context(?:_\d+)?|result)\.json")


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"unsafe relative path: {value!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("control character in relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def source_path(root: Path, relative: str) -> Path:
    path = root / safe_relative(relative)
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinks are not export sources: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"source escapes experiment root: {relative}")
    return path


def fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def hash_stable_file(root: Path, relative: str) -> tuple[dict, tuple[int, ...]]:
    path = source_path(root, relative)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {relative}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(path.stat()):
        raise RuntimeError(f"source changed while hashing: {relative}; create a fresh export")
    return {"path": relative, "sha256": digest.hexdigest(), "size": before.st_size}, fingerprint(after)


def newest_snapshot(root: Path, requested: str | None) -> Path:
    base = source_path(root, "progress_snapshots")
    if requested is not None:
        if not SNAPSHOT_NAME.fullmatch(requested):
            raise ValueError("--snapshot must be a timestamp directory name")
        candidates = [base / requested]
    else:
        candidates = sorted((p for p in base.iterdir() if SNAPSHOT_NAME.fullmatch(p.name)), reverse=True)
    for candidate in candidates:
        relative = candidate.relative_to(root).as_posix()
        source_path(root, relative)
        if not candidate.is_dir():
            continue
        progress = source_path(root, relative + "/progress.json")
        ledger = source_path(root, relative + "/request_ledger.sqlite")
        # snapshot_progress.py publishes progress.json after its consistent DB
        # backup and analysis files finish. Ignore a newer unfinished snapshot.
        if progress.is_file() and ledger.is_file():
            payload = json.loads(progress.read_text())
            if not payload.get("snapshot_utc"):
                raise ValueError(f"snapshot lacks completion metadata: {relative}")
            return candidate
    raise ValueError("no completed progress snapshot with progress.json and ledger")


def select_files(root: Path, snapshot: str | None) -> tuple[list[str], dict]:
    root = root.resolve(strict=True)
    selected: set[str] = set()
    missing_optional = []
    for relative in REQUIRED:
        if not source_path(root, relative).is_file():
            raise FileNotFoundError(f"required export source missing: {relative}")
        selected.add(relative)
    for relative in OPTIONAL:
        if source_path(root, relative).is_file():
            selected.add(relative)
        else:
            missing_optional.append(relative)
    predictions = source_path(root, "llm_full/predictions")
    completed = 0
    for directory in sorted(predictions.iterdir()):
        source_path(root, directory.relative_to(root).as_posix())
        if not directory.is_dir():
            continue
        result = source_path(root, (directory / "result.json").relative_to(root).as_posix())
        if not result.is_file():
            continue
        # Parse the atomic completion marker so a truncated file cannot qualify.
        if not isinstance(json.loads(result.read_text()), dict):
            raise TypeError(f"invalid prediction result: {result}")
        for path in directory.iterdir():
            if PREDICTION_JSON.fullmatch(path.name):
                relative = path.relative_to(root).as_posix()
                if source_path(root, relative).is_file():
                    selected.add(relative)
        completed += 1
    tokens = source_path(root, "llm_full/token_checks")
    if tokens.exists():
        for path in tokens.glob("*.json"):
            relative = path.relative_to(root).as_posix()
            if source_path(root, relative).is_file():
                selected.add(relative)
    folder = newest_snapshot(root, snapshot)
    for path in folder.rglob("*"):
        relative = path.relative_to(root).as_posix()
        source_path(root, relative)
        if path.name.endswith(("-wal", "-shm", "-journal", ".tmp", ".part")):
            continue
        if path.is_file():
            selected.add(relative)
    return sorted(selected), {
        "selected_snapshot": folder.relative_to(root).as_posix(),
        "completed_prediction_directories": completed,
        "missing_optional": missing_optional,
    }


def build_export(root: Path, snapshot: str | None = None) -> dict:
    root = root.resolve(strict=True)
    relative_files, selection = select_files(root, snapshot)
    files, signatures = [], {}
    for relative in relative_files:
        item, signature = hash_stable_file(root, relative)
        files.append(item)
        signatures[relative] = signature
    # Reject an overwritten file even if its first hashing pass had completed.
    for relative, signature in signatures.items():
        if fingerprint(source_path(root, relative).stat()) != signature:
            raise RuntimeError(f"source changed while building export: {relative}")
    now = datetime.now(timezone.utc)
    output = source_path(root, "exports") / now.strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "fewshot-completed-export-v1", "generated_utc": now.isoformat(),
        "source_root": str(root), **selection, "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files), "files": files,
        "model_calls": 0,
        "consistency": "Each selected file was stable while hashed and rechecked before manifest publication. Verify all transferred hashes at destination. Completed predictions may be newer than the selected ledger snapshot; this is not a globally atomic live-run snapshot.",
        "excluded": ["live provider DB/WAL/SHM", "incomplete prediction directories",
                     "launch logs", "live status/heartbeat/latest_progress", "older progress snapshots",
                     "temporary SQLite journals and temporary files"],
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output / "files.txt").open("x") as handle:
        handle.write("\n".join(relative_files) + "\n")
    return {"manifest": str(output / "manifest.json"), "files_from": str(output / "files.txt"),
            "file_count": len(files), "total_bytes": manifest["total_bytes"], **selection}


def verify_export(root: Path, manifest_path: Path) -> dict:
    root = root.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "fewshot-completed-export-v1":
        raise ValueError("unexpected manifest schema")
    seen = set()
    for expected in manifest["files"]:
        relative = safe_relative(expected["path"])
        if relative in seen:
            raise ValueError(f"duplicate manifest path: {relative}")
        seen.add(relative)
        actual, _ = hash_stable_file(root, relative)
        if actual != expected:
            raise ValueError(f"destination hash/size mismatch: {relative}")
    if len(seen) != manifest["file_count"]:
        raise ValueError("manifest file count mismatch")
    return {"status": "verified", "files": len(seen), "model_calls": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--snapshot", help="Optional exact completed snapshot timestamp")
    parser.add_argument("--verify", type=Path, help="Verify a transferred manifest against this experiment root")
    args = parser.parse_args()
    root = args.experiment.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("experiment root must be a directory")
    result = verify_export(root, args.verify) if args.verify else build_export(root, args.snapshot)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
