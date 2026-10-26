"""Small offline entrypoint, independent of model providers and agent frameworks."""
import argparse
import hashlib
import json
from pathlib import Path

from . import __version__
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(prog="mutabench")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("evaluate", help="Evaluate a CSV against frozen tasks and independent labels")
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--labels", type=Path, required=True)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output file; existing results are never overwritten")
    try:
        result = evaluate(args.tasks, args.labels, args.predictions)
        result["input_sha256"] = {
            name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
            for name in ("tasks", "labels", "predictions")
        }
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(output=str(args.output), summary=result["summary"]), allow_nan=False))


if __name__ == "__main__":
    main()

