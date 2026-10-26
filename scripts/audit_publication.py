"""Check the code checkout for credentials, oversized files and escaping links.

This is a release check for the curated code tree, not a data-permission audit.
Matches report filenames only; credential contents are never printed.
"""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(rb"(?<![A-Za-z0-9_])(?:sk-[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
SKIP = {".git", "__pycache__", ".pytest_cache", "build", "dist"}


def audit():
    count = total = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in SKIP or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError("External symbolic link: " + str(relative))
        if not path.is_file():
            continue
        if path.name in {"AGENTS.md", "CLAUDE.md", ".env"} or "private_config" in relative.parts:
            raise ValueError("Local/private file in export: " + str(relative))
        content = path.read_bytes()
        if len(content) >= 50 * 1024**2:
            raise ValueError("Large data file in code export: " + str(relative))
        if SECRET.search(content):
            raise ValueError("Credential-like content: " + str(relative))
        if path.suffix == ".py":
            ast.parse(content, filename=str(relative))
        count += 1
        total += len(content)
    return dict(status="PASS", files=count, bytes=total, credential_matches=0)


if __name__ == "__main__":
    print(json.dumps(audit()))
