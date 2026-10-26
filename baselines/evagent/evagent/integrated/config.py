"""Explicit, versioned configuration for the integrated experiment entry point."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METHODS = {"global", "cr", "acr", "workflow"}
PROTOCOLS = {
    "global": "integrated-global-v1",
    "cr": "integrated-cr-v2",
    "acr": "integrated-acr-v1",
    "workflow": "integrated-workflow-v1",
}


@dataclass(frozen=True)
class ExperimentConfig:
    values: dict[str, Any]
    source: Path

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        path = path.resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Experiment config must be a JSON object")
        allowed = {
            "method", "backend", "model", "thinking_level", "temperature", "max_tokens",
            "timeout_s", "api_key_env", "proxy", "endpoint", "sample_file", "context_file",
            "rg_llm_root", "assay_memory_file", "output_dir", "token_budget", "protocol",
            "prompt_version", "max_repairs", "group_size", "anchor_count", "seed",
            "memory_mode", "basic_memory_db", "learning_memory_db", "strategy_file",
            "evidence_mode", "knowledge_root", "repeats", "verification_candidates",
            "evidence_bundle_file", "evidence_sources",
        }
        unknown = raw.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        for name in ("method", "model", "thinking_level", "sample_file", "output_dir"):
            if not isinstance(raw.get(name), str) or not raw[name].strip():
                raise ValueError(f"Explicit non-empty {name} is required")
        method = raw["method"]
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        values = {
            "backend": "direct", "protocol": PROTOCOLS[method], "prompt_version": "v2",
            "max_repairs": 1 if method == "cr" else 0, "memory_mode": "none",
            "evidence_mode": "none", "temperature": 0.0, "group_size": 25,
            "anchor_count": 4, "seed": 1, "repeats": 1, "verification_candidates": 0,
            "timeout_s": 3600, "api_key_env": "GEMINI_API_KEY", "token_budget": None,
            **raw,
        }
        if values["backend"] not in {"direct", "deepagents"}:
            raise ValueError("backend must be direct or deepagents")
        if values["thinking_level"] not in {"minimal", "low", "medium", "high"}:
            raise ValueError("Unknown thinking_level")
        if values["memory_mode"] not in {"none", "basic", "learning", "hybrid"}:
            raise ValueError("Unknown memory_mode")
        if values["evidence_mode"] not in {"none", "skill", "literature", "structure", "curated", "bundle"}:
            raise ValueError("Unknown evidence_mode")
        if values["evidence_mode"] == "bundle":
            if not values.get("evidence_bundle_file") or not values.get("evidence_sources"):
                raise ValueError("Bundle mode requires a file and evidence_sources")
            if method != "cr":
                raise ValueError("Frozen bundle mode currently supports CR only")
        elif values.get("evidence_bundle_file") or values.get("evidence_sources"):
            raise ValueError("Bundle fields require evidence_mode=bundle")
        if values["prompt_version"] not in {"v1", "v2", "v3"}:
            raise ValueError("Unknown prompt_version")
        if values["prompt_version"] == "v3" and method != "cr":
            raise ValueError("Evidence-aware prompt v3 is CR-only")
        if method == "cr" and values["prompt_version"] == "v1":
            if "protocol" not in raw:
                values["protocol"] = "integrated-cr-v1"
            if "max_repairs" not in raw:
                values["max_repairs"] = 0
        expected_protocol = "integrated-cr-v1" if method == "cr" and values["prompt_version"] == "v1" else PROTOCOLS[method]
        if values["protocol"] != expected_protocol:
            raise ValueError(f"protocol must be {expected_protocol} for the selected method/version")
        for name, minimum, maximum in (("temperature", 0, 2), ("timeout_s", 0.001, None)):
            number = values[name]
            if type(number) not in (int, float) or not math.isfinite(number) or number < minimum or (
                maximum is not None and number > maximum
            ):
                raise ValueError(f"Invalid numeric {name}")
        maximum_tokens = values.get("max_tokens")
        if maximum_tokens is not None and (type(maximum_tokens) is not int or maximum_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        for name, minimum in (("max_repairs", 0), ("group_size", 2), ("anchor_count", 0),
                              ("repeats", 1), ("verification_candidates", 0), ("seed", 0)):
            if type(values[name]) is not int or values[name] < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if values["anchor_count"] >= values["group_size"]:
            raise ValueError("anchor_count must be smaller than group_size")
        if method == "acr" and values["anchor_count"] < 1:
            raise ValueError("ACR requires at least one shared anchor")
        budget = values["token_budget"]
        if budget is not None and (type(budget) is not int or budget <= 0):
            raise ValueError("token_budget must be a positive integer or null for dry-run")
        if values.get("strategy_file") and values["memory_mode"] not in {"learning", "hybrid"}:
            raise ValueError("strategy_file requires learning or hybrid memory")
        for name in ("sample_file", "context_file", "rg_llm_root", "assay_memory_file",
                     "output_dir", "basic_memory_db", "learning_memory_db", "strategy_file",
                     "knowledge_root", "evidence_bundle_file"):
            if values.get(name) is not None:
                candidate = Path(values[name]).expanduser()
                values[name] = str((path.parent / candidate).resolve())
        from .gateway import public_model_config
        public_model_config(cls(values, path).model_config())
        return cls(values, path)

    def model_config(self) -> dict[str, Any]:
        names = ("model", "thinking_level", "temperature", "max_tokens", "timeout_s",
                 "api_key_env", "proxy", "endpoint")
        return {name: self.values[name] for name in names if name in self.values}
