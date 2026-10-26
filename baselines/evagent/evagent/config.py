"""Configuration with no secrets stored in source or traceable objects."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RG_LLM_ROOT = Path("/data/yanjie_huang/RG-LLM")
DEFAULT_DATA_ROOT = Path(
    "/data/yanjie_huang/enzyme1_server/rna_benchmark/z_rnagym_70/assays_classified"
)

# Deterministic baseline-performance quantiles among assays with >=250 valid variants.
# This is an engineering pilot, not a paper evaluation subset.
PILOT_ASSAYS = (
    "Ke_2017_mRNA",
    "Kobori_2015_ribozyme_j12",
    "PKN1_HUMAN_Tsuboyama_2023_1URF",
    "Tome_2014_GFP_aptamer",
    "SPTN1_CHICK_Tsuboyama_2023_1TUD",
    "OBSCN_HUMAN_Tsuboyama_2023_1V1C",
)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "gemini-3.8-flash"
    endpoint: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    )
    api_key_env: str = "GEMINI_API_KEY"
    proxy: str | None = "http://127.0.0.1:17892"
    temperature: float = 0.0
    thinking_level: str = "medium"
    include_thoughts: bool = True
    max_tokens: int | None = None
    timeout_s: int = 3600
    retries: int = 3

    @classmethod
    def from_rg_llm(cls, path: Path) -> "ModelConfig":
        raw = json.loads(path.read_text())
        proxy_override = os.environ.get("EVAGENT_PROXY_URL")
        if proxy_override is None:
            proxy: str | None = "http://127.0.0.1:17892"
        elif proxy_override.strip().lower() in {"", "none", "direct"}:
            proxy = None
        else:
            proxy = proxy_override.strip()
        return cls(
            model_id=raw.get("model_id", cls.model_id),
            endpoint=raw.get("endpoint", cls.endpoint),
            api_key_env=raw.get("api_key_env", cls.api_key_env),
            # Default to EVA_a100's listener. EVAGENT_PROXY_URL=direct is an
            # explicit recovery path when the local proxy is unavailable.
            proxy=proxy,
            temperature=float(raw.get("temperature", 0.0)),
            # EVAgent deliberately overrides the RG-LLM high default.
            thinking_level="medium",
            include_thoughts=bool(raw.get("include_thoughts", True)),
            max_tokens=raw.get("max_tokens"),
        )

    def public_dict(self) -> dict[str, Any]:
        """Safe to trace: contains an env-var name, never its value."""
        return asdict(self)


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 1
    grouping_seed: int | None = None
    reasoning_batch_size: int = 24
    local_group_size: int = 32
    anchor_count: int = 8
    repeats: int = 3
    max_concurrency: int = 4
    verification_candidates: int = 16
    schema_retries: int = 1
    l2_regularization: float = 1e-3

    def validate(self) -> None:
        if self.local_group_size < 2:
            raise ValueError("local_group_size must be >=2")
        if not 0 <= self.anchor_count < self.local_group_size:
            raise ValueError("anchor_count must be in [0, local_group_size)")
        if self.repeats < 1:
            raise ValueError("repeats must be >=1")
        if self.reasoning_batch_size < 1 or self.max_concurrency < 1:
            raise ValueError("batch size and concurrency must be positive")


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    rg_llm_root: Path = DEFAULT_RG_LLM_ROOT
    data_root: Path = DEFAULT_DATA_ROOT

    @classmethod
    def defaults(cls, rg_llm_root: Path = DEFAULT_RG_LLM_ROOT) -> "AppConfig":
        model_path = rg_llm_root / "configs" / "gemini-3.8-flash.json"
        model = ModelConfig.from_rg_llm(model_path) if model_path.exists() else ModelConfig()
        return cls(model=model, rg_llm_root=rg_llm_root)

    def fingerprint(self) -> str:
        payload = {
            "model": self.model.public_dict(),
            "pipeline": asdict(self.pipeline),
            "rg_llm_root": str(self.rg_llm_root),
            "data_root": str(self.data_root),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()
