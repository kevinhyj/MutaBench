"""Adapt the existing LangGraph roles without legacy output fallback or label access."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from evagent.config import PipelineConfig
from evagent.graph import EVAgentWorkflow
from evagent.model import StructuredResponse
from evagent.schemas import AssayMemory, ModelCallRecord, VariantEvidence

from .ranking import final_json_object
from .self_check import SelfCheckPolicy, ValidationIssue, ValidationReport, run_self_check
from .tasks import ModelTask


class InvalidWorkflowOutput(RuntimeError):
    """Not a legacy ModelError: the graph must not replace this with neutral predictions."""

    def __init__(self, message: str, status: str = "invalid_output"):
        super().__init__(message)
        self.status = status


def _json_after(text: str, marker: str) -> Any:
    return json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())[0]


class RoleValidator:
    def __init__(self, run_name: str, user: str, assay: str):
        self.run_name, self.assay = run_name, assay
        self.ids: list[str] = []
        if run_name.startswith("variant_reasoner"):
            self.ids = [row["variant_id"] for row in
                        _json_after(user, "Candidates with local WT/mutant sequence context:")]
        elif run_name.startswith("local_ranker"):
            self.ids = [row["variant_id"] for row in _json_after(user, "Variant evidence:")]
        elif run_name == "uncertainty_verifier":
            self.ids = _json_after(user, "Current uncertain region, already listed approximately best-to-worst:")

    def validate(self, text: str) -> ValidationReport:
        try:
            raw = final_json_object(text)
            if not isinstance(raw, dict):
                raise ValueError("Role output must be a JSON object")
            if self.run_name == "assay_analyst":
                value = AssayMemory.model_validate({"assay_id": self.assay, **raw})
                if not value.assay_summary.strip() or not value.property_direction.strip():
                    raise ValueError("Assay summary and property direction must be non-empty")
            elif self.run_name.startswith("variant_reasoner"):
                rows = raw.get("variants")
                if not isinstance(rows, list):
                    raise ValueError("Role must return variants")
                expected_fields = {"variant_id", "mutation", "direction", "predicted_score",
                                   "mechanisms", "evidence", "confidence"}
                for row in rows:
                    if not isinstance(row, dict) or not expected_fields <= row.keys():
                        raise ValueError("Incomplete variant evidence")
                    if type(row["predicted_score"]) not in (int, float):
                        raise ValueError("Scores must be JSON numbers")
                    VariantEvidence.model_validate(row)
                ids = [row["variant_id"] for row in rows]
                if len(ids) != len(self.ids) or set(ids) != set(self.ids):
                    raise ValueError("Variant evidence must cover exact IDs once")
                if any(row.get("fallback", False) for row in rows):
                    raise ValueError("Fallback evidence is invalid in the integrated protocol")
            else:
                ids = raw.get("ranking", [])
                if not isinstance(ids, list) or len(ids) != len(self.ids) or set(ids) != set(self.ids):
                    raise ValueError("Role ranking must be an exact complete permutation")
            return ValidationReport(valid=True, value=raw)
        except (ValueError, TypeError, KeyError) as error:
            return ValidationReport(valid=False, value=None,
                                    issues=(ValidationIssue("invalid_role_output", str(error)),))


class StrictRoleAdapter:
    def __init__(self, generator: Any, task: ModelTask, context_text: str, max_repairs: int):
        self.generator, self.task = generator, task
        self.context_text, self.max_repairs = context_text, max_repairs
        self.attempts: list[dict[str, Any]] = []

    def generate_json(self, *, system: str, user: str, run_name: str,
                      metadata: dict[str, Any] | None = None) -> StructuredResponse:
        validator = RoleValidator(run_name, user, self.task.assay)
        if self.context_text:
            system += "\n\nRead-only context:\n" + self.context_text

        def generate(prompt: str, attempt_index: int) -> dict[str, Any]:
            value = self.generator.generate(system=system, user=prompt, run_name=run_name,
                                            metadata={**(metadata or {}), "repair_index": attempt_index})
            return value

        checked = run_self_check(original_prompt=user, generate=generate, validator=validator,
                                 policy=SelfCheckPolicy(max_repairs=self.max_repairs))
        self.attempts.extend({**value, "role": run_name} for value in checked["attempts"])
        if checked["status"] != "ok":
            raise InvalidWorkflowOutput(f"{run_name}: {checked['status']}", checked["status"])
        calls = []
        for value in checked["attempts"]:
            usage = value.get("usage_summary", {})
            calls.append(ModelCallRecord(
                run_name=run_name, answer=value.get("final_text", ""),
                thought_summary=value.get("thought_summary", ""),
                usage={"totalTokenCount": usage.get("total_tokens", 0),
                       "promptTokenCount": usage.get("input_tokens", 0),
                       "candidatesTokenCount": usage.get("output_tokens", 0)},
                elapsed_s=value.get("elapsed_s", 0.0),
                finish_reason=value.get("finish_reason", ""),
                request_sha256=value.get("request_sha256", ""),
                cache_hit=value.get("cache_hit", False),
            ))
        return StructuredResponse(checked["value"], calls)


def execute_workflow(task: ModelTask, *, generator: Any, config: dict[str, Any],
                     context_text: str, run_dir: Path, fingerprint: str) -> dict[str, Any]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    adapter = StrictRoleAdapter(generator, task, context_text, config["max_repairs"])
    if task.assay_memory:
        adapter.context_text += "\n\nFrozen assay context:\n" + json.dumps(task.assay_memory, sort_keys=True)
    workflow = EVAgentWorkflow(adapter, PipelineConfig(
        seed=config["seed"], grouping_seed=config["seed"],
        reasoning_batch_size=config["group_size"], local_group_size=config["group_size"],
        anchor_count=config["anchor_count"], repeats=config["repeats"],
        verification_candidates=config["verification_candidates"], max_concurrency=1,
        schema_retries=0,
    ))
    initial = {"assay_id": task.assay, "metadata": task.metadata,
               "wt_sequence": task.wt_sequence, "variants": task.variants}
    # New protocol namespace, containing only label-free state.
    connection = sqlite3.connect(run_dir / "workflow-checkpoints.sqlite", check_same_thread=False)
    try:
        graph = workflow.compile(checkpointer=SqliteSaver(connection))
        state = graph.invoke(initial, config={"configurable": {"thread_id": fingerprint}})
        if any(row.get("fallback") for row in state.get("variant_evidence", [])):
            raise InvalidWorkflowOutput("Incomplete workflow evidence")
        if any(row.get("record_type") == "fallback" for row in state.get("reasoning_records", [])):
            raise InvalidWorkflowOutput("Workflow used legacy fallback")
        ranking = [row["variant_id"] for row in state["predictions"]]
        return {"status": "ok", "ranking": ranking, "scores": state["aggregate_scores"],
                "attempts": adapter.attempts, "workflow_state": state}
    except InvalidWorkflowOutput as error:
        return {"status": error.status, "ranking": None, "scores": None,
                "error": str(error), "attempts": adapter.attempts}
    finally:
        connection.close()
