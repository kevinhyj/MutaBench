"""Bounded task-background query planning; predictions never see discovery transcripts."""

import json
from pathlib import Path

from ..gateway import digest
from .contracts import BoundedModel, parse_json, save
from .literature import check_query


def plan_queries(spec, gateway, root):
    context = json.loads(Path(spec["context_file"]).read_text())
    # Explicit allowlist. No candidate sequences, labels or original benchmark title.
    request = {
        "entity": spec["literature_query"],
        "family": spec["family"],
        "task_background": context["metadata"].get("ASSAY_DESCRIPTION", ""),
    }
    system = (
        "Plan two precise literature search queries about the named RNA entity and its background mechanism. "
        "Do not search for mutations, variants, fitness, rankings, benchmark outcomes or assay authors. "
        "Preserve the RNA entity name in both queries. Focus on folding, recognition, processing or binding as applicable. "
        'Return only JSON: {"queries":["query one","query two"]}. No claims or predicted effects.'
    )
    # Cache context includes the actual model and immutable query-planning policy.
    identity = digest({"request": request, "model": gateway.model_config, "system": system})
    path = root / "query_plans" / (identity + ".json")
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["sha256"] != digest({k: v for k, v in saved.items() if k != "sha256"}):
            raise ValueError("Query-plan cache changed")
        return saved["queries"]
    model = BoundedModel(
        gateway,
        {"assay": spec["assay"], "role": "literature_discovery", "query_context_sha256": identity},
        2,
    )
    queries = None
    for attempt in range(2):
        text = model.generate(system, json.dumps(request), phase="preparation_queries")
        try:
            result = parse_json(text)
            if (
                set(result) != {"queries"}
                or not isinstance(result["queries"], list)
                or len(result["queries"]) != 2
            ):
                raise ValueError("Expected two queries")
            for q in result["queries"]:
                check_query(q)
            queries = list(dict.fromkeys([spec["literature_query"], *result["queries"]]))
            break
        except (ValueError, TypeError):
            request["format_correction"] = (
                "Use exactly two background queries; never query candidate effects."
            )
    if queries is None:
        raise ValueError("Literature query planner failed; inspect recorded response")
    saved = {
        "queries": queries,
        "request_context": request,
        "model_calls": model.records,
        "schema": "research-query-plan-v1",
    }
    saved["sha256"] = digest(saved)
    save(path, saved)
    return queries
