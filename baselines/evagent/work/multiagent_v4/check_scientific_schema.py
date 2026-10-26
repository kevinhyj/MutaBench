"""One accounted schema-only probe; no assay, candidates, or scientific prediction."""
import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.messages import HumanMessage

from evagent.integrated.gateway import ExecutionPaused, RequestGateway, digest
from evagent.integrated.research.contracts import save
from evagent.integrated.research.framework_model import LedgerAgentModel
from evagent.integrated.research.framework_runner import AnalystReport

config_path, schema_ledger, root = map(Path, sys.argv[1:])
if (root / "receipt.json").exists():
    raise ValueError("Probe already attempted; inspect its receipt")
config = json.loads(config_path.read_text())["model_config"]
with sqlite3.connect(schema_ledger.as_uri() + "?mode=ro", uri=True) as db:
    request = json.loads(db.execute("SELECT request_json FROM attempts WHERE id=2").fetchone()[0])
schemas = request["tool_declarations"]
assert {s["name"] for s in schemas} == {"rna_secondary", "rna_contacts", "rna_annotations", "literature_search", "AnalystReport"}
key = config.get("api_key_env", "GEMINI_API_KEY")
credentials = dotenv_values("/home/yanjie/.config/evagent/credentials.env")
if credentials.get(key):
    os.environ[key] = credentials[key]
gateway = RequestGateway(model_config=config, run_dir=root,
                         fingerprint=digest({"purpose": "scientific_schema_compatibility_only", "model": config, "schema": schemas}),
                         token_budget=100000, budget_mode="limited", reserve_budget=True)
model = LedgerAgentModel(gateway, {"role": "schema_probe", "phase": "transport_check"}, maximum=1)
receipt = {"status": "failed", "purpose": "schema only; no scientific inputs"}
try:
    bound = model.bind_tools([{"type": "function", "function": s} for s in schemas], tool_choice="required")
    response = bound.invoke([HumanMessage(content="This is an API schema compatibility test with no RNA task. Call AnalystReport exactly once with claims=[], assessments=[], pairwise_preferences=[], limitations=['Schema connectivity probe; no scientific conclusion'], abstain=true. Do not call any other function.")])
    assert len(response.tool_calls) == 1 and response.tool_calls[0]["name"] == "AnalystReport"
    final = AnalystReport.model_validate(response.tool_calls[0]["args"])
    assert not final.claims and not final.assessments and not final.pairwise_preferences and final.abstain
    receipt.update(status="passed", report=final.model_dump())
except ExecutionPaused as error:
    receipt.update(status="paused", error=str(error)[:500])
except Exception as error:
    receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:500])
finally:
    receipt.update(model_calls=model.records, tool_proposals=model.tool_call_records,
                   ledger=gateway.summary(), checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    diagnostics = []
    for (raw,) in gateway.db.execute("SELECT response_json FROM attempts"):
        diagnostics.append(json.loads(raw or "{}").get("response_metadata", {}))
    receipt["response_metadata"] = diagnostics
    gateway.close()
    save(root / "receipt.json", receipt)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(0 if receipt["status"] == "passed" else 2)
