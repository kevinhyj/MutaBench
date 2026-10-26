"""One bounded gateway-accounted real native tool roundtrip, with no auto restart."""
import datetime
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict

from evagent.integrated.gateway import ExecutionPaused, RequestGateway, digest
from evagent.integrated.research.contracts import save
from evagent.integrated.research.framework_model import LedgerAgentModel

config_path, root = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text())
model_config = config["model_config"]
key = model_config.get("api_key_env", "GEMINI_API_KEY")
credentials = dotenv_values("/home/yanjie/.config/evagent/credentials.env")
if credentials.get(key):
    os.environ[key] = credentials[key]
if (root / "receipt.json").exists():
    raise ValueError("Smoke already attempted; inspect its receipt before any explicit recovery")
calls = []


@tool
def connectivity_probe(nonce: str) -> str:
    """Return the server's fixed response to the supplied probe nonce."""
    calls.append(nonce)
    save(root / "executed_tools.json", calls)
    return "EVAGENT_NATIVE_OK:" + nonce


class ProbeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observed: str


gateway = RequestGateway(model_config=model_config, run_dir=root,
                         fingerprint=digest({"purpose": "framework-native-roundtrip-v4", "model": model_config}),
                         token_budget=100000, budget_mode="limited", reserve_budget=True)
model = LedgerAgentModel(gateway, {"phase": "transport_check", "role": "native_probe"}, maximum=3)
receipt = {"status": "failed"}
try:
    agent = create_agent(model, [connectivity_probe],
                         response_format=ToolStrategy(ProbeReport, handle_errors=False),
                         system_prompt="Call connectivity_probe exactly once with nonce 'framework_v4'. After receiving the tool result, return ProbeReport.observed with exactly that result. Do not use external knowledge.")
    state = agent.invoke({"messages": [{"role": "user", "content": "Perform the native tool roundtrip now."}]},
                         config={"recursion_limit": 10, "max_concurrency": 1})
    final = ProbeReport.model_validate(state.get("structured_response"))
    receipt.update(status="passed" if final.observed == "EVAGENT_NATIVE_OK:framework_v4" and calls == ["framework_v4"] else "failed",
                   report=final.model_dump())
except ExecutionPaused as error:
    receipt.update(status="paused", error=str(error)[:500])
except Exception as error:
    receipt.update(status="failed", error_type=type(error).__name__, error=str(error)[:500])
finally:
    receipt.update(model_calls=model.records, tool_proposals=model.tool_call_records,
                   ledger=gateway.summary(), checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    gateway.close()
    save(root / "receipt.json", receipt)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
raise SystemExit(0 if receipt["status"] == "passed" else 2)
