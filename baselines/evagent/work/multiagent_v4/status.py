"""Compact read-only process, checkpoint, and provider reconciliation for a campaign."""
import datetime
import json
import sqlite3
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text())
root = (config_path.parent / config["output_dir"]).resolve()
out = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "config": str(config_path)}
for name in ("progress", "calibration", "runner_failure"):
    path = root / (name + ".json")
    if path.exists():
        value = json.loads(path.read_text())
        out[name] = {k: v for k, v in value.items() if k not in {"ledger", "units"}}
        if "ledger" in value:
            out[name]["ledger"] = {k: value["ledger"][k] for k in (
                "current_run_reported_tokens", "prior_reported_tokens", "unknown_usage_attempts", "paused", "pause_reason")}
launches = []
for path in root.glob("launch-*.json"):
    launch = json.loads(path.read_text())
    command = Path("/proc") / str(launch["pid"]) / "cmdline"
    launch["alive"] = command.exists() and str(config_path).encode() in command.read_bytes()
    launches.append(launch)
out["launches"] = launches
out["pause_marker"] = (root / "PAUSE").exists()
ledger = root / "provider/request_ledger.sqlite"
if ledger.exists():
    with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as db:
        out["current_attempts_by_status"] = [dict(zip(("status", "attempts", "reported_tokens"), row))
                                             for row in db.execute("SELECT status,count(*),sum(tokens) FROM attempts GROUP BY status")]
        row = db.execute("SELECT id,status,request_json,response_json,started,elapsed_s FROM attempts ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            out["latest_attempt"] = {"id": row[0], "status": row[1], "metadata": json.loads(row[2])["metadata"],
                                     "started": row[4], "elapsed_s": row[5],
                                     "response_metadata": json.loads(row[3] or "{}").get("response_metadata")}
units = [json.loads(p.read_text()) for p in (root / "units").glob("*.json")]
out["settled_unit_count"] = len(units)
out["invalid_units"] = [{k: u.get(k) for k in ("assay", "group", "condition", "status", "reason")}
                        for u in units if u.get("status") != "ok"]
print(json.dumps(out, ensure_ascii=False, indent=2))
