"""Single campaign launch with existing credentials and persistent failure state."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

from evagent.integrated.research.campaign import command
from evagent.integrated.research.contracts import save

path = Path(sys.argv[1]).resolve()
config = json.loads(path.read_text())
root = (path.parent / config["output_dir"]).resolve()
credentials = dotenv_values("/home/yanjie/.config/evagent/credentials.env")
if credentials.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = credentials["GEMINI_API_KEY"]
if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("Missing configured API credential")
launch = {
    "pid": os.getpid(),
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "config": str(path),
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.executable,
    "restart_policy": "none; unknown responses/cost require resolution",
}
save(root / ("launch-" + str(os.getpid()) + ".json"), launch)
try:
    exit_code = command(path)
except BaseException as error:
    save(
        root / "runner_failure.json",
        {"error_type": type(error).__name__, "reason": str(error)[:1000], "launch": launch},
    )
    raise
raise SystemExit(exit_code)
