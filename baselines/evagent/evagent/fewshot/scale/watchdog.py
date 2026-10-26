"""Server cron entry point; process identity plus two independent exclusive locks."""
from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from evagent.fewshot.protocol import digest
from evagent.api_timeouts import generation_timeout_seconds
from evagent.fewshot.runner import load, save


def identity(pid):
    try:
        folder = Path("/proc") / str(pid)
        stat = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": stat[19], "state": stat[0],
                "command": (folder / "cmdline").read_bytes().split(b"\0")[:-1]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record, root):
    current = identity(record["pid"])
    expected = [str(Path(sys.executable).absolute()), "-I", "-m", "evagent.fewshot.scale.campaign", "run", "--root", str(root)]
    return bool(current and current["state"] != "Z" and current["start_ticks"] == record["start_ticks"]
                and [x.decode() for x in current["command"]] == expected)


def cron(root, install):
    scheduler_path = root / "scheduler.json"
    if scheduler_path.exists() and load(scheduler_path).get("backend") == "systemd_user_timer":
        scheduler = load(scheduler_path)
        subprocess.run(["systemctl", "--user", "enable" if install else "disable", "--now",
                        scheduler["timer"]], check=True)
        save(root / "cron_status.json", {"enabled": install, "backend": "systemd_user_timer",
             "cron_unavailable": "user crontab denied by server spool permissions",
             "interval_seconds": 60, "time": time.time()})
        return
    marker = "# EVAGENT_SCALE_V3_" + digest(str(root))[:16]
    old = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if old.returncode not in (0, 1):
        raise RuntimeError("Cannot read user crontab")
    lines = [line for line in old.stdout.splitlines() if not line.endswith(marker)]
    if install:
        cmd = [str(Path(sys.executable).absolute()), "-I", "-m", "evagent.fewshot.scale.watchdog", "check", "--root", str(root)]
        lines.append("* * * * * " + shlex.join(cmd) + " >> " + shlex.quote(str(root / "watchdog.log")) + " 2>&1 " + marker)
    payload = "\n".join(lines) + ("\n" if lines else "")
    subprocess.run(["crontab", "-"], input=payload, text=True, check=True)
    save(root / "cron_status.json", {"enabled": install, "marker": marker, "time": time.time(),
                                    "interval_seconds": 60})


def install(root):
    try:
        cron(root, True)
        return
    except (subprocess.CalledProcessError, RuntimeError):
        # Some hosts run cron but deny users traversal of /var/spool/cron.
        # A lingering user manager provides the same server-owned 60s schedule.
        user = subprocess.check_output(["id", "-un"], text=True).strip()
        subprocess.run(["loginctl", "--no-ask-password", "enable-linger", user], check=True)
        linger = subprocess.check_output(["loginctl", "show-user", user, "-p", "Linger", "--value"], text=True).strip()
        if linger != "yes":
            raise RuntimeError("Persistent server scheduler unavailable")
    name = "evagent-scale-v3-" + digest(str(root))[:12]
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    def quote(value):
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    def command(module, action):
        return " ".join(quote(x) for x in [Path(sys.executable).absolute(), "-I", "-m", module, action, "--root", root])
    worker = name + "-worker.service"
    monitor = name + "-watchdog.service"
    timer = name + ".timer"
    (units / worker).write_text("[Unit]\nDescription=EVAgent bounded few-shot experiment\n\n[Service]\n"
        "Type=exec\nRestart=no\nKillMode=control-group\nTimeoutStopSec=10\nUMask=0077\n"
        + "ExecStart=" + command("evagent.fewshot.scale.campaign", "run") + "\n"
        + "StandardOutput=append:" + str(root / "worker.log") + "\n"
        + "StandardError=append:" + str(root / "worker.log") + "\n")
    (units / monitor).write_text("[Unit]\nDescription=EVAgent identity-checked watchdog\n\n[Service]\n"
        "Type=oneshot\nTimeoutStartSec=50\nUMask=0077\n"
        + "ExecStart=" + command("evagent.fewshot.scale.watchdog", "check") + "\n")
    (units / timer).write_text("[Unit]\nDescription=EVAgent every-minute recovery\n\n[Timer]\n"
        "OnCalendar=*-*-* *:*:00\nAccuracySec=1s\nPersistent=true\nUnit=" + monitor + "\n\n[Install]\nWantedBy=timers.target\n")
    save(root / "scheduler.json", {"backend": "systemd_user_timer", "linger": True,
         "timer": timer, "worker": worker, "monitor": monitor,
         "fallback_reason": "cron spool permissions deny user crontab", "time": time.time()})
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    cron(root, True)


def timeout_attempts(root, now=None):
    path = root / "campaign.sqlite"
    if not path.exists():
        return []
    model_path = root / "model_config.json"
    configured = load(model_path).get("timeout_s") if model_path.exists() else None
    deadline_s = generation_timeout_seconds(configured) + 60
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return [r[0] for r in db.execute("SELECT id FROM attempts WHERE status='inflight' AND started<?",
                                        ((now or time.time()) - deadline_s,))]


def check(root):
    lease = (root / "watchdog.lock").open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return
    try:
        record_path = root / "process.json"
        record = load(record_path) if record_path.exists() else None
        if record and alive(record, root):
            overdue = timeout_attempts(root)
            if not overdue:
                return
            # The stream timeout can be per-read; enforce an elapsed attempt
            # bound, allowing 60 seconds for cleanup after the configured 900s.
            scheduler = load(root / "scheduler.json") if (root / "scheduler.json").exists() else {}
            if scheduler.get("backend") == "systemd_user_timer":
                subprocess.run(["systemctl", "--user", "stop", scheduler["worker"]], check=True)
            else:
                os.killpg(record["pid"], signal.SIGTERM)
            deadline = time.time() + 10
            while alive(record, root) and time.time() < deadline:
                time.sleep(.2)
            if alive(record, root):
                os.killpg(record["pid"], signal.SIGKILL)
                deadline = time.time() + 5
                while alive(record, root) and time.time() < deadline:
                    time.sleep(.1)
            save(root / "recovery" / f"timeout-{int(time.time())}.json",
                 {"old_process": record, "overdue_attempt_ids": overdue, "time": time.time()})
            if alive(record, root):
                return
        if (root / "DONE.json").exists() or (root / "STOP.json").exists():
            cron(root, False)
            return
        if (root / "PAUSE").exists():
            return
        # Even a missing/stale PID record cannot bypass the worker's lifetime lock.
        with (root / "worker.lock").open("a+") as worker_lease:
            try:
                fcntl.flock(worker_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        if not (root / "FROZEN.json").exists():
            return
        scheduler = load(root / "scheduler.json") if (root / "scheduler.json").exists() else {}
        if scheduler.get("backend") == "systemd_user_timer":
            subprocess.run(["systemctl", "--user", "start", "--no-block", scheduler["worker"]], check=True)
            return
        cmd = [str(Path(sys.executable).absolute()), "-I", "-m", "evagent.fewshot.scale.campaign", "run", "--root", str(root)]
        with (root / "worker.log").open("ab") as log:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, close_fds=True)
        current = identity(child.pid)
        if current:
            save(record_path, {"pid": child.pid, "start_ticks": current["start_ticks"],
                               "command": cmd, "started": time.time()})
    finally:
        lease.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("install", "check", "disable"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "install":
        install(root)
    elif args.command == "disable":
        cron(root, False)
    else:
        check(root)
