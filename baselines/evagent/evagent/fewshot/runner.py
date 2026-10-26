"""Frozen request planning and resumable fixed-CR execution.

Prediction reads episode.json and support.json only. Evaluation is a separate command.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from evagent.integrated.chunked_scoring import ScoringResponseValidator
from evagent.integrated.gateway import RequestGateway, public_model_config
from evagent.integrated.research.concurrent_gateway import ConcurrentGateway
from evagent.integrated.self_check import SelfCheckPolicy, default_repair_prompt, run_self_check
from evagent.runtime import ExecutionPaused

from . import PROTOCOL
from .protocol import CONDITIONS, all_requests, digest, resolve_conditions


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(root):
    paths = sorted(Path(root).rglob("episode.json"))
    if not paths:
        raise ValueError("No frozen episodes found")
    seen = set()
    records = []
    for path in paths:
        episode, support = load(path), load(path.with_name("support.json"))
        if episode["episode_id"] in seen:
            raise ValueError("Duplicate episode identity")
        seen.add(episode["episode_id"])
        records.append((path.parent, episode, support))
    # Interleave assays within each seed/split so an early budget pause does not
    # cover every RNA episode before reaching the first protein episode.
    yield from sorted(records, key=lambda record: (
        record[1].get("seed", 0),
        {"random": 0, "low_order_to_high_order": 1}.get(record[1].get("split"), 2),
        record[1].get("assay_id", ""), record[1]["episode_id"]))


def source_hashes():
    from evagent.integrated import chunked_scoring, gateway, ranking, self_check
    from evagent.integrated.research import concurrent_gateway
    paths = list(Path(__file__).parent.glob("*.py")) + [
        Path(m.__file__) for m in (chunked_scoring, gateway, ranking, self_check, concurrent_gateway)]
    package = Path(__file__).parents[1]
    return {str(p.relative_to(package)): sha(p) for p in sorted(paths)}


def make_plan(episodes_root, model_config, conditions=None):
    selected = resolve_conditions(conditions)
    model = public_model_config(model_config)
    if not model["max_output_tokens"]:
        raise ValueError("Explicit max_output_tokens is required for bounded admission")
    records, hashes = [], {}
    for folder, episode, support in episodes(episodes_root):
        hashes[episode["episode_id"]] = {n: sha(folder / n) for n in ("episode.json", "support.json")}
        for request in all_requests(episode, support, conditions):
            records.append({k: request[k] for k in
                            ("episode_id", "condition", "chunk_index", "expected_ids", "support_ids", "request_hash")})
            records[-1]["utf8_bytes"] = len((request["system"] + "\n\n" + request["user"]).encode())
    plan = {"schema": "fewshot-run-v1", "protocol": PROTOCOL, "model_config": model,
            "source_hashes": source_hashes(), "episode_input_hashes": hashes,
            "query_chunk_size": 25, "max_repairs": 1, "memory_mode": "none",
            "requests": records, "planned_initial_requests": len(records),
            "maximum_generation_requests": 2 * len(records),
            "cost_estimate": {"initial_prompt_utf8_bytes": sum(r["utf8_bytes"] for r in records),
                              "basis": "UTF-8 bytes are a conservative planning proxy, not provider token counts or charges",
                              "maximum_output_tokens_per_request": model["max_output_tokens"],
                              "currency_cost": None},
            "historical_baseline_reuse": False,
            "conditions": [c.__dict__ for c in selected]}
    plan["fingerprint"] = digest(plan)
    return plan


def freeze_plan(episodes_root, run_dir, model_config, conditions=None):
    plan = make_plan(episodes_root, model_config, conditions)
    path = Path(run_dir) / "plan.json"
    if path.exists() and load(path) != plan:
        raise ValueError("Frozen few-shot inputs, code or model changed; use a new run directory")
    if not path.exists():
        save(path, plan)
    return plan


class ContextTooLong(ValueError):
    pass


class GoogleContextCheck:
    """Provider model limits and token counting; created only for explicit live execution."""
    def __init__(self, model_config, cache_dir):
        import urllib.request
        self.config = public_model_config(model_config)
        self.cache_dir = Path(cache_dir)
        base = (self.config.get("endpoint") or "https://generativelanguage.googleapis.com").rstrip("/")
        if not base.endswith("/v1beta"):
            base += "/v1beta"
        self.model_path = "models/" + self.config["model"].removeprefix("models/")
        self.endpoint = base + "/" + self.model_path
        proxy = self.config.get("proxy")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler(
            {"http": proxy, "https": proxy} if proxy else {}))
        info = self._request("")
        if not info.get("inputTokenLimit") or not info.get("outputTokenLimit"):
            raise ValueError("Provider model context limits unavailable")
        self.input_limit = int(info["inputTokenLimit"])
        if self.config["max_output_tokens"] > info["outputTokenLimit"]:
            raise ValueError("Configured output limit exceeds provider model limit")
        save(self.cache_dir / "model_limits.json", {
            "model": self.config["model"], "input_token_limit": self.input_limit,
            "output_token_limit": info["outputTokenLimit"], "basis": "provider models.get"})

    def _request(self, suffix, payload=None):
        import os
        import urllib.request
        credential = os.environ.get(self.config["api_key_env"])
        if not credential:
            raise ValueError("Provider credential environment variable is missing")
        request = urllib.request.Request(
            self.endpoint + suffix, data=None if payload is None else json.dumps(payload).encode(),
            headers={"x-goog-api-key": credential, "Content-Type": "application/json"})
        with self.opener.open(request, timeout=60) as response:
            return json.load(response)

    def __call__(self, system, user):
        key = digest([self.config, system, user])
        path = self.cache_dir / (key + ".json")
        if path.exists():
            result = load(path)
        else:
            # Developer API SDK 2.22 rejects CountTokensConfig.system_instruction.
            # This REST shape is also used by the existing fullseq.Service.count.
            counted = self._request(":countTokens", {"generateContentRequest": {
                "model": self.model_path,
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}]}})
            count = counted.get("totalTokens")
            if type(count) is not int or count < 0:
                raise ValueError("Provider token count unavailable")
            result = {"input_tokens": count, "limit": self.input_limit,
                      "request_hash": key, "basis": "provider countTokens"}
            save(path, result)
        if result["input_tokens"] > self.input_limit:
            raise ContextTooLong("context_limit_exceeded; no examples or sequences truncated")
        return result

    def close(self):
        pass


def repair_prompt(original, previous, validation, attempt):
    return default_repair_prompt(original, previous, validation, attempt).replace(
        "Do not inspect or request labels, answer keys, benchmark scores, previous tasks, or result files.",
        "Use only measured examples already supplied in the original task; never inspect hidden results or other tasks.")


def execute(episodes_root, run_dir, model_config, *, token_budget, live=False,
            transport=None, context_check=None, max_units=None, workers=64, conditions=None):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("workers must be an integer between 1 and 64")
    if max_units is not None and (type(max_units) is not int or max_units < 0):
        raise ValueError("max_units must be a nonnegative integer")
    if not live and transport is None:
        raise ValueError("Generation requires --live and an explicit new experiment token budget")
    if transport is None and context_check is not None:
        raise ValueError("Live execution requires provider context counting")
    root = Path(run_dir)
    plan = freeze_plan(episodes_root, root, model_config, conditions)
    mode = "live" if transport is None else "offline_test"
    execution_config = {"mode": mode, "workers": workers, "context_preflight":
                        "provider_count_tokens" if mode == "live" else "offline_test_injection"}
    execution_path = root / "execution_config.json"
    if execution_path.exists() and load(execution_path) != execution_config:
        raise ValueError("Execution mode/workers mismatch; use the frozen execution settings")
    save(execution_path, execution_config)
    checker = context_check
    # Injectable transport/checker are for offline tests, never a mock scientific result.
    if transport is not None and checker is None:
        raise ValueError("Offline transport requires an explicit offline context checker")
    gateway_class = ConcurrentGateway if workers > 1 else RequestGateway
    gateway = gateway_class(model_config=model_config, run_dir=root / "provider",
                            fingerprint=plan["fingerprint"], token_budget=token_budget,
                            reserve_budget=True, transport=transport,
                            **({"max_concurrent_requests": workers} if workers > 1 else {}))
    executed = 0
    counter_lock = threading.Lock()
    stopped = threading.Event()
    state, error_type = "complete", None

    def stop_admission():
        # Close admission under the same lock used for durable request admission.
        # Existing wire requests still settle normally; no retry or repair starts.
        with gateway.lock:
            stopped.set()
            if not gateway.summary()["paused"]:
                gateway._pause("Few-shot worker stopped; inspect the retained failure before new work")
                gateway.db.commit()

    def check_admission():
        if stopped.is_set() or (root / "PAUSE").exists():
            raise ExecutionPaused("Few-shot execution stopped or operator pause")

    def save_running_status():
        with counter_lock:
            completed = executed
        save(root / "status.json", {"status": "running", "error_type": None,
                                    "new_request_units": completed, "ledger": gateway.summary(),
                                    "execution_mode": mode, "workers": workers})

    def pending_requests():
        for folder, episode, _support in episodes(episodes_root):
            # Decode precisely the bytes checked against the frozen plan. If
            # input files change during a long run, later episodes cannot use
            # the changed data under the earlier campaign fingerprint.
            payloads = {name: (folder / name).read_bytes()
                        for name in ("episode.json", "support.json")}
            actual = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
            if actual != plan["episode_input_hashes"].get(episode["episode_id"]):
                raise ValueError("Frozen episode inputs changed during execution")
            episode = json.loads(payloads["episode.json"])
            support = json.loads(payloads["support.json"])
            for request in all_requests(episode, support, conditions):
                result_path = root / "predictions" / request["request_hash"] / "result.json"
                if result_path.exists():
                    old = load(result_path)
                    if old.get("request_hash") != request["request_hash"]:
                        raise ValueError("Saved prediction identity mismatch")
                    if old.get("execution_mode") != mode:
                        raise ValueError("Saved prediction execution mode mismatch")
                    continue
                yield request

    def run_request(request):
        nonlocal executed
        result_path = root / "predictions" / request["request_hash"] / "result.json"
        try:
            check_admission()
            save(root / "heartbeat.json", {"status": "running", "episode_id": request["episode_id"],
                                            "condition": request["condition"], "chunk_index": request["chunk_index"],
                                            "workers": workers})
            save(result_path.with_name("request.json"), request)
            generated = []

            def generate(prompt, attempt):
                check_admission()
                check = checker(request["system"], prompt)
                save(result_path.with_name(f"context_{attempt}.json"), check)
                check_admission()
                response = gateway.generate(
                    system=request["system"], user=prompt, run_name="fewshot_cr",
                    metadata={"episode_id": request["episode_id"],
                              "condition": request["condition"]["name"],
                              "chunk_index": request["chunk_index"], "attempt": attempt})
                generated.append(response)
                return response

            try:
                result = run_self_check(original_prompt=request["user"], generate=generate,
                                        validator=ScoringResponseValidator(request["expected_ids"]),
                                        policy=SelfCheckPolicy(max_repairs=plan["max_repairs"]),
                                        repair_prompt_builder=repair_prompt)
            except ContextTooLong:
                result = {"status": "context_limit_exceeded", "value": None, "attempts": generated,
                          "usage_summary": {key: sum((a.get("usage_summary") or {}).get(key, 0) or 0
                                                     for a in generated)
                                            for key in ("input_tokens", "output_tokens", "total_tokens")}}
            result.update({k: request[k] for k in
                           ("episode_id", "condition", "chunk_index", "expected_ids", "request_hash")})
            result["execution_mode"] = mode
            save(result_path, result)
            with counter_lock:
                executed += 1
        except BaseException:
            stop_admission()
            raise

    try:
        save_running_status()
        if checker is None:
            checker = GoogleContextCheck(model_config, root / "token_checks")
        requests = pending_requests()
        if workers == 1:
            for request in requests:
                if max_units is not None and executed >= max_units:
                    state = "unit_limit_reached"
                    break
                run_request(request)
                save_running_status()
        else:
            submitted, exhausted = 0, False
            pending = set()
            # Keep only workers futures admitted. Executor shutdown waits for all
            # in-flight requests before the shared ledger is summarized/closed.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    while pending or not exhausted:
                        while not exhausted and not stopped.is_set() and len(pending) < workers:
                            request = next(requests, None)
                            if request is None:
                                exhausted = True
                                break
                            if max_units is not None and submitted >= max_units:
                                state, exhausted = "unit_limit_reached", True
                                break
                            pending.add(pool.submit(run_request, request))
                            submitted += 1
                        if not pending:
                            break
                        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in finished:
                            future.result()
                        save_running_status()
                except BaseException:
                    stop_admission()
                    for future in pending:
                        future.cancel()
                    raise
    except BaseException as error:
        stop_admission()
        state = "paused" if isinstance(error, ExecutionPaused) else "error"
        error_type = type(error).__name__
        # Do not serialize exception text: third-party exceptions may embed credentials.
        save(root / "failure.json", {"status": state, "error_type": error_type})
        raise
    finally:
        summary = {"status": state, "error_type": error_type, "new_request_units": executed,
                   "ledger": gateway.summary(), "execution_mode": mode, "workers": workers}
        save(root / "status.json", summary)
        save(root / "heartbeat.json", {"status": state})
        gateway.close()
        if isinstance(checker, GoogleContextCheck):
            checker.close()
    return summary
