"""Direct and optional Deep Agents adapters using one EVAgent request gateway.

Benchmark tool/eviction policy is adapted from ypxia's rnagym_deep_agent/agent.py.
The ledger-backed model and instance-local harness factory are integration work;
they intentionally replace the collaborator's global profile and provider retries.
"""

from __future__ import annotations

import inspect
import sqlite3
import sys
import threading
import types
import uuid
from pathlib import Path
from typing import Any

from .gateway import RequestGateway, digest


def _instance_profile_factory(factory, profile):
    """Supply a per-instance profile without modifying Deep Agents' registry.

    Deep Agents 0.7.13 exposes profile selection only through a private global
    resolver. Copying the factory's globals gives this graph its own resolver;
    no module, registry, model identifier, or other running agent is changed.
    A future public profile argument is preferred automatically. Fail closed if
    the pinned factory's private seam changes, rather than silently enabling tools.
    """
    if "harness_profile" in inspect.signature(factory).parameters:
        return lambda **kwargs: factory(harness_profile=profile, **kwargs)
    if not isinstance(factory, types.FunctionType) or (
        "_harness_profile_for_model" not in factory.__code__.co_names
    ):
        raise RuntimeError("Unsupported Deep Agents factory: no isolated harness profile interface")
    isolated = types.FunctionType(
        factory.__code__,
        {**factory.__globals__, "_harness_profile_for_model": lambda model, spec: profile},
        name=factory.__name__, argdefs=factory.__defaults__, closure=factory.__closure__,
    )
    isolated.__kwdefaults__ = factory.__kwdefaults__
    isolated.__annotations__ = factory.__annotations__
    return isolated


def _chat_model(gateway: RequestGateway):
    # All optional framework imports stay inside the optional backend.
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    class LedgerChatModel(BaseChatModel):
        model_name: str
        _gateway: Any = PrivateAttr()
        _calls: list = PrivateAttr(default_factory=list)
        _run_name: str = PrivateAttr(default="experiment")
        _request_metadata: dict = PrivateAttr(default_factory=dict)

        @property
        def _llm_type(self):
            return "evagent-ledger"

        @property
        def _identifying_params(self):
            return self._gateway.model_config

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            if tools or tool_choice not in (None, "none"):
                raise ValueError("Tools are disabled for this benchmark backend")
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self._calls:
                self._gateway.pause("Unexpected additional model call in one-shot benchmark graph")
            if stop or kwargs.get("tools"):
                raise ValueError("Unexpected stop/tool override in the benchmark backend")
            encoded = []
            for message in messages:
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(message.type)
                if role is None or getattr(message, "tool_calls", None):
                    raise ValueError("Benchmark backend accepts only textual messages")
                content = message.content
                if isinstance(content, list):
                    if any(not isinstance(block, dict) or block.get("type") != "text"
                           for block in content):
                        raise ValueError("Native multimodal message blocks require a later protocol")
                    content = "".join(block["text"] for block in content)
                if not isinstance(content, str):
                    raise ValueError("Benchmark message content must be text")
                encoded.append({"role": role, "content": content})
            result = self._gateway.invoke_messages(
                messages=encoded, run_name=self._run_name,
                metadata=self._request_metadata,
            )
            self._calls.append(result)
            # Terminal failures remain in the returned result; no model repair or
            # tool loop is attempted. Empty text is permitted for a blocked turn.
            message = AIMessage(content=result.get("final_text", ""), response_metadata={
                **result.get("response_metadata", {}),
                "finish_reason": result.get("finish_reason"),
                "request_sha256": result["request_sha256"], "status": result["status"],
            })
            return ChatResult(generations=[ChatGeneration(message=message)])

    model = LedgerChatModel(model_name=gateway.model_config["model"], cache=False)
    model._gateway = gateway
    return model


class DeepAgentsGenerator:
    def __init__(self, gateway: RequestGateway):
        if sys.version_info < (3, 11):
            raise RuntimeError("Deep Agents requires Python >=3.11; use the .venv-agent environment")
        try:
            from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, create_deep_agent
            from deepagents.backends import StateBackend
            from deepagents.middleware.filesystem import FilesystemMiddleware
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("Install the agent extra in .venv-agent to use Deep Agents") from exc
        self.gateway = gateway
        self.lock = threading.RLock()
        self.model = _chat_model(gateway)
        self.backend = StateBackend()
        self.profile = HarnessProfile(
            excluded_tools=frozenset({
                "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
                "execute", "task", "write_todos",
            }),
            excluded_middleware=frozenset({"SummarizationMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        )
        self.factory = _instance_profile_factory(create_deep_agent, self.profile)
        self.filesystem_middleware = FilesystemMiddleware(
            backend=self.backend, tools=["read_file"], tool_token_limit_before_evict=None,
            human_message_token_limit_before_evict=None,
        )
        self.checkpoint_path = gateway.run_dir / "deepagents_checkpoints.sqlite"
        self.checkpoint_connection = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.checkpoint_connection)
        self.graph = None

    def generate(self, *, system: str, user: str, run_name: str,
                 metadata: dict | None = None) -> dict:
        with self.lock:
            self.model._calls = []
            self.model._run_name = run_name
            self.model._request_metadata = metadata or {}
            self.graph = self.factory(
                model=self.model, tools=[], subagents=[], memory=None, skills=None,
                system_prompt=system, backend=self.backend,
                middleware=[self.filesystem_middleware], checkpointer=self.checkpointer,
                name="evagent-benchmark",
            )
            # Checkpoints are execution state, not a prediction cache. A fresh
            # thread prevents retry/restart from appending to an old conversation;
            # actual identical provider requests still hit the gateway cache.
            thread = digest([self.gateway.fingerprint, system, user, metadata or {}])
            self.graph.invoke(
                {"messages": [{"role": "user", "content": user}]},
                config={"configurable": {"thread_id": f"{thread}-{uuid.uuid4().hex}"},
                        "recursion_limit": 8},
            )
            if len(self.model._calls) != 1:
                self.gateway.pause("Deep Agents benchmark did not complete exactly one model call")
            calls = self.model._calls
            result = next((call for call in calls if call["status"] != "ok"), calls[-1])
            return {**result, "model_call_records": list(calls)}

    def summary(self) -> dict:
        return self.gateway.summary()

    def close(self) -> None:
        self.checkpoint_connection.close()
        self.gateway.close()


def build_generator(backend: str, *, model_config: dict, run_dir: Path,
                    fingerprint: str, token_budget: int, transport=None):
    if backend not in ("direct", "deepagents"):
        raise ValueError(f"Unknown backend: {backend}")
    gateway = RequestGateway(
        model_config=model_config, run_dir=run_dir, fingerprint=fingerprint,
        token_budget=token_budget, transport=transport,
    )
    if backend == "direct":
        return gateway
    try:
        return DeepAgentsGenerator(gateway)
    except BaseException:
        gateway.close()
        raise
