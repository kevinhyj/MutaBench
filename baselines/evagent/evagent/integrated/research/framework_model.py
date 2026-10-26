"""LangChain's agent execution loop backed exclusively by RequestGateway.

The adapter keeps Gemini's signed native response parts unchanged and binds
declarative tools per invocation. It does not execute tools, retry models,
summarize messages, or use a second prediction cache. Legacy backends are not
affected. See Google's thought-signature guide and LangChain BaseChatModel.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from evagent.integrated.gateway import RequestGateway, assert_public, digest
from evagent.runtime import ExecutionPaused

from .contracts import CallLimit, ResearchFailure


PROTOCOL = 'native_framework_v4'


def canonical_wire_schema(schema):
    """Avoid expanding bounded arrays/strings in the native ANY grammar.

    Length/cardinality limits remain in tool descriptions and in the original
    Pydantic/tool validators. Only the provider-facing copy is simplified; it is
    the copy hashed and retained by RequestGateway. Structural types, required
    fields, enums, closed objects and numerical bounds are preserved.
    """
    if isinstance(schema, list):
        return [canonical_wire_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    result = copy.deepcopy(schema)
    # Recurse only through schema positions. Enum/default/example values are data
    # and must remain byte-for-byte equivalent even when they contain schema keys.
    for key in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
        if isinstance(result.get(key), dict):
            result[key] = {name: canonical_wire_schema(value) for name, value in result[key].items()}
    for key in ('items', 'additionalProperties', 'contains', 'not', 'if', 'then', 'else',
                'propertyNames', 'unevaluatedProperties', 'anyOf', 'allOf', 'oneOf', 'prefixItems'):
        if isinstance(result.get(key), (dict, list)):
            result[key] = canonical_wire_schema(result[key])
    kind = schema.get('type')
    keys = {'string': ('minLength', 'maxLength', 'characters'),
            'array': ('minItems', 'maxItems', 'items')}.get(kind) if isinstance(kind, str) else None
    if keys is not None:
        minimum, maximum, unit = keys
        limits = []
        for key, label in ((minimum, 'minimum'), (maximum, 'maximum')):
            if key in result:
                value = result.pop(key)
                if type(value) is not int or value < 0:
                    raise ValueError('Tool length/cardinality limits must be nonnegative integers')
                limits.append(f'{label} {value} {unit}')
        if limits:
            suffix = 'Local validation requires ' + ', '.join(limits) + '.'
            result['description'] = (str(result.get('description', '')).strip() + ' ' + suffix).strip()
    return result


def _text(content):
    if isinstance(content, str):
        return content
    if (isinstance(content, list) and all(isinstance(block, dict)
            and block.get('type') == 'text' and isinstance(block.get('text'), str)
            for block in content)):
        return ''.join(block['text'] for block in content)
    raise ValueError('Framework v4 only accepts complete text/numerical evidence')


def encode_messages(messages):
    """Preserve native model parts and resolve every tool result to its call.

    Provider call IDs may be absent. LangChain still needs stable IDs, so local
    IDs are kept separately; signed provider parts are never edited to add one.
    """
    encoded, pending = [], {}
    for message in messages:
        content = _text(message.content)
        if isinstance(message, ToolMessage):
            call = pending.pop(message.tool_call_id, None)
            if call is None or message.name not in (None, call['name']):
                raise ValueError('Tool result does not match an outstanding native call')
            try:
                response = json.loads(content)
            except json.JSONDecodeError:
                response = {'result': content}
            if not isinstance(response, dict):
                response = {'result': response}
            function_response = {'name': call['name'], 'response': response}
            if call['provider_id'] is not None:
                function_response['id'] = call['provider_id']
            encoded.append({'role': 'tool', 'content': content,
                            'tool_call_id': message.tool_call_id,
                            'native_parts': [{'function_response': function_response}]})
            continue
        if pending:
            raise ValueError('Every native call must receive exactly one tool result')
        if isinstance(message, SystemMessage):
            encoded.append({'role': 'system', 'content': content})
        elif isinstance(message, HumanMessage):
            encoded.append({'role': 'user', 'content': content})
        elif isinstance(message, AIMessage):
            parts = copy.deepcopy(message.additional_kwargs.get('google_native_parts', []))
            calls = [part['function_call'] for part in parts if part.get('function_call')]
            if len(calls) != len(message.tool_calls):
                raise ValueError('Native signed parts and LangChain tool calls disagree')
            for native, call in zip(calls, message.tool_calls):
                if (native['name'] != call['name'] or native.get('args', {}) != call['args']
                        or not isinstance(call['id'], str) or not call['id']
                        or call['id'] in pending
                        or native.get('id') is not None and native['id'] != call['id']):
                    raise ValueError('Native tool identity or arguments were modified')
                pending[call['id']] = {'name': call['name'], 'provider_id': native.get('id')}
            row = {'role': 'assistant', 'content': content}
            if parts:
                row['native_parts'] = parts
            encoded.append(row)
        else:
            raise ValueError(f'Unsupported framework message: {type(message).__name__}')
    if pending:
        raise ValueError('Native tool result is missing before the next model call')
    return encoded


class LedgerAgentModel(BaseChatModel):
    """A bounded role-specific BaseChatModel usable by ``create_agent``.

    ``records`` includes actual transport attempts (including permitted retries),
    or cached attempt references on replay. ExecutionPaused is never translated
    into a tool error or semantic validation error.
    """

    model_name: str = 'gateway-bound'
    _gateway: Any = PrivateAttr()
    _request_metadata: dict = PrivateAttr(default_factory=dict)
    _maximum: int = PrivateAttr()
    _records: list = PrivateAttr(default_factory=list)
    _tool_maximum: int = PrivateAttr()
    _tool_call_records: list = PrivateAttr(default_factory=list)
    _invocation_lock: Any = PrivateAttr(default_factory=threading.RLock)

    def __init__(self, gateway: RequestGateway, metadata: dict, maximum: int,
                 tool_maximum: int | None = None):
        if gateway.model_config.get('research_tool_protocol') != PROTOCOL:
            raise ValueError('LedgerAgentModel requires native_framework_v4')
        if type(maximum) is not int or maximum < 1:
            raise ValueError('A positive per-role model-call allowance is required')
        tool_maximum = 2 * maximum if tool_maximum is None else tool_maximum
        if type(tool_maximum) is not int or tool_maximum < 1:
            raise ValueError('A positive per-role tool-call allowance is required')
        assert_public(metadata)
        super().__init__(model_name=gateway.model_config['model'], cache=False,
                         metadata=copy.deepcopy(metadata))
        self._gateway = gateway
        self._request_metadata = copy.deepcopy(metadata)
        self._maximum = maximum
        self._tool_maximum = tool_maximum

    @property
    def records(self):
        return self._records

    @property
    def gateway(self):
        return self._gateway

    @property
    def request_metadata(self):
        return copy.deepcopy(self._request_metadata)

    @property
    def tool_call_records(self):
        """Native proposals, including rejected/invalid calls; not executions."""
        return self._tool_call_records

    @property
    def _llm_type(self):
        return 'evagent-ledger-native-framework-v4'

    @property
    def _identifying_params(self):
        return {'model_name': self.model_name, 'protocol': PROTOCOL,
                'model_config_sha256': digest(self._gateway.model_config),
                'role_metadata_sha256': digest(self._request_metadata)}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if kwargs:
            raise ValueError(f'Unsupported framework model overrides: {sorted(kwargs)}')
        declarations = []
        for tool in tools:
            function = convert_to_openai_tool(tool)['function']
            declaration = {key: copy.deepcopy(function[key])
                           for key in ('name', 'description', 'parameters') if key in function}
            if 'parameters' in declaration:
                declaration['parameters'] = canonical_wire_schema(declaration['parameters'])
            declarations.append(declaration)
        choice = {'required': 'any'}.get(tool_choice, tool_choice) or 'auto'
        return self.bind(tool_declarations=declarations, tool_choice=choice)

    def _paused_records(self, call_hash):
        """Retrieve even the final failing attempt when gateway invocation raises."""
        with self._gateway.lock:
            rows = self._gateway.db.execute(
                "SELECT request_sha256,status,response_json,elapsed_s FROM attempts "
                "WHERE json_extract(request_json,'$.metadata.framework_call_sha256')=? ORDER BY id",
                (call_hash,),
            ).fetchall()
        return [{'request_sha256': key, 'status': status, 'cache_hit': False,
                 'usage_summary': json.loads(response or '{}').get('usage_summary'),
                 'elapsed_s': elapsed} for key, status, response, elapsed in rows]

    def _record_tool_proposals(self, records):
        """Reserve a complete native batch before LangChain executes any call.

        Even invalid tool arguments and structured-report calls consume the tool
        allowance. Failed provider attempts' partial proposals are recorded but
        never admitted. The separate tool execution journal records what ran.
        """
        journal = self._gateway.run_dir.parent / 'framework_tool_proposals'
        over_limit = False
        for record in records:
            with self._gateway.lock:
                row = self._gateway.db.execute(
                    'SELECT response_json FROM attempts WHERE request_sha256=?',
                    (record['request_sha256'],),
                ).fetchone()
            raw = json.loads(row[0] or '{}') if row else {}
            calls = raw.get('tool_calls') or []
            if not isinstance(calls, list) or not calls:
                continue
            within_limit = len(self._tool_call_records) + len(calls) <= self._tool_maximum
            over_limit = over_limit or not within_limit
            receipts = []
            for index, call in enumerate(calls):
                value = call if isinstance(call, dict) else {}
                receipts.append({
                    'request_sha256': record['request_sha256'],
                    'proposal_index': index,
                    'tool_call_id': value.get('id') or f"call_{record['request_sha256'][:20]}_{index}",
                    'tool': value.get('name'), 'arguments': value.get('args'),
                    'role': self._request_metadata.get('role'),
                    'provider_status': record['status'], 'cache_hit': bool(record.get('cache_hit')),
                    'admitted': record['status'] == 'ok' and within_limit,
                    'reason': ('tool_allowance_exceeded' if not within_limit else
                               'provider_response_not_ok' if record['status'] != 'ok' else None),
                })
            journal.mkdir(parents=True, exist_ok=True)
            with (journal / (digest(self._request_metadata) + '.jsonl')).open('a') as handle:
                for receipt in receipts:
                    handle.write(json.dumps(receipt, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            self._tool_call_records.extend(receipts)
        return over_limit

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError('Stop overrides are not part of the frozen framework protocol')
        if set(kwargs) - {'tool_declarations', 'tool_choice'}:
            raise ValueError(f'Unsupported framework model overrides: {sorted(kwargs)}')
        encoded = encode_messages(messages)
        declarations = kwargs.get('tool_declarations', [])
        choice = kwargs.get('tool_choice', 'auto')
        with self._invocation_lock:
            remaining = self._maximum - len(self._records)
            if remaining < 1:
                raise CallLimit('Framework per-role model-call allowance exhausted')
            metadata = {**self._request_metadata, 'phase': 'analysis',
                        'model_turn': len(self._records), 'provider_attempt_limit': remaining}
            metadata['framework_call_sha256'] = digest([metadata, encoded, declarations, choice])
            try:
                result = self._gateway.invoke_messages(
                    messages=encoded, metadata=metadata, run_name='research-framework-analysis',
                    tool_declarations=declarations, tool_choice=choice)
            except ExecutionPaused:
                paused = self._paused_records(metadata['framework_call_sha256'])
                self._records.extend({**record, 'role': metadata.get('role'), 'phase': 'analysis'}
                                     for record in paused)
                self._record_tool_proposals(paused)
                raise
            records = result.get('transport_attempt_records') or [{key: result.get(key) for key in (
                'request_sha256', 'status', 'cache_hit', 'usage_summary', 'elapsed_s')}]
            self._records.extend({**record, 'role': metadata.get('role'), 'phase': 'analysis'}
                                 for record in records)
            if self._record_tool_proposals(records):
                raise CallLimit('Framework native tool-call allowance exhausted before execution')
            if result['status'] != 'ok':
                raise ResearchFailure('Framework provider terminal status: ' + result['status'])
            native_parts = copy.deepcopy(result.get('native_parts', []))
            provider_calls = result.get('tool_calls', [])
            if not native_parts:
                # Mock/custom transports may provide the normalized representation.
                # The Google transport always returns the complete signed parts.
                if result.get('final_text'):
                    native_parts.append({'text': result['final_text']})
                native_parts.extend({'function_call': copy.deepcopy(call)} for call in provider_calls)
            native_calls = [part['function_call'] for part in native_parts if part.get('function_call')]
            if native_calls != provider_calls:
                self._gateway.pause('Native response parts and normalized calls disagree')
            tool_calls = [{'name': call['name'], 'args': call['args'],
                           'id': call.get('id') or f"call_{result['request_sha256'][:20]}_{index}",
                           'type': 'tool_call'} for index, call in enumerate(provider_calls)]
            usage = result['usage_summary']
            message = AIMessage(
                content=result.get('final_text', ''), tool_calls=tool_calls,
                additional_kwargs={'google_native_parts': native_parts},
                response_metadata={**result.get('response_metadata', {}),
                                   'finish_reason': result['finish_reason'], 'status': result['status'],
                                   'request_sha256': result['request_sha256']},
                usage_metadata={'input_tokens': usage['input_tokens'],
                                'output_tokens': usage['total_tokens'] - usage['input_tokens'],
                                'total_tokens': usage['total_tokens'],
                                'output_token_details': {'reasoning': usage.get('reasoning_tokens', 0)}},
            )
            return ChatResult(generations=[ChatGeneration(message=message)],
                              llm_output={'usage_summary': usage, 'request_sha256': result['request_sha256']})

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # SDK I/O remains in the one gateway, whose lock owns campaign admission.
        return await asyncio.to_thread(self._generate, messages, stop, run_manager, **kwargs)
