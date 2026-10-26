"""Authentication recovery must preserve history and never retry a refusal."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/offline_ablation/protein_rna.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('protein_rna_auth_checks', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AuthenticationRecoveryTests(unittest.TestCase):
    def scenario(self, prior_status, *, resume, error_type='gateway_error', response=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credential = root / 'credential'
            credential.write_text('synthetic-credential')
            credential.chmod(0o600)
            runner.write(root / 'config.json', dict(concurrency=1, source_base=str(root), max_attempts=3,
                protocol=dict(total_deadline_seconds=3, timeout_seconds=3, endpoint='https://invalid.test',
                              requested_model='gemini-3.8-flash')))
            unit = dict(id='unit', condition='structure', expected_ids=['one', 'two'],
                        request_sha256='frozen-request', request_path='request.json')
            runner.write(root / 'index.json', [unit])
            runner.write(root / 'request.json', {'synthetic': True})
            attempt_path = root / 'attempts/unit/001.json'
            attempt = dict(status=prior_status, http_status=403, body=dict(error=dict(
                type=error_type, message='API token required. Provide sk-*.')))
            runner.write(attempt_path, attempt)
            response_path = root / 'responses/unit.json'
            if response:
                runner.write(response_path, dict(status=prior_status, request_sha256='frozen-request',
                                                 raw_attempt=str(attempt_path)))
            calls = []

            class FakeClient:
                def __init__(self, *args, **kwargs):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                async def post(self, endpoint, **kwargs):
                    calls.append(endpoint)
                    return types.SimpleNamespace(status_code=200,
                        json=lambda: dict(modelVersion='gemini-3.8-flash', usageMetadata={'totalTokenCount': 1}))

            common = types.SimpleNamespace(parse=lambda body, ids: dict(status='ok', scores={'one': .1, 'two': .2}))
            args = types.SimpleNamespace(output=root, credential_file=credential, resume_auth_failures=resume)
            with patch.object(runner, 'validate'), patch.object(runner, 'legacy_module', return_value=common), patch('httpx.AsyncClient', FakeClient):
                asyncio.run(runner.run(args))
            outcome = runner.read(response_path)
            return (calls, outcome, (root / 'auth_failure_responses/unit.json').exists(),
                    attempt_path.exists(), (root / 'attempts/unit/002.json').exists())

    def test_proven_token_rejection_resumes_with_failure_history_intact(self):
        calls, outcome, history, old_attempt, new_attempt = self.scenario('provider_blocked', resume=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(outcome['status'], 'ok')
        self.assertTrue(history and old_attempt and new_attempt)

    def test_no_resume_flag_never_retries_authentication(self):
        calls, outcome, *_ = self.scenario('provider_blocked', resume=False)
        self.assertEqual(calls, [])
        self.assertEqual(outcome['status'], 'provider_blocked')

    def test_resume_flag_cannot_retry_explicit_refusal_or_other_403(self):
        for status, error_type in [('explicit_refusal', 'gateway_error'), ('provider_blocked', 'policy_error')]:
            calls, outcome, history, *_ = self.scenario(status, resume=True, error_type=error_type)
            self.assertEqual(calls, [])
            self.assertEqual(outcome['status'], status)
            self.assertFalse(history)

    def test_unknown_delivery_requires_review_even_with_resume_flag(self):
        calls, outcome, *_ = self.scenario('started', resume=True, response=False)
        self.assertEqual(calls, [])
        self.assertEqual(outcome['status'], 'requires_review')


if __name__ == '__main__':
    unittest.main()
