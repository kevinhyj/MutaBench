"""Default Gemini transport parses env credentials and does not send OAuth."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/offline_ablation/protein_rna_v2.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('rna_native_auth_checks', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeAuthenticationTests(unittest.TestCase):
    def test_plain_env_and_quoted_export_entries_read_same_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'credential'
            for text in ('synthetic-token', 'GEMINI_API_KEY=synthetic-token\n',
                         'export GEMINI_API_KEY="synthetic-token"\nOTHER=unused\n'):
                path.write_text(text)
                path.chmod(0o600)
                self.assertEqual(runner.credential_value(path), 'synthetic-token')
            path.write_text('GEMINI_API_KEY=one\nGEMINI_API_KEY=two\n')
            with self.assertRaises(ValueError):
                runner.credential_value(path)

    def test_native_proxy_wire_uses_token_value_and_no_bearer_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credential = root / 'credentials.env'
            credential.write_text('GEMINI_API_KEY=synthetic-token\n')
            credential.chmod(0o600)
            runner.write(root / 'config.json', dict(concurrency=5, source_base=str(root), max_attempts=3,
                protocol=dict(total_deadline_seconds=3, timeout_seconds=3,
                              endpoint='https://generativelanguage.googleapis.com/v1beta/models/synthetic:generateContent',
                              http_proxy='http://127.0.0.1:7777', requested_model='gemini-3.8-flash')))
            runner.write(root / 'index.json', [dict(id='synthetic', expected_ids=['a', 'b'],
                request_sha256='frozen', request_path='request.json')])
            runner.write(root / 'request.json', {'synthetic': True})
            observed = {}

            class FakeClient:
                def __init__(self, *args, **kwargs):
                    observed['proxy'] = kwargs.get('proxy')

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                async def post(self, endpoint, **kwargs):
                    observed['headers'] = kwargs['headers']
                    return types.SimpleNamespace(status_code=200,
                        json=lambda: dict(modelVersion='gemini-3.8-flash', usageMetadata={'totalTokenCount': 1}))

            args = types.SimpleNamespace(output=root, credential_file=credential, resume_auth_failures=False)
            common = types.SimpleNamespace(parse=lambda body, ids: dict(status='ok', scores={'a': .1, 'b': -.2}))
            with patch.object(runner, 'validate'), patch.object(runner, 'legacy_module', return_value=common), patch('httpx.AsyncClient', FakeClient):
                asyncio.run(runner.run(args))
            self.assertEqual(observed['proxy'], 'http://127.0.0.1:7777')
            self.assertEqual(observed['headers'], {'x-goog-api-key': 'synthetic-token'})
            self.assertEqual(runner.read(root / 'responses/synthetic.json')['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
