"""Synthetic provider/interrupt regression checks; never connects to a gateway."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1] / 'tools/offline_ablation'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('ppi_transport_checks', DIRECTORY / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
dispatch_spec = importlib.util.spec_from_file_location('ppi_dispatch_checks', DIRECTORY / 'dispatch_frozen.py')
dispatch = importlib.util.module_from_spec(dispatch_spec)
dispatch_spec.loader.exec_module(dispatch)


class TransportTests(unittest.TestCase):
    def test_google_proxy_route_uses_api_key_header_without_bearer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, config, common = self.fixture(root)
            config.update(proxy='http://127.0.0.1:7777', authentication='x-goog-api-key',
                          endpoint='https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent')
            captured = []
            def send(request, **kwargs):
                captured.append(request)
                return self.reply()
            opener = types.SimpleNamespace(open=send)
            with patch.object(runner.urllib.request, 'build_opener', return_value=opener), \
                 patch.object(runner.urllib.request, 'ProxyHandler', wraps=runner.urllib.request.ProxyHandler) as proxy:
                result = runner.run_unit(root, unit, config, 'synthetic-token', common)
            self.assertEqual(result['status'], 'ok')
            proxy.assert_called_once_with({'http': 'http://127.0.0.1:7777', 'https': 'http://127.0.0.1:7777'})
            self.assertEqual(captured[0].get_header('X-goog-api-key'), 'synthetic-token')
            self.assertIsNone(captured[0].get_header('Authorization'))
            self.assertEqual(json.loads(captured[0].data), runner.read(root / unit['request_path']))

    def test_quota_recovery_preserves_failed_response_and_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, _, _ = self.fixture(root)
            runner.atomic(root / 'plan.json', [unit])
            attempt = root / 'attempts/unit/001.json'
            runner.atomic(attempt, dict(http_status=402, request_sha256=unit['request_sha256'],
                                       body={'error': {'code': 'insufficient_quota'}}))
            response = root / 'responses/unit.json'
            runner.atomic(response, dict(unit_id='unit', status='http_error', http_status=402,
                                        request_sha256=unit['request_sha256'], raw_attempt=str(attempt)))
            response_bytes, attempt_bytes = response.read_bytes(), attempt.read_bytes()
            result = dispatch.recover_quota_failures(runner, root)
            self.assertEqual(len(result), 1)
            self.assertFalse(response.exists())
            self.assertEqual(Path(result[0]['archived_response']).read_bytes(), response_bytes)
            self.assertEqual(attempt.read_bytes(), attempt_bytes)

    def test_quota_recovery_does_not_retry_authentication_or_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, _, _ = self.fixture(root)
            runner.atomic(root / 'plan.json', [unit])
            response = root / 'responses/unit.json'
            for status, code in (('permanent_refusal', 403), ('http_error', 403), ('delivery_unknown', None)):
                runner.atomic(response, dict(unit_id='unit', status=status, http_status=code,
                                            request_sha256=unit['request_sha256']))
                before = response.read_bytes()
                self.assertEqual(dispatch.recover_quota_failures(runner, root), [])
                self.assertEqual(response.read_bytes(), before)

    def test_raw_env_and_json_credentials_are_loaded_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in (
                ('raw.key', 'synthetic-token\n'),
                ('credentials.env', '# synthetic\nexport GEMINI_API_KEY="synthetic-token"\n'),
                ('provider.json', json.dumps({'api_key': 'synthetic-token', 'provider_id': 'synthetic'})),
                ('unexpanded.env', 'GEMINI_API_KEY="$(synthetic-command)"\n')):
                path = root / name
                path.write_text(text)
                path.chmod(0o600)
                expected = '$(synthetic-command)' if name == 'unexpanded.env' else 'synthetic-token'
                self.assertEqual(runner.load_credential(path), expected)

    def test_insecure_missing_and_duplicate_credentials_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'credentials.env'
            for text, mode in (('GEMINI_API_KEY=synthetic-token', 0o640),
                               ('UNRELATED_KEY=synthetic-token', 0o600),
                               ('GEMINI_API_KEY=one\nGEMINI_API_KEY=two', 0o600),
                               ('GEMINI_API_KEY="two words"', 0o600)):
                path.write_text(text)
                path.chmod(mode)
                with self.assertRaises(ValueError):
                    runner.load_credential(path)

    def fixture(self, root):
        request_path = root / 'requests/unit.json'
        runner.atomic(request_path, {'synthetic': True})
        unit = dict(id='unit', group_id='synthetic_assay', condition='structure',
                    request_path='requests/unit.json', request_sha256=runner.file_hash(request_path),
                    expected_ids=['one', 'two'])
        config = dict(max_attempts=3, endpoint='https://invalid.test', timeout_seconds=2,
                      requested_model='gemini-3.8-flash')
        common = types.SimpleNamespace(parse=lambda body, ids: dict(status='ok', scores={'one': .1, 'two': .2}))
        return unit, config, common

    def reply(self, model='gemini-3.8-flash-exp-a'):
        class Reply:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return json.dumps({'modelVersion': model}).encode()
        return Reply()

    def test_started_marker_does_not_consume_a_completed_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, config, common = self.fixture(root)
            previous = root / 'attempts/unit/001.json'
            runner.atomic(previous, {'status': 'transport_error'})
            runner.atomic(previous.with_name('001.started.json'), {'started': 1})
            old = previous.read_bytes()
            opener = types.SimpleNamespace(open=lambda *args, **kwargs: self.reply())
            with patch.object(runner.urllib.request, 'build_opener', return_value=opener):
                result = runner.run_unit(root, unit, config, 'synthetic-token', common)
            self.assertEqual(result['attempt'], 2)
            self.assertEqual(previous.read_bytes(), old)
            self.assertTrue((root / 'attempts/unit/002.json').exists())
            self.assertFalse((root / 'attempts/unit/003.json').exists())

    def test_unpaired_start_marker_is_not_resent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, config, common = self.fixture(root)
            marker = root / 'attempts/unit/001.started.json'
            runner.atomic(marker, {'started': 1})
            old = marker.read_bytes()
            with patch.object(runner.urllib.request, 'build_opener') as opener:
                result = runner.run_unit(root, unit, config, 'synthetic-token', common)
            opener.assert_not_called()
            self.assertEqual(result['status'], 'delivery_unknown')
            self.assertEqual(marker.read_bytes(), old)

    def test_other_model_is_terminal_and_not_used_for_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, config, common = self.fixture(root)
            opener = types.SimpleNamespace(open=lambda *args, **kwargs: self.reply('gemini-3.8-pro'))
            with patch.object(runner.urllib.request, 'build_opener', return_value=opener):
                result = runner.run_unit(root, unit, config, 'synthetic-token', common)
            self.assertEqual(result['status'], 'model_identity_mismatch')
            self.assertTrue(result['terminal'])
            self.assertNotIn('scores', result)

    def test_first_authentication_failure_stops_before_queue_is_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, config, common = self.fixture(root)
            key = root / 'credential'
            key.write_text('synthetic-token')
            key.chmod(0o600)
            config.update(credential_file=str(key), concurrency=10)
            runner.atomic(root / 'config.json', config)
            runner.atomic(root / 'plan.json', [unit, {**unit, 'id': 'unit2'}])
            with patch.object(runner, 'verify_freeze'), patch.object(runner, 'legacy_common', return_value=common), \
                 patch.object(runner, 'run_unit', return_value=dict(status='http_error', http_status=403)) as call:
                runner.run(root)
            self.assertEqual(call.call_count, 1)

    def test_previously_excluded_cohort_is_the_67_outside_reviewed_queue(self):
        sequence = {f'assay_{index:02d}': {} for index in range(88)}
        reviewed = {f'assay_{index:02d}': {} for index in range(21)}
        excluded = runner.ppi_cohort(sequence, reviewed, 'previously_excluded_67')
        kept = runner.ppi_cohort(sequence, reviewed, 'reviewed_21')
        union = runner.ppi_cohort(sequence, reviewed, 'executable_88')
        self.assertEqual(len(excluded), 67)
        self.assertEqual(len(kept), 21)
        self.assertTrue(set(excluded).isdisjoint(kept))
        self.assertEqual(set(excluded) | set(kept), set(sequence))
        self.assertEqual(union, set(sequence))

    def test_native_pro_preview_uses_medium_thinking_route(self):
        self.assertEqual(
            runner.native_generate_endpoint('gemini-3.1-pro-preview'),
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent')
        self.assertEqual(runner.SCORING_MODELS['gemini-3.1-pro-preview'], 'MEDIUM')

    def test_flash_thinking_levels_are_frozen_per_directory(self):
        self.assertEqual(runner.normalize_thinking_level('high'), 'HIGH')
        self.assertEqual(runner.resolve_thinking_level('gemini-3.8-flash', 'LOW'), 'LOW')
        self.assertEqual(runner.resolve_thinking_level('gemini-3.1-pro-preview', None), 'MEDIUM')
        with self.assertRaises(ValueError):
            runner.normalize_thinking_level('max')
        with self.assertRaises(ValueError):
            runner.resolve_thinking_level('gemini-3.1-pro-preview', 'HIGH')

    def test_medium_clone_only_changes_thinking_level(self):
        clone_spec = importlib.util.spec_from_file_location(
            'ppi_clone_thinking', DIRECTORY / 'clone_ppi_thinking_from_medium.py')
        clone = importlib.util.module_from_spec(clone_spec)
        clone_spec.loader.exec_module(clone)
        request = {
            'contents': [{'role': 'user', 'parts': [{'text': 'frozen-prompt'}]}],
            'generationConfig': {
                'temperature': 0, 'seed': 42, 'maxOutputTokens': 65536,
                'thinkingConfig': {'includeThoughts': True, 'thinkingLevel': 'MEDIUM'},
            },
        }
        high = clone.retarget_request(request, 'HIGH')
        self.assertEqual(high['contents'], request['contents'])
        self.assertEqual(high['generationConfig']['thinkingConfig']['thinkingLevel'], 'HIGH')
        self.assertEqual(high['generationConfig']['maxOutputTokens'], 65536)
        self.assertEqual(request['generationConfig']['thinkingConfig']['thinkingLevel'], 'MEDIUM')
        with self.assertRaises(ValueError):
            clone.retarget_request(request, 'MEDIUM')

    def test_payload_writes_requested_thinking_level(self):
        common = types.SimpleNamespace(INSTRUCTION=(
            'Use only the supplied sequence and assay context. '
            'No coordinates, structural features, MSA, or experimental variant measurements are supplied.\n'))
        task = dict(modality='protein-protein', new_assay_context={},
                    reference_chains=[dict(chain='A', polymer_type='protein', reference_sequence='ACDE')])
        unit = dict(expected_ids=['v_aaa'])
        features = dict(structure_definition=dict(partner_groups=[['A']]))
        protocol = dict(max_output_tokens=16384, thinking_level='HIGH', requested_model='gemini-3.8-flash')
        with patch.object(runner, 'base_sites', return_value=(None, {'v_aaa': [{'chain': 'A', 'position': 1, 'wt': 'A', 'mutant': 'C'}]})), \
             patch.object(runner, 'render_evidence', return_value={}):
            request = runner.payload(task, unit, features, 'structure', protocol, common)
        self.assertEqual(request['generationConfig']['thinkingConfig']['thinkingLevel'], 'HIGH')
        self.assertEqual(request['generationConfig']['seed'], 42)
        self.assertEqual(request['generationConfig']['temperature'], 0)

    def test_unknown_or_resized_historical_cohort_is_rejected(self):
        sequence = {f'assay_{index:02d}': {} for index in range(88)}
        reviewed = {f'assay_{index:02d}': {} for index in range(21)}
        with self.assertRaises(ValueError):
            runner.ppi_cohort(sequence, reviewed, 'all_88')
        with self.assertRaises(ValueError):
            runner.ppi_cohort(sequence, {f'assay_{index:02d}': {} for index in range(20)}, 'reviewed_21')


if __name__ == '__main__':
    unittest.main()
