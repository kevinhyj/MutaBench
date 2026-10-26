"""Strict synthetic identity repair checks; no data, labels or API connection."""
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/offline_ablation/recover_identifier_prefix.py'
spec = importlib.util.spec_from_file_location('ppi_identifier_checks', path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class IdentifierTests(unittest.TestCase):
    ids = ['v_' + 'a' * 20, 'v_' + 'b' * 20]

    def body(self, rows, finish='STOP'):
        return {'candidates': [{'finishReason': finish, 'content': {'parts': [
            {'text': '```json\n' + json.dumps({'variants': rows}) + '\n```'}]}}]}

    def rows(self):
        return [{'variant_id': 'a' * 20, 'predicted_score': .2},
                {'variant_id': self.ids[1], 'predicted_score': -.3}]

    def test_only_exact_prefix_is_restored_and_scores_unchanged(self):
        result = recovery.normalize_body(self.body(self.rows()), self.ids)
        self.assertEqual(result['scores'], dict(zip(self.ids, [.2, -.3])))
        self.assertEqual(result['identifier_mappings'], [{'original_id': 'a' * 20, 'canonical_id': self.ids[0]}])

    def test_unknown_duplicate_missing_and_invalid_scores_rejected(self):
        cases = []
        rows = self.rows(); rows[0]['variant_id'] = 'c' * 20; cases.append(rows)
        rows = self.rows(); rows[1]['variant_id'] = self.ids[0]; cases.append(rows)
        cases.append(self.rows()[:1])
        for value in [True, None, float('nan'), 1.1]:
            rows = self.rows(); rows[0]['predicted_score'] = value; cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError):
                recovery.normalize_body(self.body(rows), self.ids)

    def test_refused_incomplete_and_already_correct_outputs_are_not_repaired(self):
        for finish in ('SAFETY', 'MAX_TOKENS', 'OTHER'):
            with self.assertRaises(ValueError):
                recovery.normalize_body(self.body(self.rows(), finish), self.ids)
        rows = self.rows(); rows[0]['variant_id'] = self.ids[0]
        with self.assertRaises(ValueError):
            recovery.normalize_body(self.body(rows), self.ids)


if __name__ == '__main__':
    unittest.main()
