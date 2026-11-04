"""Audit committed evidence without MLX, model weights, or private artifacts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'reports/evidence/t4'


def read(name):
    return json.loads((EVIDENCE / name).read_text())


class PublicEvidenceTests(unittest.TestCase):
    def test_frozen_sources(self):
        for name, expected in read('frozen-contract.json')['source_sha256'].items():
            with self.subTest(path=name):
                actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_paired_results(self):
        rows = [json.loads(line) for line in
                (EVIDENCE / 'paired-transitions.jsonl').read_text().splitlines()]
        frozen = read('frozen-contract.json')
        report = read('recomputed-comparison.json')
        ids = [row['sample_id'] for row in rows]
        self.assertEqual(ids, frozen['test_ids'])
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(rows), report['first_pass_predictions_per_arm'])
        transitions = Counter()
        for row in rows:
            before, after = row['baseline_strict'], row['adapter_strict']
            self.assertIs(type(before), bool)
            self.assertIs(type(after), bool)
            expected = ('both_correct' if before and after else
                        'improved' if after else 'regressed' if before else 'both_wrong')
            self.assertEqual(row['transition'], expected)
            transitions[expected] += 1
        self.assertEqual(dict(transitions), report['transitions'])
        self.assertEqual(Counter(row['expected_action'] for row in rows),
                         {'call': 95, 'no_call': 95})
        for arm in ('baseline', 'adapter'):
            summary = read(arm + '-summary.json')
            self.assertEqual(summary, report['results'][arm])
            metric = summary['strict_decision_match']
            correct = sum(row[arm + '_strict'] for row in rows)
            self.assertEqual(metric['numerator'], correct)
            self.assertEqual(metric['denominator'], len(rows))
            self.assertAlmostEqual(metric['value'], correct / len(rows))


if __name__ == '__main__':
    unittest.main()
