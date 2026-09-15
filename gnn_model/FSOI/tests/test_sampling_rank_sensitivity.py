"""Regression checks for interpreting N/n and HT result schemas."""

from pathlib import Path
import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_sampling_rank_sensitivity import main, read_cycle_scalings, rank_comparisons


class ScalingAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'channels.csv'

    def frame(self, ht=False):
        df = pd.DataFrame(dict(instrument=['atms'] * 3, channel=[1, 2, 3],
            pair_idx=[0] * 3, sum_impact=[1., -2., 3.],
            sum_impact_scaled=[2., -4., 6.], sample_scale=[2.] * 3,
            total_count=[2, 2, 2], raw_total_count=[2, 2, 2]))
        if ht:
            df['sum_impact_scaled_uniform'] = df.sum_impact_scaled
            df['sum_impact_scaled'] = [7., -3., 4.]
            df['sum_impact_ht'] = df.sum_impact_scaled
            df['population_scaling_method'] = 'horvitz_thompson_row_inclusion'
        return df

    def test_archived_schema_is_explicit(self):
        self.frame().to_csv(self.path, index=False)
        cycle, _, metrics = read_cycle_scalings(self.path, 'radiosonde', 'jul2025')
        self.assertEqual(metrics[0], 'uniform_total')
        self.assertNotIn('ht_total', cycle)
        with self.assertRaises(ValueError):
            read_cycle_scalings(self.path, 'radiosonde', 'jul2025', require_ht=True)

    def test_ht_is_baseline_and_same_sample_n_over_n_is_compared(self):
        self.frame(ht=True).to_csv(self.path, index=False)
        cycle, _, metrics = read_cycle_scalings(self.path, 'radiosonde', 'jul2025', True)
        self.assertEqual(metrics[0], 'ht_total')
        comparison = rank_comparisons(cycle, metrics)
        uniform = comparison[comparison.alternative.eq('uniform_total')].iloc[0]
        self.assertEqual(uniform.baseline, 'ht_total')
        self.assertEqual(uniform.n_rank_changes, 2)
        self.assertEqual(uniform.n_sign_changes, 0)

    def test_missing_channel_mean_stays_undefined_across_chunks(self):
        df = self.frame(ht=True)
        df.loc[2, ['sum_impact', 'sum_impact_scaled', 'sum_impact_ht',
                   'sum_impact_scaled_uniform', 'total_count']] = 0
        pd.concat([df, df]).to_csv(self.path, index=False)
        cycle, _, metrics = read_cycle_scalings(self.path, 'aircraft', 'jul2025', True, chunksize=2)
        missing = cycle[cycle.channel.eq(3)].iloc[0]
        self.assertTrue(np.isnan(missing.valid_value_mean))
        self.assertEqual(missing.ht_total, 0)
        comparison = rank_comparisons(cycle, metrics)
        self.assertEqual(comparison.loc[comparison.alternative.eq('valid_value_mean'), 'n_channels_excluded'].item(), 1)

    def test_partial_or_inconsistent_ht_is_rejected(self):
        df = self.frame(ht=True).drop(columns='population_scaling_method')
        df.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            read_cycle_scalings(self.path, 'radiosonde', 'jul2025')
        df = self.frame(ht=True)
        df.loc[0, 'population_scaling_method'] = 'uniform'
        df.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            read_cycle_scalings(self.path, 'radiosonde', 'jul2025')

    def test_cli_outputs_four_month_ht_comparisons(self):
        root = Path(self.tmp.name)
        for month in ('jan2025', 'apr2025', 'jul2025', 'oct2025'):
            dest = root / f'radiosonde_{month}' / 'csv'
            dest.mkdir(parents=True)
            self.frame(ht=True).to_csv(dest / 'fsoi_by_channel.csv', index=False)
        with patch.object(sys, 'argv', ['audit', '--root', str(root), '--require-ht', '--expected-runs', '4']):
            with contextlib.redirect_stdout(io.StringIO()):
                main()
        summary = pd.read_csv(root / 'sampling_rank_sensitivity' / 'channel_scaling_summary.csv')
        seasonal = summary[summary.month.eq('four_month_equal_mean')]
        self.assertEqual(len(seasonal), 3)
        self.assertTrue(seasonal.n_months.eq(4).all())
        comparisons = pd.read_csv(root / 'sampling_rank_sensitivity' / 'channel_rank_comparisons.csv')
        self.assertTrue(comparisons.baseline.eq('ht_total').all())
        with patch.object(sys, 'argv', ['audit', '--root', str(root), '--expected-runs', '12']):
            with self.assertRaises(ValueError):
                main()


if __name__ == '__main__':
    unittest.main()
