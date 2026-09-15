"""Regression tests for the pre-submission fixes: frozen target groups, xb decoder
conditioning, group-weighted channel audits, and candidate loss weights."""

from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

FSOI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FSOI_DIR))
from summarize_target_coverage import freeze, load_records, summarize  # noqa: E402
import make_target_metric_config  # noqa: E402
from audit_sampling_rank_sensitivity import read_cycle_scalings  # noqa: E402
import compute_fsoi_weights  # noqa: E402

SETTINGS = dict(n_sin_lat=18, n_lon=36, min_observations_per_cell=2,
                min_cells_per_group=3, min_observations_per_group=10)


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


def write_audit(run, eligible, n_inventory=None):
    """eligible maps (variable, channel, hPa) -> per-cycle eligibility flags."""
    directory = run / 'evaluation' / 'target_metric'
    directory.mkdir(parents=True)
    n_cycles = len(next(iter(eligible.values())))
    for pair in range(n_cycles):
        rows = [dict(SETTINGS, target_instrument='radiosonde', variable=v, target_channel=ch,
                     pressure_hpa=p, eligible=flags[pair], n_cells=5, n_scored_targets=20,
                     n_invalid_coordinates=0, pair_idx=pair)
                for (v, ch, p), flags in eligible.items()]
        pd.DataFrame(rows).to_csv(directory / f'pair{pair:04d}_x.csv', index=False)
    if n_inventory:
        pd.DataFrame(dict(pair_idx=range(n_inventory))).to_csv(
            run / 'evaluation' / 'target_metric_cycles.csv', index=False)


class FrozenTargetGroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        always, never = [True] * 4, [False] * 4
        write_audit(self.root / 'radiosonde_jan', {
            ('temperature', 1, 1000.): always, ('temperature', 1, 10.): always,
            ('dewpoint_temperature', 2, 1000.): always, ('dewpoint_temperature', 2, 10.): never,
            ('u_wind', 3, 10.): never})
        # The fifth July cycle had no radiosonde target node at all.
        write_audit(self.root / 'radiosonde_jul', {
            ('temperature', 1, 1000.): always, ('temperature', 1, 10.): [True, True, False, False],
            ('dewpoint_temperature', 2, 1000.): always, ('dewpoint_temperature', 2, 10.): never,
            ('u_wind', 3, 10.): never}, n_inventory=5)

    def test_freeze_keeps_groups_supported_in_every_run(self):
        raw = load_records(self.root)
        spec = freeze(raw, summarize(self.root, raw), 0.75)['radiosonde']
        self.assertEqual(spec['target_variables'], ['temperature', 'dewpoint_temperature'])
        self.assertEqual(spec['levels_by_variable'],
                         {'temperature': [1000.0], 'dewpoint_temperature': [1000.0]})
        self.assertEqual(spec['metric_settings'], SETTINGS)
        retention = {Path(k).name: v for k, v in spec['expected_cycle_retention'].items()}
        self.assertEqual(retention, {'radiosonde_jan': 1.0, 'radiosonde_jul': 0.8})
        dropped = {(d['variable'], d['pressure_hpa']) for d in spec['dropped_groups']}
        self.assertIn(('temperature', 10.0), dropped)
        self.assertIn(('u_wind', 10.0), dropped)

    def test_run_config_requires_and_applies_the_frozen_spec(self):
        raw = load_records(self.root)
        spec_path = self.root / 'frozen.json'
        spec_path.write_text(json.dumps(freeze(raw, summarize(self.root, raw), 0.75)))
        output = self.root / 'radiosonde_target_run.yaml'
        argv = ['make', '--target', 'radiosonde', '--mode', 'run', '--output', str(output)]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                make_target_metric_config.main()
        with patch.object(sys, 'argv', argv + ['--frozen-spec', str(spec_path)]):
            quiet(make_target_metric_config.main)
        config = yaml.safe_load(output.read_text(encoding='utf-8'))
        forecast = config['forecast']
        self.assertEqual(forecast['target_variables'], ['temperature', 'dewpoint_temperature'])
        self.assertEqual(forecast['verification_metric']['pressure_levels_hpa'], [1000.0])
        self.assertFalse(forecast['verification_metric']['audit_only'])
        self.assertEqual(config['target_metric_freeze']['spec_file'], str(spec_path))

    def test_levels_outside_target_variables_are_rejected(self):
        base = {'forecast': {'target_instruments': ['radiosonde']}}
        with self.assertRaises(ValueError):
            make_target_metric_config.make_config(
                base, variables=['temperature'], levels_by_variable={'u_wind': [500]})


class GroupWeightedAuditTests(unittest.TestCase):
    def test_target_groups_are_averaged_with_frozen_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'channels.csv'
            pd.DataFrame(dict(instrument=['atms'] * 2, channel=[1, 1], pair_idx=[0, 0],
                              sum_impact=[1., 5.], sum_impact_scaled=[1., 5.], sample_scale=[1.] * 2,
                              total_count=[1, 1], raw_total_count=[1, 1],
                              group_weight=[.75, .25])).to_csv(path, index=False)
            cycle, _, _ = read_cycle_scalings(path, 'radiosonde', 'jul2025')
        self.assertAlmostEqual(cycle.raw_total.item(), 2.0)


@unittest.skipUnless(all(__import__('importlib').util.find_spec(m) for m in ('torch', 'torch_geometric', 'psutil')),
                     'requires torch, torch_geometric, and psutil')
class PseudoTargetConditioningTests(unittest.TestCase):
    def setUp(self):
        import torch
        sys.path.insert(0, str(FSOI_DIR.parent))
        self.torch = torch
        self.source = SimpleNamespace(
            lat=torch.tensor([10., 20., 30.]), lon=torch.tensor([0., 90., -90.]),
            pressure_level=torch.tensor([0, 4, 15]),
            input_times=torch.tensor([1751328000, 1751350000, 1751370000], dtype=torch.int64))

    def test_rows_carry_level_and_time_like_real_targets(self):
        from fsoi_model_extensions import pseudo_target_conditioning
        from process_timeseries import _encode_target_time_features
        idx = self.torch.tensor([2, 0])
        out = pseudo_target_conditioning(self.source, 'radiosonde', idx)
        self.assertEqual(out['pressure_level'].tolist(), [15, 0])
        self.assertEqual(tuple(out['target_metadata'].shape), (2, 7))
        expected = _encode_target_time_features(np.array([1751370000, 1751328000]), np.array([-90., 0.]))
        np.testing.assert_allclose(out['target_metadata'][:, 2:].numpy(), expected, atol=1e-6)
        np.testing.assert_allclose(out['target_metadata'][:, 0].numpy(), np.deg2rad([30., 10.]), atol=1e-6)
        self.assertNotIn('pressure_level', pseudo_target_conditioning(self.source, 'atms'))

    def test_missing_level_or_time_is_an_error(self):
        from fsoi_model_extensions import pseudo_target_conditioning
        no_level = SimpleNamespace(**{k: v for k, v in vars(self.source).items() if k != 'pressure_level'})
        with self.assertRaisesRegex(ValueError, 'pressure_level'):
            pseudo_target_conditioning(no_level, 'aircraft')
        no_time = SimpleNamespace(**{k: v for k, v in vars(self.source).items() if k != 'input_times'})
        with self.assertRaisesRegex(ValueError, 'input_times'):
            pseudo_target_conditioning(no_time, 'atms')


class CandidateWeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        for run in ('radiosonde_jan2025', 'radiosonde_jul2025', 'aircraft_jan2025'):
            rows, channel_rows = [], []
            for cycle in range(8):
                ea = 2.0
                impacts = dict(atms=-0.4 + 0.02 * rng.standard_normal(),
                               amsua=0.2 + 0.02 * rng.standard_normal(),
                               ssmis=0.3 * (-1) ** cycle)
                for inst, value in impacts.items():
                    rows.append(dict(curr_bin=f'b{cycle}', lead_step=0, instrument=inst, ea=ea,
                                     sum_impact_ht=value, mean_impact_hajek=value / 100,
                                     population_scaling_method='horvitz_thompson_row_inclusion',
                                     target_metric_config=run.split('_')[0]))
                for channel, value in ((1, -0.3), (2, 0.2 * (-1) ** cycle)):
                    channel_rows.append(dict(rows[0], curr_bin=f'b{cycle}', instrument='atms',
                                             channel=channel, sum_impact_ht=value))
            (self.root / run / 'csv').mkdir(parents=True)
            pd.DataFrame(rows).to_csv(self.root / run / 'csv' / compute_fsoi_weights.INSTRUMENT_CSV, index=False)
            pd.DataFrame(channel_rows).to_csv(self.root / run / 'csv' / compute_fsoi_weights.CHANNEL_CSV, index=False)
        self.obs = self.root / 'observation_config.yaml'
        self.obs.write_text(yaml.safe_dump(dict(
            instrument_weights=dict(atms=1.0, amsua=1.0, ssmis=1.0),
            channel_weights=dict(atms=[1.0, 1.0, 1.0]),
            observation_config=dict(satellite=dict(atms=dict(features=['a', 'b', 'c']))))))

    def test_signs_intervals_and_weights(self):
        out = self.root / 'weights'
        argv = ['weights', '--fsoi_dirs', str(self.root / '*_*2025'), '--expected-runs', '3',
                '--obs_config', str(self.obs), '--output_dir', str(out), '--n-boot', '300']
        with patch.object(sys, 'argv', argv):
            quiet(compute_fsoi_weights.main)
        summary = pd.read_csv(out / 'instrument_impact_summary.csv', index_col='instrument')
        self.assertEqual(summary.loc['atms', 'status_total'], 'beneficial')
        self.assertEqual(summary.loc['amsua', 'status_total'], 'detrimental')
        self.assertEqual(summary.loc['ssmis', 'status_total'], 'inconclusive')
        w = summary['weight_total']
        self.assertGreater(w['atms'], w['ssmis'])
        self.assertGreater(w['ssmis'], w['amsua'])
        self.assertAlmostEqual(w.mean(), 1.0)
        # Targets have equal weight: two radiosonde months do not outweigh one aircraft month.
        self.assertAlmostEqual(summary.loc['atms', 'impact_total'], -0.2, delta=0.02)
        channel_cfg = yaml.safe_load((out / 'observation_config_candidate_total_channel.yaml').read_text())
        weights = channel_cfg['channel_weights']['atms']
        self.assertEqual(len(weights), 3)
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(weights[1], weights[2])

    def test_old_per_group_outputs_are_rejected(self):
        run = self.root / 'radiosonde_oct2025' / 'csv'
        run.mkdir(parents=True)
        pd.DataFrame(dict(curr_bin=['b0'], lead_step=[0], instrument=['atms'], ea=[1.],
                          sum_impact_scaled=[1.])).to_csv(run / compute_fsoi_weights.INSTRUMENT_CSV, index=False)
        with self.assertRaisesRegex(ValueError, 'missing'):
            compute_fsoi_weights.load_runs([run.parent], compute_fsoi_weights.INSTRUMENT_CSV)


if __name__ == '__main__':
    unittest.main()
