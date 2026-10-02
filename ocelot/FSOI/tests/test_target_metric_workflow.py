"""Configuration, audit persistence, and raw combined-closure regression tests."""

from pathlib import Path
import ast
import io
from contextlib import redirect_stdout
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_target_metric_config import make_config  # noqa: E402
from fsoi_target_metric import (begin_target_cycle, combined_closure, get_target_plan,  # noqa: E402
                                save_target_plans, configure_observation_verification,
                                metric_provenance, SparseTargetError)
from summarize_target_coverage import summarize  # noqa: E402
from test_target_metric import CONFIG, fixture  # noqa: E402


class WorkflowTests(unittest.TestCase):
    def test_real_inference_audit_entrypoint_uses_balanced_metric_without_forward(self):
        # Load the real orchestration function without importing unavailable HPC model dependencies.
        path = Path(__file__).resolve().parents[1] / 'fsoi_inference.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'compute_fsoi_for_pair')
        namespace = dict(torch=torch, np=np, begin_target_cycle=begin_target_cycle,
                         configure_observation_verification=configure_observation_verification,
                         get_target_plan=get_target_plan, metric_provenance=metric_provenance,
                         SparseTargetError=SparseTargetError,
                         _as_scalar_bin=lambda x: x, _expand_seviri_instrument_aliases=lambda x: x)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        model, batch = fixture()
        batch.to = lambda device: batch
        batch.bin_name = '2025070112'
        fc = dict(target_instruments=['radiosonde'], lead_steps=[0],
                  verification_metric=dict(CONFIG, audit_only=True))
        with redirect_stdout(io.StringIO()):
            result = namespace['compute_fsoi_for_pair'](
                model, batch, batch, {'forecast': fc}, {}, {}, {}, 5,
                verbose=False, verification_target='obs')
        self.assertEqual(model.calls, 0)
        self.assertFalse(model.training)
        self.assertEqual(len(model._fsoi_target_plans), 1)
        self.assertEqual(result['fsoi_by_step'], {})
        self.assertEqual(model.fsoi_verification_loss_kwargs['target_instruments'], ['radiosonde'])

    def test_fd_requires_explicit_shared_target_not_source_network(self):
        from fsoi_validation import _validation_loss_kwargs
        model, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'Configure the balanced'):
            _validation_loss_kwargs(model, 'atms', 0)
        model.fsoi_verification_loss_kwargs = dict(target_instruments=['radiosonde'],
                                                   use_area_weights=False, loss_reduction='mean')
        result = _validation_loss_kwargs(model, 'atms', 0)
        self.assertEqual(result['target_instruments'], ['radiosonde'])
        self.assertFalse(result['use_area_weights'])

    def test_real_configs_generate_explicit_audit_and_run_metrics(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        for name in ('radiosonde_all', 'aircraft', 'surface_obs'):
            base = yaml.safe_load((root / f'fsoi_config_{name}.yaml').read_text(encoding='utf-8'))
            for mode in ('audit', 'run'):
                config = make_config(base, mode=mode)
                fc = config['forecast']
                self.assertNotIn('use_area_weights', fc)
                self.assertEqual(fc['loss_reduction'], 'mean')
                self.assertEqual(fc['verification_metric']['audit_only'], mode == 'audit')
                self.assertEqual(fc['stratify_by_pressure'], name != 'surface_obs')
                self.assertTrue(fc['stratify_by_variable'])

    def test_all_observation_configs_use_the_single_balanced_objective(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        for path in root.glob('fsoi_config*.yaml'):
            if 'mesh' in path.name:
                continue
            forecast = yaml.safe_load(path.read_text(encoding='utf-8'))['forecast']
            metric = configure_observation_verification(forecast)
            self.assertEqual(metric['spatial_weighting'], 'equal_area')
            self.assertEqual(metric['group_balancing'], 'variable_level')

    def test_missing_config_defaults_to_balanced_and_alternatives_fail(self):
        fc = {'target_instruments': ['radiosonde']}
        self.assertEqual(configure_observation_verification(fc)['spatial_weighting'], 'equal_area')
        for flag in ('use_area_weights', 'use_instrument_weights', 'use_channel_weights'):
            with self.assertRaises(ValueError):
                configure_observation_verification(dict(fc, **{flag: True}))

    def test_config_does_not_mutate_base_and_rejects_bad_groups(self):
        base = {'forecast': {'target_instruments': ['radiosonde']}}
        make_config(base)
        self.assertNotIn('verification_metric', base['forecast'])
        with self.assertRaises(ValueError):
            make_config(base, levels=[4])

    def test_audit_writes_failed_groups_and_exact_weights(self):
        model, batch = fixture()
        begin_target_cycle(model, dict(CONFIG, pressure_levels_hpa=[1000, 850, 500]))
        plan = get_target_plan(model, batch['radiosonde_target_step0'], 'radiosonde', 'radiosonde_target_step0')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'run' / 'evaluation' / 'target_metric'
            save_target_plans(model, directory, 0, '2025070100')
            summary = summarize(temp)
            self.assertEqual(summary['cycles'].sum(), 6)
            self.assertEqual(summary.loc[summary.pressure_hpa.eq(500), 'eligible_cycles'].sum(), 0)
            with np.load(next(directory.glob('*.npz'))) as saved:
                np.testing.assert_array_equal(saved['weights'], plan.weights)

    def test_closure_never_uses_population_expansion(self):
        frame = pd.DataFrame(dict(pair_idx=[0, 0, 1, 1], lead_step=[0] * 4,
                                  instrument=['atms', 'amsua'] * 2, curr_bin=['a', 'a', 'b', 'b'],
                                  ea=[2., 2., 1., 1.], eb=[1., 1., 1., 1.],
                                  sum_impact=[.25, .75, 0., 0.], sum_impact_scaled=[100.] * 4,
                                  control_repeat_abs_difference=[1e-8, 1e-8, np.nan, np.nan]))
        result = combined_closure(frame)
        self.assertEqual(result.iloc[0].closure_ratio, 1.)
        self.assertEqual(result.iloc[0].signal_threshold, 1e-7)
        self.assertFalse(result.iloc[1].signal_valid)
        self.assertTrue(np.isnan(result.iloc[1].closure_ratio))


if __name__ == '__main__':
    unittest.main()
