"""CPU tests of target masking, spatial/group weights, and the real FSOI loss paths."""

import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsoi_target_metric import (  # noqa: E402
    SparseTargetError, begin_target_cycle, build_target_plan, equal_area_cells,
    get_target_plan, masked_weighted_loss, target_validity,
)
from fsoi_utils import (  # noqa: E402
    _reduce_weighted_error, compute_forecast_error, compute_per_level_fsoi,
    compute_per_level_fsoi_by_variable,
)


CONFIG = dict(spatial_weighting='equal_area', n_sin_lat=2, n_lon=4,
              pressure_levels_hpa=[1000, 850], min_cells_per_group=1,
              min_observations_per_group=1, min_observations_per_cell=1)


class Batch(dict):
    @property
    def node_types(self):
        return list(self)

    @property
    def edge_types(self):
        return []

    def clone(self):
        return Batch({
            k: SimpleNamespace(**{
                name: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for name, value in vars(node).items()})
            for k, node in self.items()})


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64), requires_grad=False)
        self.device = torch.device('cpu')
        self.instrument_channels = {'radiosonde': [{'variable_name': 'temperature'}, {'variable_name': 'u_wind'}]}
        self.instrument_name_to_id = {'atms': 0, 'radiosonde': 1}
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        n = batch['radiosonde_target_step0'].y.shape[0]
        value = batch['atms_input'].x[:, 7:].sum()
        return {'radiosonde_target': [value * torch.tensor([1., 2.], dtype=torch.float64).expand(n, 2)]}


def fixture():
    y = torch.zeros((6, 2), dtype=torch.float64)
    mask = torch.ones_like(y, dtype=torch.bool)
    y[0, 1], mask[0, 1] = -9., False
    target = SimpleNamespace(y=y, target_channel_mask=mask,
                             lat=torch.tensor([0., 0., 0., 0., 0., 0.]),
                             lon=torch.tensor([-140., -140., 40., -140., 40., 40.]),
                             pressure_level=torch.tensor([0, 0, 0, 2, 2, 2]))
    source = SimpleNamespace(x=torch.zeros((1, 8), dtype=torch.float64),
                             input_channel_mask=torch.ones((1, 1), dtype=torch.bool))
    return Model(), Batch(radiosonde_target_step0=target, atms_input=source)


class TargetMetricTests(unittest.TestCase):
    def plan(self, config=None):
        model, batch = fixture()
        begin_target_cycle(model, config or CONFIG)
        return model, batch, get_target_plan(model, batch['radiosonde_target_step0'],
                                             'radiosonde', 'radiosonde_target_step0')

    def test_equal_area_geometry_and_boundary_wrap(self):
        edges = np.arcsin(np.linspace(-1, 1, 19))
        areas = np.diff(np.sin(edges)) * (2 * np.pi / 36)
        np.testing.assert_allclose(areas, 4 * np.pi / (18 * 36))
        ids = equal_area_cells([-90, 90, 0, 0, 91], [-180, 180, -180, 180, 0])
        self.assertEqual(ids[2], ids[3])
        self.assertTrue(0 <= ids[0] < 648 and 0 <= ids[1] < 648)
        self.assertEqual(ids[4], -1)

    def test_1000_vs_10_reports_have_equal_cell_weight(self):
        n = 1010
        plan = build_target_plan(np.zeros((n, 1)), np.ones((n, 1), bool), np.zeros(n),
                                 np.r_[np.full(1000, -140.), np.full(10, 40.)], None,
                                 ['temperature'], CONFIG)
        self.assertAlmostEqual(plan.weights[:1000].sum(), .5)
        self.assertAlmostEqual(plan.weights[1000:].sum(), .5)
        p = torch.tensor(np.r_[np.full(1000, 2.), np.ones(10)][:, None])
        self.assertAlmostEqual(plan.loss(p, torch.zeros_like(p)).item(), 2.5)

    def test_group_weights_are_fixed_and_normalized(self):
        _, _, plan = self.plan()
        self.assertAlmostEqual(plan.weights.sum(), 1.)
        for key, group in plan.groups.items():
            self.assertAlmostEqual(group['weights'].sum(), 1.)
            self.assertAlmostEqual(group['coefficient'], .25)
        self.assertEqual(plan.weights[0, 1], 0.)

    def test_equal_variable_weights_with_unequal_level_counts(self):
        _, _, plan = self.plan(dict(CONFIG, levels_by_variable={'temperature': [1000]}))
        self.assertAlmostEqual(plan.weights[:, 0].sum(), .5)
        self.assertAlmostEqual(plan.weights[:, 1].sum(), .5)

    def test_group_decomposition_and_gradients(self):
        _, batch, plan = self.plan()
        ref = batch['radiosonde_target_step0'].y
        p = torch.arange(12., dtype=torch.float64).reshape(6, 2).requires_grad_()
        combined = plan.loss(p, ref)
        summed = sum(g['coefficient'] * plan.loss(p, ref, group=k) for k, g in plan.groups.items())
        torch.testing.assert_close(combined, summed)
        torch.testing.assert_close(torch.autograd.grad(combined, p)[0], torch.autograd.grad(summed, p)[0])

    def test_nonfinite_and_sentinel_invalid_targets_do_not_enter_loss(self):
        ref = torch.tensor([[-9., float('nan'), 1.]])
        p = torch.tensor([[float('nan'), float('inf'), 3.]], requires_grad=True)
        loss = masked_weighted_loss(p, ref, torch.tensor([[0., 0., .1]]))
        self.assertEqual(loss.item(), 4.)
        torch.testing.assert_close(torch.autograd.grad(loss, p)[0], torch.tensor([[0., 0., 4.]]))

    def test_nonfinite_active_prediction_fails(self):
        with self.assertRaises(RuntimeError):
            masked_weighted_loss(torch.tensor([[float('nan')]]), torch.zeros(1, 1), torch.ones(1, 1))

    def test_zero_loss_is_valid_and_has_zero_gradient(self):
        p = torch.ones(1, 1, requires_grad=True)
        loss = masked_weighted_loss(p, p.detach(), torch.ones_like(p))
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(torch.autograd.grad(loss, p)[0].item(), 0.)

    def test_small_weight_denominator_not_clamped_to_one(self):
        self.assertAlmostEqual(_reduce_weighted_error(torch.tensor([.4]), torch.tensor([.1]),
                                                      loss_reduction='mean').item(), 4.)

    def test_empty_groups_fail_not_zero(self):
        with self.assertRaises(SparseTargetError):
            masked_weighted_loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))

    def test_only_canonical_target_mask_is_used(self):
        node = SimpleNamespace(y=torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            target_validity(node)
        node.valid_mask = torch.ones(1, 2, dtype=torch.bool)
        with self.assertRaises(ValueError):
            target_validity(node)
        node.target_channel_mask = torch.tensor([[True, False]])
        torch.testing.assert_close(target_validity(node), node.target_channel_mask)

    def test_sparse_cycle_has_records_but_no_renormalized_loss(self):
        _, batch, plan = self.plan(dict(CONFIG, min_observations_per_group=3))
        self.assertFalse(plan.eligible)
        self.assertEqual(len(plan.records), 4)
        with self.assertRaises(SparseTargetError):
            plan.loss(torch.zeros(6, 2), batch['radiosonde_target_step0'].y)

    def test_absent_requested_level_stays_in_audit(self):
        _, _, plan = self.plan(dict(CONFIG, pressure_levels_hpa=[1000, 850, 500]))
        missing = [r for r in plan.records if r['pressure_hpa'] == 500]
        self.assertEqual(len(missing), 2)
        self.assertTrue(all(not r['eligible'] for r in missing))

    def test_empty_target_store_is_audited_without_coordinates(self):
        model, _ = fixture()
        begin_target_cycle(model, CONFIG)
        node = SimpleNamespace(y=torch.empty(0, 2), target_channel_mask=torch.empty(0, 2, dtype=torch.bool))
        plan = get_target_plan(model, node, 'radiosonde', 'radiosonde_target_step0')
        self.assertEqual(len(plan.records), 4)
        self.assertFalse(plan.eligible)
        self.assertTrue(all(r['n_scored_targets'] == 0 for r in plan.records))

    def test_surface_has_variable_groups_without_pressure(self):
        model, batch = fixture()
        model.instrument_channels['surface_obs'] = model.instrument_channels['radiosonde']
        node = batch['radiosonde_target_step0']
        del node.pressure_level
        begin_target_cycle(model, CONFIG)
        plan = get_target_plan(model, node, 'surface_obs', 'surface_obs_target_step0')
        self.assertEqual(len(plan.groups), 2)
        self.assertAlmostEqual(plan.weights[:, 0].sum(), .5)

    def test_unknown_variables_and_pressures_fail(self):
        model, batch, _ = self.plan()
        with self.assertRaises(ValueError):
            get_target_plan(model, batch['radiosonde_target_step0'], 'radiosonde', 'radiosonde_target_step0', ['typo'])
        with self.assertRaises(ValueError):
            get_target_plan(model, batch['radiosonde_target_step0'], 'radiosonde', 'radiosonde_target_step0', levels=[2])

    def test_frozen_targets_change_detected_between_endpoints(self):
        model, batch, _ = self.plan()
        batch['radiosonde_target_step0'].target_channel_mask[0, 1] = True
        with self.assertRaisesRegex(RuntimeError, 'changed within'):
            get_target_plan(model, batch['radiosonde_target_step0'], 'radiosonde', 'radiosonde_target_step0')

    def test_alternative_spatial_weighting_is_rejected(self):
        for mode in ('uniform', 'cosine'):
            with self.assertRaisesRegex(ValueError, 'equal_area'):
                self.plan(dict(CONFIG, spatial_weighting=mode))

    def test_level_perturbation_only_changes_that_group(self):
        _, batch, plan = self.plan()
        y = batch['radiosonde_target_step0'].y
        p = torch.ones_like(y)
        baseline = plan.loss(p, y, group=(2, 0))
        p[:3, 0] += 100
        torch.testing.assert_close(plan.loss(p, y, group=(2, 0)), baseline)

    def test_real_combined_loss_uses_mask_and_matches_plan(self):
        model, batch, plan = self.plan()
        batch['atms_input'].x[:, 7] = 1.
        with redirect_stdout(io.StringIO()):
            actual = compute_forecast_error(model, batch, 0, {}, {}, False, ['radiosonde'], loss_reduction='mean')
        expected = plan.loss(model(batch)['radiosonde_target'][0], batch['radiosonde_target_step0'].y)
        torch.testing.assert_close(actual, expected)

    def test_default_metric_requires_coverage_instead_of_pooled_fallback(self):
        model, batch = fixture()
        batch['atms_input'].x[:, 7] = 1.
        with self.assertRaises(SparseTargetError):
            compute_forecast_error(model, batch, 0, {}, {}, False, ['radiosonde'])
        self.assertEqual(model.fsoi_verification_metric['spatial_weighting'], 'equal_area')

    def test_combined_loss_rejects_cosine_and_unnamed_networks(self):
        model, batch, _ = self.plan()
        with self.assertRaisesRegex(ValueError, 'cosine'):
            compute_forecast_error(model, batch, 0, {}, {}, True, ['radiosonde'])
        with self.assertRaisesRegex(ValueError, 'one named target'):
            compute_forecast_error(model, batch, 0, {}, {})

    def test_loss_rejects_unnormalized_sum(self):
        with self.assertRaisesRegex(ValueError, 'normalized-mean'):
            masked_weighted_loss(torch.ones(1, 1), torch.zeros(1, 1), torch.ones(1, 1), 'sum')

    def test_real_stratified_paths_include_zero_loss_and_close_quadratic(self):
        for by_variable in (True, False):
            model, batch, plan = self.plan()
            xa = {'atms': torch.ones((1, 1), dtype=torch.float64, requires_grad=True)}
            xb = {'atms': torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)}
            fn = compute_per_level_fsoi_by_variable if by_variable else compute_per_level_fsoi
            with redirect_stdout(io.StringIO()):
                results = fn(model, batch, xa, xb, {'satellite': {'atms': {'features': ['bt']}}},
                             0, {}, {}, target_instruments=['radiosonde'], loss_reduction='mean',
                             valid_masks={'atms': torch.ones(1, 1, dtype=torch.bool)})
            self.assertEqual(model.calls, 2)
            self.assertEqual(len(results), 4 if by_variable else 2)
            predicted = sum(r['group_weight'] * r['fsoi_values']['atms'].sum().item() for r in results)
            actual = sum(r['group_weight'] * (r['ea_p'] - r['eb_p']) for r in results)
            self.assertAlmostEqual(predicted, actual)
            self.assertAlmostEqual(actual, 2.5)

    def test_matched_replacement_mask_and_path_reuse_target_weights(self):
        from fsoi_ose import compute_matched_conditional_fsoi_for_pair

        for mode in ('background_replacement', 'full_mask'):
            model, batch, plan = self.plan()
            batch['atms_input'].x[:, 7] = 1.
            with redirect_stdout(io.StringIO()):
                result = compute_matched_conditional_fsoi_for_pair(
                    model, batch, {'atms': torch.ones(1, 1, dtype=torch.float64)},
                    {'atms': torch.zeros(1, 1, dtype=torch.float64)}, ['atms'],
                    {'satellite': {'atms': {'features': ['bt']}}}, {}, ['radiosonde'], None, None,
                    {}, {}, False, 'mean', 0, 0, '20250701_12', '20250701_00',
                    denial_mode=mode, run_control_repro_check=True,
                    path_integration_t_values=[0., .25, .5, .75, 1.] if mode == 'background_replacement' else None)
            self.assertAlmostEqual(result['ea_control'], 2.5)
            self.assertAlmostEqual(result['ea_denied'], 0.)
            self.assertAlmostEqual(result['matched_fsoi'], 2.5)
            self.assertEqual(result['target_metric_ids'], plan.metric_id)
            if mode == 'background_replacement':
                self.assertAlmostEqual(result['path_integrated_fsoi'], 2.5)
                self.assertEqual(model.calls, 6)  # endpoints + 3 interior points + one reproducibility check
            else:
                self.assertEqual(model.calls, 3)
            self.assertFalse(batch['radiosonde_target_step0'].target_channel_mask[0, 1])

    def test_directional_difference_on_combined_metric(self):
        _, batch, plan = self.plan()
        y = batch['radiosonde_target_step0'].y
        p = torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(6, 2).requires_grad_()
        v = torch.where(torch.arange(12).reshape(6, 2) % 2 == 0, 1., -1.)
        ad = (torch.autograd.grad(plan.loss(p, y), p)[0] * v).sum()
        eps = 1e-4
        fd = (plan.loss(p + eps * v, y) - plan.loss(p - eps * v, y)) / (2 * eps)
        torch.testing.assert_close(ad, fd, atol=1e-7, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
