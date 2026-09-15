"""Frozen observation-space verification weights shared by FSOI and OSE.

Equal-area cells have uniform longitude and uniform sin(latitude) intervals.
Only errors are aggregated: predictions remain at the observation locations.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


PRESSURES = np.array([1000, 925, 850, 700, 500, 400, 300, 250,
                      200, 150, 100, 70, 50, 30, 20, 10], dtype=float)
VERSION = "balanced_target_metric_v2"


def target_metric_config(config=None):
    """One observation-space objective, with explicit coverage and group settings."""
    defaults = dict(
        spatial_weighting='equal_area', group_balancing='variable_level',
        n_sin_lat=18, n_lon=36, min_observations_per_cell=2,
        min_cells_per_group=3, min_observations_per_group=10,
        pressure_levels_hpa=PRESSURES.tolist(), levels_by_variable={},
        sparse_policy='exclude_cycle', audit_only=False,
    )
    if config is not None and not isinstance(config, dict):
        raise ValueError("verification_metric must be a configuration mapping")
    supplied = config or {}
    if set(supplied) - set(defaults):
        raise ValueError(f"Unknown verification_metric settings: {sorted(set(supplied) - set(defaults))}")
    defaults.update(supplied)
    if defaults['spatial_weighting'] != 'equal_area':
        raise ValueError("Observation-space verification requires spatial_weighting=equal_area")
    if defaults['group_balancing'] != 'variable_level':
        raise ValueError("Observation-space verification requires group_balancing=variable_level")
    if defaults['sparse_policy'] not in {'exclude_cycle', 'error'}:
        raise ValueError("sparse_policy must be exclude_cycle or error; group dropping is not allowed")
    return defaults


def configure_observation_verification(forecast):
    """Resolve defaults and reject settings that would change the balanced objective."""
    targets = forecast.get('target_instruments')
    if not isinstance(targets, list) or len(targets) != 1:
        raise ValueError("Observation-space verification requires exactly one named target network")
    for setting in ('use_area_weights', 'use_instrument_weights', 'use_channel_weights'):
        if forecast.get(setting, False):
            raise ValueError(f"{setting}=true is not supported by balanced observation-space verification")
    if str(forecast.get('loss_reduction', 'mean')).lower() not in {
            'mean', 'mse', 'normalized', 'average', 'avg'}:
        raise ValueError("Observation-space verification requires a normalized-mean J")
    forecast['loss_reduction'] = 'mean'
    forecast['verification_metric'] = target_metric_config(forecast.get('verification_metric'))
    return forecast['verification_metric']


class SparseTargetError(ValueError):
    """A prespecified verification group has insufficient target coverage."""


def equal_area_cells(latitude, longitude, n_sin_lat=18, n_lon=36):
    """Return cell IDs; -1 marks invalid coordinates. Each cell has area 4pi/K."""
    if int(n_sin_lat) != n_sin_lat or int(n_lon) != n_lon or min(n_sin_lat, n_lon) < 1:
        raise ValueError("Equal-area grid dimensions must be positive integers")
    lat, lon = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError("Latitude/longitude shape mismatch")
    valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
    ids = np.full(lat.shape, -1, dtype=np.int64)
    u = (np.sin(np.deg2rad(lat[valid])) + 1) / 2
    a = np.minimum((u * n_sin_lat).astype(int), n_sin_lat - 1)
    b = ((np.mod(lon[valid] + 180, 360) / 360) * n_lon).astype(int)
    ids[valid] = a * n_lon + b
    return ids


def pressure_indices(levels):
    indices = []
    for level in levels:
        found = np.flatnonzero(np.isclose(PRESSURES, float(level), atol=1, rtol=0))
        if len(found) != 1:
            raise ValueError(f"Unknown pressure {level!r}; specify hPa, not an index")
        indices.append(int(found[0]))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Pressure groups must be nonempty and unique")
    return indices


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def target_validity(node):
    """Require the actual data-loader mask, never infer satellite validity from zero."""
    import torch

    y = node.y
    mask = getattr(node, "target_channel_mask", None)
    if mask is None:
        raise ValueError("Missing target_channel_mask; refusing an unmasked verification loss")
    if mask.shape != y.shape:
        raise ValueError(f"Target mask shape {mask.shape} != target shape {y.shape}")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Target validity mask must contain only zero/one values")
    return mask.to(device=y.device, dtype=torch.bool) & torch.isfinite(y)


def masked_weighted_loss(prediction, reference, weights, reduction="mean", factor=1.0):
    """Select active entries BEFORE subtraction, avoiding NaN * 0 and sentinel loss."""
    import torch

    if prediction.shape != reference.shape or weights.shape != reference.shape:
        raise ValueError("Prediction, reference, and target weights must have identical shapes")
    weights = weights.to(device=prediction.device, dtype=torch.float64)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Target weights must be finite and nonnegative")
    active = weights > 0
    if not active.any():
        raise SparseTargetError("No positively weighted valid verification targets")
    p, y, w = prediction[active], reference[active], weights[active]
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise RuntimeError("Non-finite prediction/reference at an active verification target")
    # Float64 accumulation improves small loss-difference resolution; model dtype is unchanged.
    error = ((p.double() - y.double()).square() * w).sum()
    if reduction in {"mean", "mse", "normalized", "average", "avg"}:
        return factor * error / w.sum()
    raise ValueError("Observation-space verification requires a normalized-mean reduction")


@dataclass
class TargetPlan:
    weights: np.ndarray
    groups: dict
    records: list
    cell_ids: np.ndarray
    eligible: bool
    metric_id: str = ""

    def require_eligible(self):
        if not self.eligible:
            failed = [f"{r['variable']}@{r['pressure_hpa']}: {r['reason']}"
                      for r in self.records if not r['eligible']]
            raise SparseTargetError("Insufficient target coverage: " + "; ".join(failed))

    def coefficient(self, level, channel=None):
        return sum(g['coefficient'] for (p, c), g in self.groups.items()
                   if p == level and (channel is None or c == channel))

    def loss(self, prediction, reference, group=None, level=None):
        import torch

        self.require_eligible()
        w = self.weights
        if group is not None or level is not None:
            w = np.zeros_like(w)
            chosen = [group] if group is not None else [k for k in self.groups if k[0] == level]
            for key in chosen:
                g = self.groups[key]
                w[g['rows'], key[1]] = g['weights'] * g['coefficient']
        return masked_weighted_loss(prediction, reference,
                                    torch.as_tensor(w, device=prediction.device), "mean")


def build_target_plan(y, valid, latitude, longitude, pressure, channels, config,
                      selected_variables=None, selected_levels=None):
    """Construct fixed group weights from targets alone; sparse groups invalidate the cycle."""
    config = target_metric_config(config)
    y, valid = np.asarray(y), np.asarray(valid, dtype=bool)
    if y.ndim != 2 or valid.shape != y.shape or len(channels) != y.shape[1]:
        raise ValueError("Target values/masks/channel metadata are not aligned")
    if len(set(channels)) != len(channels):
        raise ValueError("Target variable names must be unique")
    lat, lon = np.asarray(latitude).reshape(-1), np.asarray(longitude).reshape(-1)
    if lat.size != y.shape[0] or lon.size != y.shape[0]:
        raise ValueError("Target coordinates are not row aligned")
    cells = equal_area_cells(lat, lon, config.get('n_sin_lat', 18), config.get('n_lon', 36))
    thresholds = [config.get('min_observations_per_cell', 2),
                  config.get('min_cells_per_group', 3), config.get('min_observations_per_group', 10)]
    if any(int(n) != n or n < 1 for n in thresholds):
        raise ValueError("Target coverage thresholds must be positive integers")
    min_cell, min_cells, min_group = map(int, thresholds)
    if pressure is not None:
        pressure = np.asarray(pressure).reshape(-1)
        if pressure.size != y.shape[0]:
            raise ValueError("Pressure index must be row aligned")
    variables = list(channels if selected_variables is None else selected_variables)
    if not variables or len(set(variables)) != len(variables) or set(variables) - set(channels):
        raise ValueError(f"Unknown/duplicate/empty target variable selection: {variables}")
    weights, groups, records = np.zeros(y.shape, dtype=np.float64), {}, []
    for variable in variables:
        ch = channels.index(variable)
        if pressure is None:
            if selected_levels is not None:
                raise ValueError("Cannot pressure-select a target with no pressure index")
            levels = [None]
        else:
            configured = config.get('levels_by_variable', {}).get(variable,
                         config.get('pressure_levels_hpa', PRESSURES.tolist()))
            levels = pressure_indices(configured)
            if selected_levels is not None:
                requested = pressure_indices(selected_levels)
                if set(requested) - set(levels):
                    raise ValueError("Requested pressures lie outside the frozen metric definition")
                levels = requested
        alpha = 1.0 / (len(variables) * len(levels))
        for level in levels:
            at_level = np.ones(y.shape[0], dtype=bool) if level is None else pressure == level
            base = at_level & valid[:, ch] & np.isfinite(y[:, ch])
            good = base & (cells >= 0)
            ids, counts = np.unique(cells[good], return_counts=True)
            eligible_cells = ids[counts >= min_cell]
            rows = np.flatnonzero(good & np.isin(cells, eligible_cells))
            b, n = len(eligible_cells), len(rows)
            eligible = b >= min_cells and n >= min_group
            rec = dict(variable=variable, target_channel=ch + 1,
                       pressure_hpa=None if level is None else float(PRESSURES[level]),
                       p_idx=level, group_weight=alpha, n_rows_at_level=int(at_level.sum()),
                       n_valid_targets=int(base.sum()), n_invalid_coordinates=int((base & ~good).sum()),
                       n_occupied_cells=len(ids), n_cells=b, n_scored_targets=n,
                       n_sparse_cell_targets=int(good.sum()) - n,
                       eligible=eligible, reason="ok" if eligible else "below_coverage_threshold",
                       spatial_weighting='equal_area', n_sin_lat=config.get('n_sin_lat', 18),
                       n_lon=config.get('n_lon', 36), min_observations_per_cell=min_cell,
                       min_cells_per_group=min_cells, min_observations_per_group=min_group)
            records.append(rec)
            if not eligible:
                continue
            _, inverse, per_cell = np.unique(cells[rows], return_inverse=True, return_counts=True)
            w = 1.0 / (b * per_cell[inverse])
            rec['weight_effective_count'] = float(1.0 / np.square(w).sum())
            rec['max_element_weight'] = float(w.max())
            groups[(level, ch)] = dict(rows=rows, weights=w, coefficient=alpha)
            weights[rows, ch] = w * alpha
    eligible = bool(records) and all(r['eligible'] for r in records)
    if eligible and not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Combined target weights do not sum to one")
    return TargetPlan(weights, groups, records, cells, eligible)


def begin_target_cycle(model, config=None):
    """Reset only at a new cycle, never between endpoint/FD/path evaluations."""
    model.fsoi_verification_metric = target_metric_config(
        config if config is not None else getattr(model, 'fsoi_verification_metric', None))
    model._fsoi_target_plans = {}


def get_target_plan(model, node, instrument, node_name, variables=None, levels=None):
    config = target_metric_config(getattr(model, 'fsoi_verification_metric', None))
    model.fsoi_verification_metric = config
    valid = _numpy(target_validity(node))
    y = _numpy(node.y).astype(np.float64)
    if y.shape[0] == 0:
        # Empty target stores from BinDataset intentionally have no coordinates.
        lat, lon = np.empty(0), np.empty(0)
        pressure = np.empty(0) if instrument in {'radiosonde', 'aircraft'} else None
    else:
        lat, lon = _numpy(node.lat).reshape(-1), _numpy(node.lon).reshape(-1)
        if instrument in {'radiosonde', 'aircraft'} and not hasattr(node, 'pressure_level'):
            raise ValueError(f"Missing stored target pressure indices for {instrument}")
        pressure = _numpy(node.pressure_level).reshape(-1) if hasattr(node, 'pressure_level') else None
    info = getattr(model, 'instrument_channels', {}).get(instrument, [])
    if len(info) != y.shape[1]:
        raise ValueError(f"Missing aligned target channel metadata for {instrument}")
    channels = [c.get('variable_name', c.get('variable', '')) for c in info]
    if not all(channels):
        raise ValueError(f"Missing target variable names for {instrument}")
    if set(config.get('levels_by_variable', {})) - set(channels):
        raise ValueError("levels_by_variable contains unknown target variables")
    digest = hashlib.sha256(json.dumps([VERSION, config, channels, variables, levels], sort_keys=True).encode())
    for array in (y, valid, lat, lon, pressure):
        if array is not None:
            # Canonical float64 representation also permits a float64 FD repeat.
            arr = np.ascontiguousarray(array, dtype=np.float64)
            digest.update(str(arr.shape).encode())
            digest.update(arr.tobytes())
    fingerprint = digest.hexdigest()
    key = (node_name, tuple(variables or ()), tuple(levels or ()))
    cache = getattr(model, '_fsoi_target_plans', {})
    if key in cache:
        plan = cache[key]
        if plan.metric_id != fingerprint:
            raise RuntimeError("Verification targets/weights changed within the cycle")
        return plan
    plan = build_target_plan(y, valid, lat, lon, pressure, channels, config, variables, levels)
    plan.metric_id = fingerprint
    for record in plan.records:
        record.update(metric_id=fingerprint, metric_version=VERSION, target_instrument=instrument,
                      target_node=node_name, cycle_eligible=plan.eligible)
    cache[key] = plan
    model._fsoi_target_plans = cache
    return plan


def metric_provenance(model):
    if getattr(model, 'fsoi_verification_target', 'obs') == 'mesh':
        return {}  # Mesh verification has a separate metric, not an observation target plan.
    plans = getattr(model, '_fsoi_target_plans', {}).values()
    ids = sorted({p.metric_id for p in plans})
    return dict(target_metric_version=VERSION, target_metric_ids=";".join(ids),
                target_metric_config=json.dumps(target_metric_config(
                    getattr(model, 'fsoi_verification_metric', None)), sort_keys=True),
                target_mask_applied=True, target_loss_accumulation_dtype="float64")


def save_target_plans(model, directory, pair_idx, cycle):
    """Write audit rows even for excluded cycles; NPZ stores exact combined weights."""
    import csv

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, plan in getattr(model, '_fsoi_target_plans', {}).items():
        stem = f"pair{pair_idx:04d}_{key[0]}_{plan.metric_id[:12]}"
        records = [dict(r, pair_idx=pair_idx, cycle=cycle) for r in plan.records]
        fields = sorted(set().union(*(r.keys() for r in records)))
        with (directory / f"{stem}.csv").open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(records)
        np.savez_compressed(directory / f"{stem}.npz", weights=plan.weights,
                            cell_ids=plan.cell_ids, metric_id=plan.metric_id)


def combined_closure(frame):
    """Compare raw sampled, all-source attribution with its matching finite error change."""
    import pandas as pd

    rows = []
    for (pair, step), values in frame.groupby(['pair_idx', 'lead_step']):
        if values['instrument'].duplicated().any() or values['ea'].nunique() != 1 or values['eb'].nunique() != 1:
            raise ValueError("Combined closure needs one record per source and identical endpoints")
        reference = frame.loc[frame['lead_step'].eq(step), 'control_repeat_abs_difference'].dropna()
        repro = float(reference.max()) if len(reference) else float('nan')
        threshold = max(1e-12, 10 * repro) if np.isfinite(repro) else 1e-12
        impact = float(values['sum_impact'].sum())  # Never use sum_impact_scaled here.
        delta = float(values['ea'].iloc[0] - values['eb'].iloc[0])
        finite = np.isfinite(impact) and np.isfinite(delta)
        signal = finite and abs(impact) > threshold and abs(delta) > threshold
        rows.append(dict(pair_idx=pair, lead_step=step, curr_bin=values['curr_bin'].iloc[0],
                         fsoi_raw_sampled=impact, delta_j_actual=delta, signal_threshold=threshold,
                         control_reproducibility_error=repro, signal_valid=signal,
                         threshold_basis='representative_control_repeat' if np.isfinite(repro) else 'floor_only',
                         sign_agreement=float(np.sign(impact) == np.sign(delta)) if signal else np.nan,
                         closure_ratio=impact / delta if signal else np.nan,
                         relative_absolute_closure_error=abs(impact - delta) / abs(delta) if signal else np.nan))
    return pd.DataFrame(rows)
