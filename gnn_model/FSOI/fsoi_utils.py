"""
FSOI Utilities - Helper functions for FSOI computation.

This module provides utilities for:
- Extracting observation values from batches
- Computing forecast errors
- Computing gradients (adjoints)
- Aggregating FSOI results
- Validation and diagnostics

Author: Azadeh Gholoubi
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
from fsoi_sampling import population_summary
from fsoi_target_metric import (
    SparseTargetError, get_target_plan,
)


# process_timeseries.py clips valid conventional observations to [-6, 6] and
# fills missing conventional channels with exactly -9.0. Satellite inputs use
# normalized zero imputation for missing channels, so use input_channel_mask
# when available for satellite validity rather than relying on this sentinel.
SENTINEL_OBS = -9.0
SENTINEL_OBS_ATOL = 1e-3
SATELLITE_MISSING_OBS = 0.0

# Backward-compatible innovation-space fallback used only when xa is unavailable
# for post-hoc scatter sampling. Since innovation = -9.0 - xb at missing
# positions, the value is a range, not an exact sentinel.
SENTINEL_INNOVATION = SENTINEL_OBS
SENTINEL_ATOL = SENTINEL_OBS_ATOL
SENTINEL_INNOVATION_LO = -12.0
SENTINEL_INNOVATION_HI = -7.0


def observation_valid_mask(x_obs: torch.Tensor) -> torch.Tensor:
    """Return True where a conventional observation tensor is not the -9 sentinel."""
    sentinel = torch.as_tensor(SENTINEL_OBS, dtype=x_obs.dtype, device=x_obs.device)
    return torch.isfinite(x_obs) & ~torch.isclose(
        x_obs,
        sentinel,
        rtol=0.0,
        atol=SENTINEL_OBS_ATOL,
    )


def _masked_fsoi_components(
    xa_tensor: torch.Tensor,
    xb_tensor: torch.Tensor,
    g_sum: torch.Tensor,
    impact_factor: float,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FSOI components while excluding missing input channels."""
    dx_raw = xa_tensor - xb_tensor
    if valid_mask is None:
        valid = observation_valid_mask(xa_tensor)
    else:
        valid = valid_mask.to(device=xa_tensor.device, dtype=torch.bool)
        valid = valid & torch.isfinite(xa_tensor) & torch.isfinite(xb_tensor)
    if dx_raw.shape != g_sum.shape or dx_raw.shape != valid.shape:
        raise RuntimeError(
            "FSOI component shape mismatch: "
            f"dx={tuple(dx_raw.shape)}, g_sum={tuple(g_sum.shape)}, "
            f"valid={tuple(valid.shape)}"
        )

    zeros = torch.zeros_like(dx_raw)
    nans = torch.full_like(dx_raw, float("nan"))
    dx_valid = torch.where(valid, dx_raw, zeros)
    g_valid = torch.where(valid, g_sum, zeros)
    fsoi = impact_factor * dx_valid * g_valid

    # Keep NaNs in diagnostics for missing channels so later means/counts use
    # only real observed values. FSOI itself is zero at missing positions.
    innovations = torch.where(valid, dx_raw, nans)
    gradient_sums = torch.where(valid, g_sum, nans)
    return fsoi, innovations, gradient_sums, valid


STANDARD_PRESSURE_LEVELS = np.array(
    [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10],
    dtype=float,
)

# BUFR subset → aircraft type mapping (AIRCAR vs AIRCFT)
AIRCRAFT_SUBSET_MAP = {
    'AIRCAR': 'AIRCAR',  # Direct ACARS/AMDAR reports
    'AIRCFT': 'AIRCFT',  # Preprocessed aircraft reports
}

# BUFR subset → surface station type mapping (ADPSFC vs SFCSHP)
SURFACE_SUBSET_MAP = {
    'ADPSFC': 'ADPSFC',  # Land/synoptic surface observations
    'SFCSHP': 'SFCSHP',  # Ship/ocean surface observations
}


def _normalize_loss_reduction(loss_reduction: str) -> str:
    """Normalize loss-reduction names used by FSOI forecast-error metrics."""
    reduction = (loss_reduction or 'sum').lower()
    if reduction in {'mean', 'mse', 'normalized', 'average', 'avg'}:
        return 'mean'
    if reduction in {'sum', 'sse', 'total'}:
        return 'sum'
    raise ValueError(f"Unsupported FSOI loss_reduction={loss_reduction!r}; use 'mean' or 'sum'")


# ─────────────────────────────────────────────────────────────────────────────
# Stratification helpers: subtype extraction for instrument×variable×subtype
# ─────────────────────────────────────────────────────────────────────────────

def detect_aircraft_subtype(station_id_arr: Optional[np.ndarray],
                            bufr_subset: Optional[str] = None) -> Optional[str]:
    """
    Infer aircraft subtype (AIRCAR vs AIRCFT) from station ID patterns or BUFR subset.

    AIRCAR: Direct ACARS/AMDAR reports (e.g., station IDs starting with 'QUQ')
    AIRCFT: Preprocessed aircraft reports (other patterns)

    Returns: 'AIRCAR', 'AIRCFT', or None if indeterminate.
    """
    # Try BUFR subset first
    if bufr_subset and bufr_subset.upper() in AIRCRAFT_SUBSET_MAP:
        return AIRCRAFT_SUBSET_MAP[bufr_subset.upper()]

    # Fallback: infer from station ID patterns
    if station_id_arr is not None and len(station_id_arr) > 0:
        sid = str(station_id_arr).upper() if not isinstance(station_id_arr, str) else station_id_arr.upper()
        if 'QUQ' in sid or 'ACARS' in sid.upper():
            return 'AIRCAR'
        elif 'AMDAR' in sid or 'AIRCFT' in sid.upper():
            return 'AIRCFT'
    return None


def detect_surface_subtype(bufr_subset: Optional[str]) -> Optional[str]:
    """
    Infer surface station type (ADPSFC vs SFCSHP) from BUFR subset code.

    ADPSFC: Land/synoptic surface observations
    SFCSHP: Ship/ocean surface observations

    Returns: 'ADPSFC', 'SFCSHP', or None if indeterminate.
    """
    if bufr_subset:
        subset_upper = bufr_subset.upper()
        if subset_upper in SURFACE_SUBSET_MAP:
            return SURFACE_SUBSET_MAP[subset_upper]
        # Fallback: detect by name
        if 'SHIP' in subset_upper or 'SFC' in subset_upper and 'SHIP' in subset_upper:
            return 'SFCSHP'
        elif 'SFC' in subset_upper or 'ADPSFC' in subset_upper:
            return 'ADPSFC'
    return None


def nearest_pressure_level(pressure_hpa: np.ndarray) -> np.ndarray:
    """
    Map pressure values (hPa) to nearest STANDARD_PRESSURE_LEVELS for radiosonde stratification.

    Returns: array of pressure level indices (or the level value itself for downstream grouping).
    """
    pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)
    nearest_levels = np.zeros(pressure_hpa.shape, dtype=int)

    for i, p in enumerate(pressure_hpa.flat):
        if not np.isfinite(p):
            nearest_levels.flat[i] = -1  # sentinel for invalid pressure
        else:
            distances = np.abs(STANDARD_PRESSURE_LEVELS - p)
            nearest_levels.flat[i] = int(STANDARD_PRESSURE_LEVELS[np.argmin(distances)])

    return nearest_levels.reshape(pressure_hpa.shape)


def build_stratification_key(inst: str, var: str, pressure_level: Optional[int] = None,
                             subtype: Optional[str] = None) -> str:
    """
    Build a fully qualified stratification key for FSOI aggregation.

    Format: `{instrument}/{variable}` or `{instrument}/{variable}/{subtype}`
            or `{instrument}/{variable}/{pressure_level}hPa` (radiosonde)

    Example:
      - 'aircraft/temperature/AIRCAR'
      - 'surface_obs/u_wind/ADPSFC'
      - 'radiosonde/temperature/700hPa'
    """
    key = f"{inst}/{var}"
    if subtype:
        key += f"/{subtype}"
    elif pressure_level is not None and pressure_level > 0:
        key += f"/{pressure_level}hPa"
    return key


def _reduce_weighted_error(
    squared_error: torch.Tensor,
    weights: torch.Tensor,
    inst_weight: float = 1.0,
    loss_reduction: str = 'sum',
) -> torch.Tensor:
    """Reduce weighted squared error with either sum or normalized mean."""
    reduction = _normalize_loss_reduction(loss_reduction)
    if not torch.isfinite(squared_error).all() or not torch.isfinite(weights).all():
        raise RuntimeError("Non-finite forecast-error terms or weights")
    if (weights < 0).any():
        raise ValueError("Negative forecast-error weights")
    error_sum = squared_error.double().sum()
    if reduction == 'mean':
        denom = weights.double().sum()
        if denom <= 0:
            raise SparseTargetError("No valid positively weighted targets")
        return (error_sum / denom) * inst_weight
    return error_sum * inst_weight


def _sampling_record(
    sampling_info: Optional[Dict[str, dict]],
    inst_name: str,
    sampled_n: int,
) -> dict:
    """Return per-instrument sampling metadata used for scaled impact totals."""
    info = (sampling_info or {}).get(inst_name, {}) or {}
    raw_n = int(info.get('raw_n_observations', sampled_n))
    sampled_n_info = int(info.get('sampled_n_observations', sampled_n))
    if sampled_n_info <= 0:
        sampled_n_info = sampled_n
    scale = float(info.get('sample_scale', 1.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(raw_n / sampled_n_info) if sampled_n_info > 0 else 1.0
    return {
        'raw_n_observations': raw_n,
        'sampled_n_observations': sampled_n_info,
        'sample_scale': scale,
        'is_subsampled': bool(info.get('is_subsampled', scale != 1.0)),
        'sampling_design': info.get('sampling_design', 'census' if raw_n == sampled_n_info else 'unrecorded'),
        'sampling_seed': info.get('sampling_seed', np.nan),
    }


def _default_target_channel_names(inst_name: str, n_channels: int) -> dict[int, str]:
    """Default channel→variable names for common conventional targets."""
    inst = (inst_name or '').lower()
    if inst == 'radiosonde':
        # observation_config radiosonde features: [airTemperature, dewPointTemperature, wind_u, wind_v]
        base = {
            0: 'temperature',
            1: 'dewpoint_temperature',
            2: 'u_wind',
            3: 'v_wind',
        }
        return {k: v for k, v in base.items() if k < n_channels}
    if inst == 'aircraft':
        base = {
            0: 'temperature',
            1: 'u_wind',
            2: 'v_wind',
        }
        return {k: v for k, v in base.items() if k < n_channels}
    if inst in ('surface_obs', 'surface', 'synop', 'metar', 'sfcship'):
        base = {
            0: 'surface_pressure',
            1: 'temperature',
            2: 'dewpoint_temperature',
            3: 'u_wind',
            4: 'v_wind',
        }
        return {k: v for k, v in base.items() if k < n_channels}
    # Fallback: keep internal tensor indices zero-based, but expose names as
    # human-facing 1-based channel numbers.
    return {i: f'channel_{i + 1}' for i in range(n_channels)}


def sample_innovation_vs_fsoi(
    fsoi_values: Dict[str, torch.Tensor],
    innovations: Dict[str, torch.Tensor],
    max_points: int = 200000,
    seed: int = 0,
    obs_coords: Optional[Dict[str, Tuple]] = None,
    xa: Optional[Dict[str, torch.Tensor]] = None,
    valid_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> pd.DataFrame:
    """Return a lightweight random sample of (innovation, fsoi) pairs.

    This is used for innovation-vs-FSOI scatter plots without storing full tensors.
    Sample is taken across all instruments/channels available.

    Missing-channel masking strategy
    -------------------------
    Missing satellite channels are zero-imputed during preprocessing, while
    missing conventional channels use the -9.0 sentinel. Therefore the stored
    ``input_channel_mask`` passed through ``valid_masks`` is the preferred
    source of truth. When it is not available, ``xa`` provides the sentinel
    fallback for conventional observations and saved CSV re-plotting can still
    use the older innovation-range fallback.

    Args:
        fsoi_values: Per-instrument FSOI tensors [N_obs, C].
        innovations: Per-instrument innovation tensors [N_obs, C].
        max_points: Maximum total rows in output.
        seed: RNG seed (use pair_idx for reproducibility).
        obs_coords: Optional dict mapping instrument name to (lat_1d, lon_1d)
            numpy arrays of shape [N_obs], already aligned with fsoi_values
            (i.e., subsampling already applied). When provided, 'lat' and 'lon'
            columns are added to the output so scatter samples can be gridded.
        xa: Optional dict of raw analysis input tensors [N_obs, C] from which
            sentinel positions can be determined for conventional inputs.
        valid_masks: Optional dict of per-instrument boolean validity masks
            [N_obs, C], aligned with ``fsoi_values`` and ``innovations``.
    """
    if max_points is None or max_points <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    frames = []

    # Allocate roughly proportional to available points, but enforce a hard cap.
    total_available = 0
    avail = {}
    for inst, f in fsoi_values.items():
        if inst not in innovations:
            continue
        if f is None or innovations[inst] is None:
            continue
        if not torch.is_tensor(f) or not torch.is_tensor(innovations[inst]):
            continue
        if f.shape != innovations[inst].shape:
            continue
        f_np = f.detach().cpu().reshape(-1).numpy()
        inn_np = innovations[inst].detach().cpu().reshape(-1).numpy()

        if valid_masks is not None and inst in valid_masks and valid_masks[inst] is not None:
            mask_np = valid_masks[inst].detach().cpu().reshape(-1).numpy().astype(bool)
            if mask_np.shape != f_np.shape:
                raise RuntimeError(
                    f"{inst}: valid mask size {mask_np.shape} does not match "
                    f"scatter tensor size {f_np.shape}"
                )
            valid_mask = np.isfinite(f_np) & np.isfinite(inn_np) & mask_np
        elif xa is not None and inst in xa and xa[inst] is not None:
            # Conventional fallback: mask on source tensor where sentinel exists.
            xa_np = xa[inst].detach().cpu().reshape(-1).numpy()
            valid_mask = (
                np.isfinite(f_np)
                & np.isfinite(inn_np)
                & ~np.isclose(xa_np, SENTINEL_OBS, rtol=0.0, atol=SENTINEL_OBS_ATOL)
            )
        else:
            # Fallback: range mask on innovation (xa - xb spreads -9.0 by xb).
            valid_mask = (
                np.isfinite(f_np)
                & np.isfinite(inn_np)
                & ~((inn_np >= SENTINEL_INNOVATION_LO) & (inn_np <= SENTINEL_INNOVATION_HI))
            )
        valid_idx = np.flatnonzero(valid_mask)
        n = int(valid_idx.size)
        if n <= 0:
            continue
        avail[inst] = valid_idx
        total_available += n

    if total_available == 0:
        return pd.DataFrame()

    remaining = int(max_points)
    for inst, valid_idx in sorted(avail.items(), key=lambda kv: kv[1].size, reverse=True):
        if remaining <= 0:
            break
        # Proportional allocation with a minimum of 2000 for big instruments
        n = int(valid_idx.size)
        take = int(np.ceil(max_points * (n / total_available)))
        take = int(min(max(take, 2000 if n >= 20000 else 200), remaining, n))

        f = fsoi_values[inst].detach().cpu().reshape(-1)
        inn = innovations[inst].detach().cpu().reshape(-1)

        idx = rng.choice(valid_idx, size=take, replace=False)
        # Recover channel index. Internal tensors are zero-based; report
        # human-facing channels as 1-based in CSV outputs.
        C = int(fsoi_values[inst].shape[1])
        ch = (idx % C).astype(np.int64) + 1
        # Observation index (row) from the flattened sample index
        obs_idx = (idx // C).astype(np.int64)

        row: dict = {
            'instrument': inst,
            'channel': ch,
            'innovation': inn.numpy()[idx],
            'fsoi': f.numpy()[idx],
        }

        # Attach lat/lon when available (obs_coords already subsampled to match fsoi_values)
        if obs_coords and inst in obs_coords:
            lat_arr, lon_arr = obs_coords[inst]
            if lat_arr is not None and len(lat_arr) > 0:
                row['lat'] = lat_arr[obs_idx]
            if lon_arr is not None and len(lon_arr) > 0:
                row['lon'] = lon_arr[obs_idx]

        frames.append(pd.DataFrame(row))
        remaining -= take

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _require_balanced_verification(targets, instrument_weights, channel_weights,
                                   use_area_weights, loss_reduction):
    if not isinstance(targets, list) or len(targets) != 1:
        raise ValueError("Balanced verification requires exactly one named target network")
    if _normalize_loss_reduction(loss_reduction) != 'mean':
        raise ValueError("Balanced verification requires loss_reduction=mean")
    if instrument_weights or channel_weights or use_area_weights:
        raise ValueError("Balanced verification does not accept extra instrument/channel or cosine weights")


def compute_per_level_fsoi_by_variable(
    model,
    curr_batch,
    xa: Dict[str, torch.Tensor],
    xb: Dict[str, torch.Tensor],
    observation_config: dict,
    forecast_lead_step: int,
    instrument_weights: Dict[int, float],
    channel_weights: Dict[int, torch.Tensor],
    use_area_weights: bool = False,
    target_instruments: Optional[List[str]] = None,
    replace_indices: Optional[Dict[str, torch.Tensor]] = None,
    requested_target_variables: Optional[List[str]] = None,
    target_pressure_levels: Optional[List[float]] = None,
    loss_reduction: str = 'mean',
    impact_factor: float = 0.5,
    valid_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Dict]:
    """Compute per-(pressure level, target variable) FSOI and return aggregates.

    This is the closest analogue to the paper's Fig. 5 workflow: define a family
    of forecast-error metrics that select one (variable, pressure) at a time,
    then compute FSOI attribution for all input observations.

    Returns a list of dicts with:
      - p_idx, p_hpa
      - target_channel, target_variable
      - ea_p, eb_p
      - fsoi_values, innovations, gradient_sums
    (Callers can aggregate immediately to CSV and discard tensors if desired.)
    """
    device = model.device
    target_inst = (target_instruments or ['radiosonde'])[0]
    target_nt = f"{target_inst}_target_step{forecast_lead_step}"

    # Build xa/xb batches
    batch_xa = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(batch_xa, target_instruments, forecast_lead_step)
    replace_batch_inputs(batch_xa, xa, observation_config, replace_indices=replace_indices)

    batch_xb = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(batch_xb, target_instruments, forecast_lead_step)
    replace_batch_inputs(batch_xb, xb, observation_config, replace_indices=replace_indices)

    if target_nt not in batch_xa.node_types:
        raise ValueError(f"[PerLevelVar] Target node '{target_nt}' not found in batch")

    if not hasattr(batch_xa[target_nt], 'y') or batch_xa[target_nt].y is None:
        raise ValueError(f"[PerLevelVar] '{target_nt}' has no y targets")

    y_ref = batch_xa[target_nt].y
    if y_ref.dim() != 2:
        raise ValueError(f"[PerLevelVar] Expected y to be [N,C], got {tuple(y_ref.shape)}")
    _require_balanced_verification(
        target_instruments, instrument_weights, channel_weights, use_area_weights, loss_reduction)
    plan = get_target_plan(model, batch_xa[target_nt], target_inst, target_nt,
                           requested_target_variables, target_pressure_levels)
    plan.require_eligible()
    get_target_plan(model, batch_xb[target_nt], target_inst, target_nt,
                    requested_target_variables, target_pressure_levels)
    unique_levels = list(dict.fromkeys(p for p, ch in plan.groups))
    keep_channels = sorted({ch for p, ch in plan.groups})
    ch_name_map = {r['target_channel'] - 1: r['variable'] for r in plan.records}

    # Helper: loss for a single pressure level (or all obs) and single target channel.
    # p_idx=None means no level filtering (used for surface obs without pressure_level).
    def _loss_for(preds, batch, p_idx, ch_idx: int):
        preds_list = preds.get(target_nt) or preds.get(f"{target_inst}_target")
        if preds_list is None or len(preds_list) <= forecast_lead_step:
            raise RuntimeError(f"Missing prediction for {target_nt}")
        y_pred = preds_list[forecast_lead_step]
        if not hasattr(batch[target_nt], 'y') or batch[target_nt].y is None:
            raise RuntimeError(f"Missing target values for {target_nt}")
        y_ref_loc = batch[target_nt].y
        if y_pred.shape != y_ref_loc.shape:
            raise ValueError("Per-group prediction/target shape mismatch")

        if (p_idx, ch_idx) not in plan.groups:
            return None
        return plan.loss(y_pred, y_ref_loc, group=(p_idx, ch_idx))

    xa_list = list(xa.values())
    xb_list = list(xb.values())

    # xa forward
    print("[PerLevelVar] xa forward pass...")
    with torch.enable_grad():
        preds_xa = _unwrap_predictions(model(batch_xa))
    xa_losses = []
    for p_idx in unique_levels:
        for ch in keep_channels:
            loss = _loss_for(preds_xa, batch_xa, p_idx, ch)
            if loss is not None:
                xa_losses.append((p_idx, ch, loss))

    ga_map: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
    ea_map: Dict[tuple[int, int], float] = {}
    for i, (p_idx, ch, loss) in enumerate(xa_losses):
        is_last = (i == len(xa_losses) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xa_list,
            retain_graph=not is_last,
            allow_unused=False,
        )
        ga = {}
        for k, g in zip(xa.keys(), grads):
            if g is None:
                raise RuntimeError(f"Missing target-group gradient for {k}")
            if not torch.isfinite(g).all():
                raise RuntimeError(f"Non-finite target-group gradient for {k}")
            ga[k] = g.detach().cpu()
        ga_map[(p_idx, ch)] = ga
        ea_map[(p_idx, ch)] = float(loss.detach().item())

    # xb forward
    print("[PerLevelVar] xb forward pass...")
    with torch.enable_grad():
        preds_xb = _unwrap_predictions(model(batch_xb))
    xb_losses = []
    for p_idx in unique_levels:
        for ch in keep_channels:
            loss = _loss_for(preds_xb, batch_xb, p_idx, ch)
            if loss is not None:
                xb_losses.append((p_idx, ch, loss))

    gb_map: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
    eb_map: Dict[tuple[int, int], float] = {}
    for i, (p_idx, ch, loss) in enumerate(xb_losses):
        is_last = (i == len(xb_losses) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xb_list,
            retain_graph=not is_last,
            allow_unused=False,
        )
        gb = {}
        for k, g in zip(xb.keys(), grads):
            if g is None:
                raise RuntimeError(f"Missing target-group gradient for {k}")
            if not torch.isfinite(g).all():
                raise RuntimeError(f"Non-finite target-group gradient for {k}")
            gb[k] = g.detach().cpu()
        gb_map[(p_idx, ch)] = gb
        eb_map[(p_idx, ch)] = float(loss.detach().item())

    # Combine to FSOI per observation
    results = []
    for p_idx in unique_levels:
        # Map to hPa for readability; p_idx=None means surface (no pressure level)
        if p_idx is None:
            p_hpa = float('nan')
        else:
            p_hpa = float(STANDARD_PRESSURE_LEVELS[p_idx]) if 0 <= int(p_idx) < len(STANDARD_PRESSURE_LEVELS) else float('nan')
        for ch in keep_channels:
            key = (p_idx, ch)
            if key not in ga_map or key not in gb_map:
                continue
            ga = ga_map[key]
            gb = gb_map[key]

            # Compute on CPU (mirrors compute_per_level_fsoi memory strategy)
            fsoi_values: Dict[str, torch.Tensor] = {}
            innovations: Dict[str, torch.Tensor] = {}
            gradient_sums: Dict[str, torch.Tensor] = {}
            for inst in xa.keys():
                if inst not in ga or inst not in gb or inst not in xb:
                    continue
                xa_cpu = xa[inst].detach().cpu()
                xb_cpu = xb[inst].detach().cpu()
                valid_mask = None
                if valid_masks is not None and inst in valid_masks:
                    valid_mask = valid_masks[inst].detach().cpu()
                dx = xa_cpu - xb_cpu
                gs = ga[inst] + gb[inst]
                if dx.shape != gs.shape:
                    continue
                fsoi_i, innov_i, gsum_i, _ = _masked_fsoi_components(
                    xa_cpu,
                    xb_cpu,
                    gs,
                    impact_factor,
                    valid_mask,
                )
                fsoi_values[inst] = fsoi_i
                innovations[inst] = innov_i
                gradient_sums[inst] = gsum_i

            if not fsoi_values:
                continue
            results.append(
                {
                    'p_idx': None if p_idx is None else int(p_idx),
                    'p_hpa': p_hpa,
                    'target_channel': int(ch) + 1,
                    'target_variable': ch_name_map.get(int(ch), f'channel_{ch + 1}'),
                    'group_weight': plan.coefficient(p_idx, ch),
                    'target_metric_id': plan.metric_id,
                    'ea_p': ea_map.get(key, 0.0),
                    'eb_p': eb_map.get(key, 0.0),
                    'fsoi_values': fsoi_values,
                    'innovations': innovations,
                    'gradient_sums': gradient_sums,
                }
            )

    return results


def _unwrap_predictions(forward_output):
    """Normalize model(batch) output to the predictions dict.

    The GNNLightning forward() may return either:
      - predictions: Dict[str, List[Tensor]]
      - (predictions, mesh_features_per_step)
    """
    if isinstance(forward_output, tuple):
        return forward_output[0]
    return forward_output


def _unwrap_predictions_and_mesh(forward_output):
    """Return (predictions, mesh_features_per_step), padding None if not present."""
    if isinstance(forward_output, tuple) and len(forward_output) == 2:
        return forward_output[0], forward_output[1]
    return forward_output, None


def compute_forecast_error_on_mesh(
    model,
    batch,
    gfs_reference: torch.Tensor,
    mesh_instrument: str,
    forecast_lead_step: int,
    init_time_unix: int,
    use_area_weights: bool = True,
    loss_reduction: str = 'mean',
    return_diagnostics: bool = False,
    enable_gradients: bool = True,
) -> torch.Tensor:
    """Compute forecast error at OCELOT mesh nodes against GFS analysis.

    This is the mesh-space analogue of compute_forecast_error():

        ea_mesh = MSE( OCELOT_mesh_pred(xa),  GFS_analysis_on_mesh )

    Unlike obs-space FSOI, this verification is:
      - Globally uniform (40,962 icosahedral nodes)
      - Independent of any instrument's background field quality
      - Comparable to traditional NWP FSOI which uses gridded state as reference

    The gradient flows: observation inputs → encoder → mesh processor →
    mesh decoder → MSE vs GFS → scalar error.

    Parameters
    ----------
    model            : GNNLightning model (frozen, enable_mesh_pred must be True).
    batch            : HeteroData observation batch.
    gfs_reference    : [N_mesh, C] normalized GFS analysis tensor; NaN channels
                       are automatically excluded from the MSE.
    mesh_instrument  : 'radiosonde' or 'surface_obs'.
    forecast_lead_step : Which processor step to decode (0 = first ~+1.5h step).
    init_time_unix   : Unix timestamp of the window end (for time conditioning).
    use_area_weights : Apply cos(lat) weighting over mesh nodes (recommended).
    loss_reduction   : 'mean' or 'sum'.
    return_diagnostics : If True, return ``(loss, diagnostics)`` where
                       diagnostics contains detached per-node squared errors
                       and mesh coordinates for plotting.
    enable_gradients : Keep gradients through the mesh forecast path. FSOI uses
                       True; OSE diagnostics can use False.

    Returns
    -------
    Scalar differentiable tensor — gradients flow to observation inputs.
    """
    import numpy as np

    device = model.device
    gfs_ref = gfs_reference.to(device)  # [N_mesh, C]

    # ── 1. Forward pass — keep gradients through encoder+processor ──────────
    # The mesh decoder is normally called under no_grad; we bypass that by
    # calling _decode_one_step_to_mesh directly after the forward pass.
    original_enable_mesh = model.enable_mesh_pred

    # Temporarily enable mesh feature collection without the no_grad decoder
    model.enable_mesh_pred = True
    grad_context = torch.enable_grad if enable_gradients else torch.no_grad
    with grad_context():
        fwd_out = model(batch)
    model.enable_mesh_pred = original_enable_mesh

    _, mesh_features_per_step = _unwrap_predictions_and_mesh(fwd_out)

    if mesh_features_per_step is None or len(mesh_features_per_step) == 0:
        print("[MeshFSOI] WARNING: model did not return mesh_features_per_step. "
              "Ensure enable_mesh_pred=True in mesh_config.yaml.")
        zero = torch.tensor(0.0, device=device, requires_grad=enable_gradients)
        return (zero, {}) if return_diagnostics else zero

    if forecast_lead_step >= len(mesh_features_per_step):
        print(f"[MeshFSOI] WARNING: requested lead_step={forecast_lead_step} "
              f"but only {len(mesh_features_per_step)} steps available.")
        forecast_lead_step = len(mesh_features_per_step) - 1

    mesh_feat = mesh_features_per_step[forecast_lead_step]  # [N_mesh, D]

    # ── 2. Decode to mesh — WITH gradients (bypass _decode_all_steps_to_mesh) ─
    mesh_pred_edges = model._get_mesh_pred_edges()
    if mesh_instrument not in mesh_pred_edges:
        raise ValueError(
            f"[MeshFSOI] No mesh prediction edges for '{mesh_instrument}'. "
            f"Available: {list(mesh_pred_edges.keys())}. "
            "Check mesh_config.yaml variables and precompute_mesh_edges.py."
        )

    with grad_context():
        mesh_pred = model._decode_one_step_to_mesh(
            mesh_feat,
            mesh_instrument,
            mesh_pred_edges[mesh_instrument],
            step_idx=forecast_lead_step,
            init_time_unix=init_time_unix,
        )
    # mesh_pred: [N_mesh, C]

    if mesh_pred.shape != gfs_ref.shape:
        raise ValueError(
            f"[MeshFSOI] Shape mismatch: mesh_pred={tuple(mesh_pred.shape)}, "
            f"gfs_reference={tuple(gfs_ref.shape)}. "
            "Ensure GFS tensor was built with the same target_dim as the model."
        )

    # ── 3. NaN-masked MSE against GFS reference ──────────────────────────────
    # Channels with NaN in gfs_ref are excluded (e.g., dewPointTemperature)
    valid_mask = torch.isfinite(gfs_ref)  # [N_mesh, C], True where GFS valid
    n_valid_channels = int(valid_mask.any(dim=0).sum())
    if n_valid_channels == 0:
        print("[MeshFSOI] WARNING: all GFS channels are NaN — no valid channels to score.")
        zero = torch.tensor(0.0, device=device, requires_grad=enable_gradients)
        return (zero, {}) if return_diagnostics else zero

    sq_err_raw = (mesh_pred - gfs_ref) ** 2  # [N_mesh, C]
    _fdtype = sq_err_raw.dtype
    sq_err_unweighted = torch.where(
        valid_mask,
        sq_err_raw,
        torch.full_like(sq_err_raw, float("nan")),
    )
    sq_err = sq_err_raw * valid_mask.to(_fdtype)  # zero out NaN channels

    # ── 4. Area weighting (cosine latitude) ──────────────────────────────────
    edges = mesh_pred_edges[mesh_instrument]
    area_w = None
    if use_area_weights:
        lats = edges.get('lats')
        if lats is not None:
            lat_t = torch.from_numpy(np.asarray(lats)).to(dtype=_fdtype, device=device)  # [N_mesh]
            area_w = torch.cos(torch.deg2rad(lat_t)).abs().clamp(min=1e-6)  # [N_mesh]
            area_w = area_w.view(-1, 1)  # [N_mesh, 1]
            sq_err = sq_err * area_w
            weights_sum = (area_w * valid_mask.to(_fdtype)).sum()
        else:
            weights_sum = valid_mask.to(_fdtype).sum()
    else:
        weights_sum = valid_mask.to(_fdtype).sum()

    if loss_reduction == 'mean':
        loss = sq_err.sum() / weights_sum.clamp(min=1.0)
    else:
        loss = sq_err.sum()

    if return_diagnostics:
        diagnostics = {
            'sq_error': sq_err_unweighted.detach().cpu().to(torch.float32).numpy(),
            'valid_mask': valid_mask.detach().cpu().numpy(),
            'lat': (np.asarray(edges.get('lats'), dtype=np.float32)
                    if edges.get('lats') is not None else None),
            'lon': (np.asarray(edges.get('lons'), dtype=np.float32)
                    if edges.get('lons') is not None else None),
            'area_weight': (area_w.detach().cpu().view(-1).to(torch.float32).numpy()
                            if area_w is not None else None),
            'loss': float(loss.detach().cpu().item()),
            'mesh_instrument': mesh_instrument,
            'forecast_lead_step': int(forecast_lead_step),
        }
        return loss, diagnostics

    return loss


# ==============================================================================
# MEMORY OPTIMIZATION: Target Node Pruning
# ==============================================================================


def prune_batch_targets_inplace(batch, keep_instruments: List[str], lead_step: int):
    """
    Remove target nodes/edges for instruments not in keep_instruments.
    This prevents the model from decoding those targets at all.

    CRITICAL: Call this BEFORE model(batch) to avoid decoding heavy instruments
    like AVHRR (1M+ targets, 4M+ edges).

    Args:
        batch: HeteroData batch to modify in-place
        keep_instruments: List of instrument names to keep (e.g., ["atms", "amsua"])
        lead_step: Forecast lead step (for matching target_step{lead_step} nodes)
    """
    keep_instruments = set(keep_instruments)

    # Target node types we keep
    keep_target_types = {f"{inst}_target_step{lead_step}" for inst in keep_instruments}

    # 1) Remove unwanted target node stores
    removed_nodes = set()
    for nt in list(batch.node_types):
        if "_target_step" in nt and nt not in keep_target_types:
            del batch[nt]  # Use del for HeteroData store access
            removed_nodes.add(nt)

    # 2) Remove unwanted edges to removed targets (and any dangling edges)
    removed_edges = []
    for et in list(batch.edge_types):
        src, rel, dst = et

        # If dst is a target_step and not kept -> delete
        if "_target_step" in dst and dst not in keep_target_types:
            del batch[et]
            removed_edges.append(et)
            continue

        # Safety: if either endpoint node type no longer exists, delete edge
        if (src not in batch.node_types) or (dst not in batch.node_types):
            del batch[et]
            removed_edges.append(et)
            continue

    if removed_nodes:
        print(f"[PRUNE] Removed {len(removed_nodes)} target node types: {list(removed_nodes)[:3]}{'...' if len(removed_nodes) > 3 else ''}")
    if removed_edges:
        print(f"[PRUNE] Removed {len(removed_edges)} decoder edge types")


def subsample_target_nodes_inplace(
    batch,
    inst_name: str,
    step: int,
    max_n: int = 20000,
    seed: int = 42
):
    """
    Subsample target nodes to reduce memory while preserving signal.

    Even ATMS can have 100k+ targets. Subsampling to 20k still gives
    good gradient signal with much lower memory cost.

    Args:
        batch: HeteroData batch to modify in-place
        inst_name: Instrument name (e.g., "atms")
        step: Forecast lead step
        max_n: Maximum number of targets to keep
        seed: Random seed for reproducibility
    """
    nt = f"{inst_name}_target_step{step}"
    if nt not in batch.node_types:
        return

    N = batch[nt].y.shape[0] if hasattr(batch[nt], 'y') else 0
    if N == 0 or N <= max_n:
        return

    # Reproducible random sampling
    torch.manual_seed(seed)
    idx = torch.randperm(N, device=batch[nt].y.device)[:max_n]
    idx_sorted = idx.sort()[0]  # Sort for cache efficiency

    # Subset all node features
    for key in list(batch[nt].keys()):
        val = batch[nt][key]
        if torch.is_tensor(val) and val.shape[0] == N:
            batch[nt][key] = val[idx_sorted]

    # Subset decoder edges mesh->target and remap indices
    et = ("mesh", "to", nt)
    if et in batch.edge_types:
        edge_index = batch[et].edge_index  # [2, E], target idx in row 1
        keep_mask = torch.isin(edge_index[1], idx_sorted)
        ei = edge_index[:, keep_mask]

        # Remap target indices to 0..max_n-1 in the same order as idx_sorted
        # (node features above were stored in idx_sorted order)
        remap = -torch.ones(N, dtype=torch.long, device=idx.device)
        remap[idx_sorted] = torch.arange(idx_sorted.numel(), device=idx.device)
        ei[1] = remap[ei[1]]

        batch[et].edge_index = ei
        if hasattr(batch[et], "edge_attr"):
            batch[et].edge_attr = batch[et].edge_attr[keep_mask]

    print(f"[SUBSAMPLE] {nt}: {N} → {max_n} targets ({100*max_n/N:.1f}%)")


def zero_feature_columns(
    inputs: Dict[str, torch.Tensor],
    observation_config: dict,
    mask_map: Dict[str, List[str]],
) -> None:
    """
    In-place zero-out selected feature columns per instrument.

    Args:
        inputs: Dict of channel tensors [N, C]
        observation_config: Full observation config (provides feature ordering)
        mask_map: {instrument: [feature_name, ...]} to zero
    """
    for inst_name, feature_list in mask_map.items():
        if inst_name not in inputs:
            continue

        # Find feature ordering from config
        cfg_features = None
        for _, instruments in observation_config.items():
            if inst_name in instruments:
                cfg_features = instruments[inst_name].get('features', [])
                break

        if not cfg_features:
            continue

        tensor = inputs[inst_name]
        # Detach first so the clone is a plain leaf (no grad_fn), enabling
        # safe in-place zeroing regardless of whether tensor is a leaf or view.
        req_grad = tensor.requires_grad
        cloned = tensor.detach().clone()
        for feat in feature_list:
            if feat in cfg_features:
                idx = cfg_features.index(feat)
                if idx < cloned.shape[1]:
                    cloned[:, idx] = 0.0
        cloned.requires_grad_(req_grad)
        inputs[inst_name] = cloned


def get_fsoi_inputs(
    batch,
    observation_config: dict,
    instrument_name_to_id: dict,
    match_targets: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Extract observation CHANNELS only (for FSOI attribution).

    CRITICAL: Returns CHANNELS ONLY, not full input with metadata.

    Why:
    - Innovation δx = xa - xb must be in observation-variable space
    - xb predictions are in channel space (model outputs channels, not metadata)
    - Metadata (scan angles, lat/lon encoding) should remain fixed

    Args:
        batch: HeteroData batch from dataloader
        observation_config: Configuration dict with instrument specifications
        instrument_name_to_id: Mapping from instrument names to IDs (unused)
        match_targets: Ignored (kept for compatibility)

    Returns:
        Dict mapping instrument names to observation CHANNEL tensors
        Shape: [N_obs, n_channels] - channels only, no metadata
    """
    fsoi_inputs = {}

    for obs_type, instruments in observation_config.items():
        for inst_name, cfg in instruments.items():
            node_type_input = f"{inst_name}_input"

            if node_type_input not in batch.node_types:
                continue

            x_input = batch[node_type_input].x
            if x_input is None or x_input.numel() == 0:
                continue

            # Get number of observation channels from config
            n_channels = len(cfg.get('features', []))
            if n_channels == 0:
                print(f"[WARNING] {inst_name}: No channels in config, skipping")
                continue

            # x_input layout: [7 geo/time | n_meta instrument-metadata | n_channels obs | optional trailing]
            # Skip the leading geo/time + metadata columns to reach actual observation channels.
            n_meta = len(cfg.get('metadata', []))
            bt_start = 7 + n_meta
            x_channels = x_input[:, bt_start:bt_start + n_channels]

            # Clone, detach, enable gradients
            x_obs = x_channels.clone().detach()
            x_obs.requires_grad_(True)

            if not x_obs.requires_grad:
                raise RuntimeError(f"Failed to enable gradients for {inst_name}")

            fsoi_inputs[inst_name] = x_obs

            print(f"[FSOI Inputs] {inst_name}: extracted {n_channels} channels "
                  f"(shape={x_obs.shape}), requires_grad={x_obs.requires_grad}")

    if not fsoi_inputs:
        print("[WARNING] No FSOI inputs extracted from batch!")

    return fsoi_inputs


def get_fsoi_input_masks(
    batch,
    observation_config: dict,
    replace_indices: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract observation-channel validity masks aligned with FSOI input tensors.

    Satellite inputs use normalized zero imputation for missing channels during
    preprocessing, so their validity must come from ``input_channel_mask`` when
    available. Conventional inputs also carry this mask when persistence inputs
    are requested, with a sentinel-based fallback for older batches.
    """
    masks: Dict[str, torch.Tensor] = {}

    for obs_type, instruments in observation_config.items():
        obs_type_key = str(obs_type).lower()
        for inst_name, cfg in instruments.items():
            node_type_input = f"{inst_name}_input"
            if node_type_input not in batch.node_types:
                continue

            node_data = batch[node_type_input]
            x_input = getattr(node_data, "x", None)
            if x_input is None or x_input.numel() == 0:
                continue

            n_channels = len(cfg.get("features", []))
            if n_channels == 0:
                continue
            n_meta = len(cfg.get("metadata", []))
            bt_start = 7 + n_meta
            x_channels = x_input[:, bt_start:bt_start + n_channels]

            stored_mask = getattr(node_data, "input_channel_mask", None)
            if (
                stored_mask is not None
                and stored_mask.numel() > 0
                and tuple(stored_mask.shape) == tuple(x_channels.shape)
            ):
                mask = stored_mask.detach().clone().to(dtype=torch.bool)
            elif obs_type_key == "satellite":
                print(
                    f"[FSOI Mask WARNING] {inst_name}: input_channel_mask missing; "
                    "using finite-value fallback for zero-imputed satellite inputs"
                )
                mask = torch.isfinite(x_channels).detach().to(torch.bool)
            else:
                mask = observation_valid_mask(x_channels).detach().to(torch.bool)

            idx = (replace_indices or {}).get(inst_name)
            if idx is not None:
                idx = idx.to(device=mask.device, dtype=torch.long)
                mask = mask[idx]

            if device is not None:
                mask = mask.to(device=device)
            masks[inst_name] = mask

    return masks


def get_fsoi_metadata(
    batch,
    observation_config: dict,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract observation metadata (pressure levels, lat/lon, etc.) for FSOI attribution.

    This is used to stratify FSOI results by pressure level for radiosonde and aircraft.

    Args:
        batch: HeteroData batch from dataloader
        observation_config: Configuration dict with instrument specifications

    Returns:
        Dict mapping instrument names to metadata dicts with keys:
        - 'pressure_level': [N_obs] tensor of pressure level indices (0-15) or None
        - 'pressure_hpa': [N_obs] tensor of pressure in hPa or None
        - 'lat': [N_obs] tensor of latitude or None
        - 'lon': [N_obs] tensor of longitude or None
    """
    STANDARD_PRESSURE_LEVELS = np.array([
        1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
    ])

    fsoi_metadata = {}

    for obs_type, instruments in observation_config.items():
        for inst_name, cfg in instruments.items():
            node_type_input = f"{inst_name}_input"

            if node_type_input not in batch.node_types:
                continue

            node_data = batch[node_type_input]

            # Initialize metadata dict
            metadata = {}

            # Extract pressure level if available (for radiosonde and aircraft)
            if hasattr(node_data, 'pressure_level'):
                metadata['pressure_level'] = node_data.pressure_level.detach().cpu()

                # Map indices to actual pressure values if possible
                pressure_idx = node_data.pressure_level.detach().cpu().numpy()
                if pressure_idx.ndim > 1:
                    pressure_idx = pressure_idx.squeeze()

                # Convert indices to hPa values
                pressure_hpa = np.array([
                    STANDARD_PRESSURE_LEVELS[int(idx)] if 0 <= int(idx) < len(STANDARD_PRESSURE_LEVELS) else np.nan
                    for idx in pressure_idx
                ])
                metadata['pressure_hpa'] = torch.from_numpy(pressure_hpa)
            else:
                metadata['pressure_level'] = None
                metadata['pressure_hpa'] = None

            # Extract lat/lon — try direct attributes first (most batch stores
            # keep lat/lon as separate tensors), fall back to a combined
            # .metadata tensor of shape [N, >=2] if present.
            lat_t, lon_t = None, None
            if hasattr(node_data, 'lat') and node_data.lat is not None:
                lat_t = node_data.lat.detach().cpu().float()
                if lat_t.dim() > 1:
                    lat_t = lat_t.squeeze(1)
            if hasattr(node_data, 'lon') and node_data.lon is not None:
                lon_t = node_data.lon.detach().cpu().float()
                if lon_t.dim() > 1:
                    lon_t = lon_t.squeeze(1)
            if lat_t is None and hasattr(node_data, 'metadata'):
                node_metadata = node_data.metadata.detach().cpu()
                if node_metadata.shape[1] >= 2:
                    lat_t = node_metadata[:, 0]
                    lon_t = node_metadata[:, 1]
            metadata['lat'] = lat_t
            metadata['lon'] = lon_t

            fsoi_metadata[inst_name] = metadata

            # Log what we found
            n_obs = node_data.x.shape[0] if node_data.x is not None else 0
            has_pressure = metadata['pressure_level'] is not None
            has_latlon = metadata['lat'] is not None
            print(f"[FSOI Metadata] {inst_name}: {n_obs} obs, pressure={has_pressure}, latlon={has_latlon}")

            if has_pressure:
                pressure_levels_present = torch.unique(metadata['pressure_level']).numpy()
                print(f"  Pressure levels: {pressure_levels_present}")

    return fsoi_metadata


def replace_batch_inputs(
    batch,
    new_inputs: Dict[str, torch.Tensor],
    observation_config: dict,
    replace_indices: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """
    Replace observation CHANNEL values in batch, keeping metadata unchanged.

    For instruments where replace_indices[inst] is provided (a 1-D LongTensor
    of row positions), only those rows are updated; the rest keep the original
    values.  This is used when xa/xb were subsampled: new_inputs[inst] has
    len(idx) rows while the batch tensor has the full observation count.

    Args:
        batch: HeteroData batch (modified in-place)
        new_inputs: Dict mapping instrument names to new CHANNEL tensors
                    Shape: [N_obs or len(idx), n_channels] - channels only
        observation_config: Configuration dict to determine n_channels
        replace_indices: Optional per-instrument row indices for partial
                         replacement; None means full replacement.
    """
    for obs_type, instruments in observation_config.items():
        for inst_name, cfg in instruments.items():
            node_type_input = f"{inst_name}_input"

            if node_type_input not in batch.node_types:
                continue

            if inst_name not in new_inputs:
                continue

            # Get config info
            n_channels = len(cfg.get('features', []))
            if n_channels == 0:
                continue

            # Get original INPUT .x
            x_orig = batch[node_type_input].x
            if x_orig is None or x_orig.numel() == 0:
                continue

            # x_input layout: [7 geo/time | n_meta | n_channels obs | optional trailing]
            n_meta = len(cfg.get('metadata', []))
            bt_start = 7 + n_meta

            # Get new channels (xa or xb)
            new_channels = new_inputs[inst_name]

            if new_channels.shape[1] != n_channels:
                raise ValueError(
                    f"{inst_name}: new_channels has {new_channels.shape[1]} channels "
                    f"but config specifies {n_channels} channels"
                )

            # Split: prefix (geo/time + inst-metadata) | channels | suffix (e.g. sat-id one-hot)
            channels_base = x_orig[:, bt_start:bt_start + n_channels].detach()
            prefix = x_orig[:, :bt_start].detach()
            metadata_full = x_orig[:, bt_start + n_channels:].detach()

            # Determine whether this is a partial (indexed) or full replacement
            idx = None
            if replace_indices is not None and inst_name in replace_indices:
                idx = replace_indices[inst_name]
                if idx is not None:
                    idx = idx.to(x_orig.device).long()

            if idx is None:
                # Full replacement — row counts must match
                if new_channels.shape[0] != channels_base.shape[0]:
                    raise ValueError(
                        f"{inst_name}: new_channels has {new_channels.shape[0]} obs "
                        f"but batch has {channels_base.shape[0]} obs"
                    )
                full_channels = new_channels
                n_replaced = full_channels.shape[0]
            else:
                # Partial (indexed) replacement via differentiable scatter.
                # channels_full[idx] = new_channels via in-place copy would detach
                # the grad path; scatter() is out-of-place and keeps the autograd
                # graph:  loss -> batch.x -> full_channels -> new_channels.
                if new_channels.shape[0] != idx.numel():
                    raise ValueError(
                        f"{inst_name}: new_channels has {new_channels.shape[0]} obs "
                        f"but replace_indices has {idx.numel()} entries"
                    )
                idx_mat = idx.view(-1, 1).expand(-1, n_channels)  # [K, C]
                full_channels = channels_base.scatter(0, idx_mat, new_channels)
                n_replaced = idx.numel()

            batch[node_type_input].x = torch.cat([prefix, full_channels, metadata_full], dim=1)
            print(
                f"[Replace Inputs] {inst_name}: replaced "
                f"{('ALL' if idx is None else n_replaced)} rows; "
                f"shape={batch[node_type_input].x.shape}"
            )


def compute_forecast_error(
    model,
    batch,
    forecast_lead_step: int,
    instrument_weights: Dict[int, float],
    channel_weights: Dict[int, torch.Tensor],
    use_area_weights: bool = False,
    target_instruments: Optional[List[str]] = None,
    target_variables: Optional[List[str]] = None,
    target_pressure_levels: Optional[List[float]] = None,
    loss_reduction: str = 'mean',
) -> torch.Tensor:
    """Balanced normalized MSE for one observation-space verification network.

    Each variable has equal total weight, its configured levels have equal
    weight, and eligible equal-area cell means have equal weight within a group.
    The target_channel_mask and the frozen target plan define scored elements.
    Extra instrument/channel or cosine weights are not supported.
    """
    _require_balanced_verification(
        target_instruments, instrument_weights, channel_weights, use_area_weights, loss_reduction)
    inst = target_instruments[0]
    target_node = f"{inst}_target_step{forecast_lead_step}"
    if target_node not in batch.node_types:
        target_node = f"{inst}_target"
    if target_node not in batch.node_types or getattr(batch[target_node], 'y', None) is None:
        raise SparseTargetError(f"Missing verification targets for {inst}")

    batch = batch.clone()
    prune_batch_targets_inplace(batch, target_instruments, forecast_lead_step)
    plan = get_target_plan(model, batch[target_node], inst, target_node,
                           target_variables, target_pressure_levels)
    plan.require_eligible()
    predictions = _unwrap_predictions(model(batch))
    values = predictions.get(target_node)
    if values is None:
        values = predictions.get(f"{inst}_target")
    if values is None or len(values) <= forecast_lead_step:
        raise RuntimeError(f"Missing prediction for {target_node}, step {forecast_lead_step}")
    return plan.loss(values[forecast_lead_step], batch[target_node].y)


def compute_adjoints(
    error: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    create_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute gradients (adjoints) of forecast error with respect to inputs.

    This computes: g = ∇_x e(x)

    Args:
        error: Scalar forecast error tensor
        inputs: Dict of input tensors (with requires_grad=True)
        create_graph: Whether to keep computation graph (False for FSOI)

    Returns:
        Dict mapping instrument names to gradient tensors
    """
    # Get all input tensors as a list
    input_tensors = list(inputs.values())
    input_names = list(inputs.keys())

    if not input_tensors:
        return {}

    # Check that error requires grad
    if not error.requires_grad:
        raise ValueError("Error tensor must require gradients")

    # Check that inputs require grad
    for name, tensor in inputs.items():
        if not tensor.requires_grad:
            raise ValueError(f"Input tensor '{name}' must require gradients")

    # Compute gradients
    print(f"[Adjoints] Computing gradients for {len(input_tensors)} inputs...")

    gradients = torch.autograd.grad(
        outputs=error,
        inputs=input_tensors,
        create_graph=create_graph,
        retain_graph=False,
        allow_unused=True,
    )

    # Package as dict
    adjoints = {}
    for name, grad in zip(input_names, gradients):
        if grad is not None:
            adjoints[name] = grad
            print(f"[Adjoints] {name}: shape={grad.shape}, mean={grad.abs().mean().item():.6e}, "
                  f"max={grad.abs().max().item():.6e}")
        else:
            print(f"[WARNING] No gradient computed for {name} (unused)")

    return adjoints


def compute_fsoi_per_observation(
    xa: Dict[str, torch.Tensor],
    xb: Dict[str, torch.Tensor],
    ga: Dict[str, torch.Tensor],
    gb: Dict[str, torch.Tensor],
    return_components: bool = False,
    impact_factor: float = 0.5,
    valid_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-observation FSOI using the trapezoidal formula:

    FSOI = impact_factor * delta_x * (ga + gb)

    The standard trapezoidal setting is impact_factor = 0.5:

    FSOI = 0.5 * (xa - xb) * (ga + gb)

    Missing observation channels are assigned zero FSOI and excluded from
    diagnostic means/counts. Pass ``valid_masks`` from ``input_channel_mask``
    when available; this is required for zero-imputed satellite inputs.

    where:
    - delta_x = xa - xb (innovation)
    - ga = gradient of error w.r.t. analysis
    - gb = gradient of error w.r.t. background
    - * is elementwise multiplication

    Args:
        xa: Analysis observation values
        xb: Background observation values
        ga: Analysis adjoints
        gb: Background adjoints

    Returns:
        If return_components is False (default):
            Dict mapping instrument names to FSOI values (same shape as inputs)
        If return_components is True:
            Tuple of (fsoi_values, innovations, gradient_sums) where:
              - fsoi_values: Dict of FSOI tensors
              - innovations: Dict of δx tensors (xa - xb)
              - gradient_sums: Dict of ga + gb tensors
    """
    fsoi_values = {}
    innovations = {}
    gradient_sums = {}

    # Loop over instruments
    all_instruments = set(xa.keys()) | set(xb.keys())

    for inst_name in all_instruments:
        # Check that we have all required components
        if inst_name not in xa:
            print(f"[WARNING] {inst_name} not in analysis inputs")
            continue
        if inst_name not in xb:
            print(f"[WARNING] {inst_name} not in background inputs")
            continue
        if inst_name not in ga:
            print(f"[WARNING] {inst_name} not in analysis gradients")
            continue
        if inst_name not in gb:
            print(f"[WARNING] {inst_name} not in background gradients")
            continue

        # Compute innovation (δx) and adjoint sum
        g_sum = ga[inst_name] + gb[inst_name]

        valid_mask = None
        if valid_masks is not None:
            valid_mask = valid_masks.get(inst_name)

        fsoi, innovation_diag, gsum_diag, valid_obs = _masked_fsoi_components(
            xa[inst_name],
            xb[inst_name],
            g_sum,
            impact_factor,
            valid_mask,
        )

        fsoi_values[inst_name] = fsoi
        innovations[inst_name] = innovation_diag
        gradient_sums[inst_name] = gsum_diag

        # Diagnostics
        impact_sum = fsoi.sum().item()
        valid_values = fsoi[valid_obs]
        impact_mean = valid_values.mean().item() if valid_values.numel() else float("nan")
        positive_frac = (valid_values > 0).float().mean().item() if valid_values.numel() else float("nan")
        missing_count = int((~valid_obs).sum().item())

        print(f"[FSOI] {inst_name}: sum={impact_sum:.6e}, mean={impact_mean:.6e}, "
              f"positive={positive_frac*100:.1f}%, missing_masked={missing_count}")

    if return_components:
        return fsoi_values, innovations, gradient_sums
    return fsoi_values


def compute_per_level_fsoi(
    model,
    curr_batch,
    xa: Dict[str, torch.Tensor],
    xb: Dict[str, torch.Tensor],
    observation_config: dict,
    forecast_lead_step: int,
    instrument_weights: Dict[int, float],
    channel_weights: Dict[int, torch.Tensor],
    use_area_weights: bool = False,
    target_instruments: Optional[List[str]] = None,
    target_variables: Optional[List[str]] = None,
    replace_indices: Optional[Dict[str, torch.Tensor]] = None,
    target_pressure_levels: Optional[List[float]] = None,
    loss_reduction: str = 'mean',
    impact_factor: float = 0.5,
    valid_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Dict]:
    """
    Compute FSOI with a separate loss per radiosonde pressure level.

    Strategy:
    - Run model(xa_batch) ONCE, retain the computation graph.
    - For each pressure level p: compute e_p (MSE at that level) and run
      autograd.grad with retain_graph=True (except the last level).
    - Repeat for model(xb_batch).
    - For each level: δx ⊙ (ga_p + gb_p).

    This gives every instrument (including ATMS/AMSUA/satellites) a gradient
    tagged to the radiosonde target pressure level, filling pressure_hpa in
    fsoi_by_channel.csv for all instruments.

    Returns
    -------
    List of dicts, one per level:
        {p_idx, p_hpa, ea_p, eb_p, fsoi_values, innovations, gradient_sums}
    """
    _HPa = STANDARD_PRESSURE_LEVELS

    device = model.device
    target_inst = (target_instruments or ['radiosonde'])[0]
    target_nt = f"{target_inst}_target_step{forecast_lead_step}"

    # ── Build xa / xb batches ──────────────────────────────────────────────
    # xa and xb have already been aligned to the same row count by the ALIGNMENT
    # block in fsoi_inference (both are subsampled for heavy instruments).
    # replace_indices maps each instrument to the row positions that were decoded,
    # so the indexed rows in the full batch tensor are replaced while the rest
    # are kept as constants (no grad).
    batch_xa = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(batch_xa, target_instruments, forecast_lead_step)
    replace_batch_inputs(batch_xa, xa, observation_config,
                         replace_indices=replace_indices)

    batch_xb = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(batch_xb, target_instruments, forecast_lead_step)
    replace_batch_inputs(batch_xb, xb, observation_config,
                         replace_indices=replace_indices)

    # ── Extract unique pressure levels from target ────────────────────────
    if target_nt not in batch_xa.node_types:
        raise ValueError(f"[PerLevel] Target node '{target_nt}' not found in batch")
    if not hasattr(batch_xa[target_nt], 'pressure_level'):
        raise ValueError(f"[PerLevel] '{target_nt}' has no pressure_level attribute; "
                         "cannot stratify by pressure")

    _require_balanced_verification(
        target_instruments, instrument_weights, channel_weights, use_area_weights, loss_reduction)
    plan = get_target_plan(model, batch_xa[target_nt], target_inst, target_nt,
                           target_variables, target_pressure_levels)
    plan.require_eligible()
    get_target_plan(model, batch_xb[target_nt], target_inst, target_nt,
                    target_variables, target_pressure_levels)
    unique_levels = list(dict.fromkeys(p for p, ch in plan.groups))

    def _level_loss(preds, batch, p_idx: int):
        # The model may key predictions as "radiosonde_target" (no step suffix)
        # or "radiosonde_target_step0".  Try both.
        preds_list = preds.get(target_nt) or preds.get(f"{target_inst}_target")
        if preds_list is None or len(preds_list) <= forecast_lead_step:
            raise RuntimeError(f"Missing prediction for {target_nt}")
        y_pred = preds_list[forecast_lead_step]
        # y_ref and pressure_level come from the batch node (always has _step suffix)
        if not hasattr(batch[target_nt], 'y') or batch[target_nt].y is None:
            raise RuntimeError(f"Missing target values for {target_nt}")
        y_ref = batch[target_nt].y
        if y_pred.shape != y_ref.shape:
            raise ValueError("Pressure-group prediction/target shape mismatch")

        return plan.loss(y_pred, y_ref, level=p_idx)

    xa_list = list(xa.values())
    xa_keys = list(xa.keys())
    ga_per_level = {}
    ea_per_level = {}

    print("[PerLevel] xa forward pass...")
    with torch.enable_grad():
        preds_xa = _unwrap_predictions(model(batch_xa))
    # Log available prediction keys for diagnostics
    print(f"[PerLevel] prediction keys: {list(preds_xa.keys())}")

    # Pre-collect valid (p_idx, loss) pairs so we can use retain_graph=False
    # on the final backward pass, freeing the graph immediately.
    xa_valid_levels = []
    for p_idx in unique_levels:
        loss = _level_loss(preds_xa, batch_xa, p_idx)
        if loss is None:
            print(f"[PerLevel xa] level {p_idx}: no targets")
        else:
            xa_valid_levels.append((p_idx, loss))

    for i, (p_idx, loss) in enumerate(xa_valid_levels):
        is_last = (i == len(xa_valid_levels) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xa_list,
            retain_graph=not is_last,
            allow_unused=False,
        )
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Non-finite pressure-group control gradient")
        valid = sum(1 for g in grads if g is not None)
        # Move immediately to CPU to free GPU memory before next level
        ga_per_level[p_idx] = {n: g.detach().cpu() for n, g in zip(xa_keys, grads) if g is not None}
        ea_per_level[p_idx] = loss.item()
        p_hpa_str = f"{_HPa[p_idx]:.0f}" if 0 <= p_idx < len(_HPa) else "?"
        print(f"[PerLevel xa] level {p_idx} ({p_hpa_str} hPa): ea={loss.item():.4e}, "
              f"non-null grads={valid}/{len(xa_keys)}")

    del preds_xa
    torch.cuda.empty_cache()

    # ── xb: one forward pass, N_levels backward passes ───────────────────
    xb_list = list(xb.values())
    xb_keys = list(xb.keys())
    gb_per_level = {}
    eb_per_level = {}

    print("[PerLevel] xb forward pass...")
    with torch.enable_grad():
        preds_xb = _unwrap_predictions(model(batch_xb))

    xb_valid_levels = []
    for p_idx in unique_levels:
        loss = _level_loss(preds_xb, batch_xb, p_idx)
        if loss is None:
            print(f"[PerLevel xb] level {p_idx}: no targets")
        else:
            xb_valid_levels.append((p_idx, loss))

    for i, (p_idx, loss) in enumerate(xb_valid_levels):
        is_last = (i == len(xb_valid_levels) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xb_list,
            retain_graph=not is_last,
            allow_unused=False,
        )
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Non-finite pressure-group background gradient")
        valid = sum(1 for g in grads if g is not None)
        gb_per_level[p_idx] = {n: g.detach().cpu() for n, g in zip(xb_keys, grads) if g is not None}
        eb_per_level[p_idx] = loss.item()
        p_hpa_str = f"{_HPa[p_idx]:.0f}" if 0 <= p_idx < len(_HPa) else "?"
        print(f"[PerLevel xb] level {p_idx} ({p_hpa_str} hPa): eb={loss.item():.4e}, "
              f"non-null grads={valid}/{len(xb_keys)}")

    del preds_xb
    torch.cuda.empty_cache()

    # ── Compute FSOI per level ────────────────────────────────────────────
    level_results: List[Dict] = []
    for p_idx in unique_levels:
        if p_idx not in ga_per_level or p_idx not in gb_per_level:
            continue

        ga_p = ga_per_level[p_idx]
        gb_p = gb_per_level[p_idx]

        fsoi_p: Dict[str, torch.Tensor] = {}
        innov_p: Dict[str, torch.Tensor] = {}
        gsum_p: Dict[str, torch.Tensor] = {}

        for inst in xa_keys:
            if inst not in ga_p or inst not in gb_p or inst not in xb:
                continue
            ga_inst = ga_p.get(inst)
            gb_inst = gb_p.get(inst)
            if ga_inst is None or gb_inst is None:
                continue

            # xa and xb are already shape-aligned: the caller (fsoi_inference)
            # subsampled xa[inst] to match xb[inst] before calling this function.
            # No second subsampling needed here.
            xa_cpu = xa[inst].detach().cpu()
            xb_cpu = xb[inst].detach().cpu()
            valid_mask = None
            if valid_masks is not None and inst in valid_masks:
                valid_mask = valid_masks[inst].detach().cpu()
            dx = xa_cpu - xb_cpu
            gs = ga_inst + gb_inst   # both are already on CPU (stored via .detach().cpu())

            if dx.shape[0] != gs.shape[0]:
                print(f"[PerLevel WARNING] {inst}: dx {dx.shape} vs gs {gs.shape} "
                      f"shape mismatch, skipping")
                continue

            fsoi_i, innov_i, gsum_i, _ = _masked_fsoi_components(
                xa_cpu,
                xb_cpu,
                gs,
                impact_factor,
                valid_mask,
            )
            fsoi_p[inst] = fsoi_i
            innov_p[inst] = innov_i
            gsum_p[inst] = gsum_i

        if not fsoi_p:
            continue

        p_hpa = float(_HPa[p_idx]) if 0 <= p_idx < len(_HPa) else float('nan')
        insts = list(fsoi_p.keys())
        print(f"[PerLevel] level {p_idx} ({p_hpa:.0f} hPa): FSOI for {insts}")

        level_results.append({
            'p_idx': p_idx,
            'p_hpa': p_hpa,
            'group_weight': plan.coefficient(p_idx),
            'target_metric_id': plan.metric_id,
            'ea_p': ea_per_level.get(p_idx, 0.0),
            'eb_p': eb_per_level.get(p_idx, 0.0),
            'fsoi_values': fsoi_p,
            'innovations': innov_p,
            'gradient_sums': gsum_p,
        })

    print(f"[PerLevel] Done: {len(level_results)} levels with valid FSOI")
    return level_results


def validate_gradients(
    ga: Dict[str, torch.Tensor],
    gb: Dict[str, torch.Tensor],
    require_instruments: Optional[List[str]] = None,
    require_satellite: bool = False,
) -> bool:
    """
    Hard validation: Check that gradients are finite and non-zero.

    For FSOI to be meaningful, we need:
    1. All gradients must be finite (no NaN/Inf)
    2. Gradients should be non-zero for at least some instruments

    Args:
        ga: Analysis adjoints
        gb: Background adjoints
        require_instruments: List of instruments that MUST have valid gradients
                           Default: None (check all, don't require specific ones)
        require_satellite: If True, require at least one satellite instrument
                          Default: False (useful for debugging bins with no satellites)

    Returns:
        True if validation passes

    Raises:
        ValueError if critical gradients are invalid
    """
    print("\n" + "="*80)
    print("GRADIENT VALIDATION - Hard Check")
    print("="*80)

    # Identify satellite instruments
    satellite_instruments = [
        'atms',
        'amsua',
        'amsub',
        'mhs',
        'iasi',
        'cris',
        'airs',
        'ssmis',
        'seviri_asr',
        'seviri_csr',
        'avhrr',
        'ascat',
    ]

    # Default: no required instruments (just check all are valid)
    if require_instruments is None:
        require_instruments = []

    all_instruments = set(ga.keys()) | set(gb.keys())
    valid_satellites = []
    validation_failed = False

    # Check each instrument
    for inst_name in sorted(all_instruments):
        is_satellite = any(sat in inst_name.lower() for sat in satellite_instruments)

        # Check ga
        if inst_name not in ga:
            print(f"[SKIP] {inst_name}: Missing ga (analysis adjoint)")
            if inst_name in require_instruments:
                validation_failed = True
                print(f"  └─> CRITICAL: {inst_name} is required but missing ga!")
            continue

        ga_tensor = ga[inst_name]
        ga_finite = torch.isfinite(ga_tensor).all().item()
        ga_norm = torch.norm(ga_tensor).item()
        ga_nonzero = ga_norm > 1e-12

        # Check gb
        if inst_name not in gb:
            print(f"[SKIP] {inst_name}: Missing gb (background adjoint)")
            if inst_name in require_instruments:
                validation_failed = True
                print(f"  └─> CRITICAL: {inst_name} is required but missing gb!")
            continue

        gb_tensor = gb[inst_name]
        gb_finite = torch.isfinite(gb_tensor).all().item()
        gb_norm = torch.norm(gb_tensor).item()
        gb_nonzero = gb_norm > 1e-12

        # Overall check
        ga_ok = ga_finite and ga_nonzero
        gb_ok = gb_finite and gb_nonzero
        both_ok = ga_ok and gb_ok

        # Report status
        status = "✓ PASS" if both_ok else "✗ FAIL"
        print(f"{status} {inst_name:20s} | ga: finite={ga_finite}, norm={ga_norm:.6e} | "
              f"gb: finite={gb_finite}, norm={gb_norm:.6e}")

        # Track valid satellites
        if is_satellite and both_ok:
            valid_satellites.append(inst_name)

        # Check required instruments
        if inst_name in require_instruments and not both_ok:
            print(f"  └─> CRITICAL: {inst_name} is required but gradients are invalid!")
            validation_failed = True

    # Check satellite requirement (optional)
    if require_satellite:
        if not valid_satellites:
            satellite_in_batch = [i for i in all_instruments if any(s in i.lower() for s in satellite_instruments)]
            print(f"\n✗ FAIL: No satellite instruments have valid gradients!")
            print(f"  Satellites in batch: {satellite_in_batch if satellite_in_batch else 'None'}")
            validation_failed = True
        else:
            print(f"\n✓ PASS: {len(valid_satellites)} satellite(s) have valid gradients: {valid_satellites}")
    else:
        if valid_satellites:
            print(f"\n✓ INFO: {len(valid_satellites)} satellite(s) have valid gradients: {valid_satellites}")
        else:
            print(f"\n⚠ INFO: No satellite instruments in this batch (or all have invalid gradients)")

    # Summary
    n_valid = sum(1 for inst in all_instruments if inst in ga and inst in gb
                  and torch.isfinite(ga[inst]).all() and torch.norm(ga[inst]) > 1e-12
                  and torch.isfinite(gb[inst]).all() and torch.norm(gb[inst]) > 1e-12)

    print(f"\nSummary: {n_valid}/{len(all_instruments)} instruments have valid gradients")

    # Check required instruments
    for req_inst in require_instruments:
        if req_inst in all_instruments:
            if req_inst in ga and req_inst in gb:
                ga_ok = torch.isfinite(ga[req_inst]).all().item() and torch.norm(ga[req_inst]).item() > 1e-12
                gb_ok = torch.isfinite(gb[req_inst]).all().item() and torch.norm(gb[req_inst]).item() > 1e-12
                if ga_ok and gb_ok:
                    print(f"✓ REQUIRED: {req_inst} has valid gradients")
                else:
                    print(f"✗ REQUIRED: {req_inst} has invalid gradients")
            else:
                print(f"✗ REQUIRED: {req_inst} missing from gradients")
        else:
            print(f"⚠ REQUIRED: {req_inst} not in batch")

    print("="*80)

    if validation_failed:
        error_msg = "Gradient validation FAILED!\n"
        if require_instruments:
            error_msg += f"Required instruments with invalid gradients: {require_instruments}\n"
        if require_satellite:
            error_msg += "Required: At least one satellite instrument\n"
        error_msg += (
            "\nThis indicates a problem with the FSOI implementation. Check that:\n"
            "  - xa and xb are extracted from INPUT nodes (observation channels)\n"
            "  - Error metric uses inputs through the model\n"
            "  - requires_grad=True on input tensors\n"
            "  - Model parameters are frozen but graph is retained"
        )
        raise ValueError(error_msg)

    print("✓ All gradient validation checks PASSED\n")
    return True


def aggregate_fsoi_by_channel(
    fsoi_values: Dict[str, torch.Tensor],
    instrument_name_to_id: Dict[str, int],
    metadata: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    innovations: Optional[Dict[str, torch.Tensor]] = None,
    gradient_sums: Optional[Dict[str, torch.Tensor]] = None,
    sampling_info: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    Aggregate FSOI values by instrument and channel, optionally stratified by pressure level.

    Args:
        fsoi_values: Dict mapping instrument names to FSOI tensors [N, C]
        instrument_name_to_id: Mapping from names to IDs
        metadata: Optional dict mapping instrument names to metadata dicts
                 (from get_fsoi_metadata)

    Returns:
        DataFrame with columns: instrument, channel, mean_impact, sum_impact, count
        Plus optional: pressure_level_idx, pressure_hpa if metadata provided
        Additional diagnostics when innovations/gradient_sums are provided:
          - innovation_mean/innovation_std/innovation_abs_mean/innovation_rms
          - gradient_mean/gradient_abs_mean/gradient_rms
          - projection_mean (δx·g)
          - alignment_cosine (cosine between δx and g)
          - alignment_frac (fraction where δx and g have same sign)
    """
    EPS = 1e-12
    records = []

    def _attach_sampling(record, inst, sampled_count, impacts, valid, row_mask=None, channel=None):
        sample = _sampling_record(sampling_info, inst, sampled_count)
        record.update(sample)
        design = dict(sample)
        design.update((sampling_info or {}).get(inst, {}))
        record.update(population_summary(impacts, valid, design, row_mask, channel))
        if 'total_count' in record:
            record['total_count_scaled'] = record['estimated_valid_values_ht']
        return record

    def _attach_stats(record, inst, ch, mask=None):
        """Attach innovation/gradient stats to a record if available."""
        innov = None
        g_sum = None
        if innovations is not None and inst in innovations:
            innov = innovations[inst]
        if gradient_sums is not None and inst in gradient_sums:
            g_sum = gradient_sums[inst]

        if innov is None or g_sum is None:
            record.update({
                'innovation_mean': np.nan,
                'innovation_std': np.nan,
                'innovation_abs_mean': np.nan,
                'innovation_rms': np.nan,
                'gradient_mean': np.nan,
                'gradient_abs_mean': np.nan,
                'gradient_rms': np.nan,
                'projection_mean': np.nan,
                'alignment_cosine': np.nan,
                'alignment_frac': np.nan,
            })
            return record

        # Select channel and optional mask. Missing sentinel channels are NaN
        # in innovations/gradient_sums, so keep only finite values.
        innov_vec = innov[:, ch]
        g_vec = g_sum[:, ch]
        finite_mask = torch.isfinite(innov_vec) & torch.isfinite(g_vec)
        if mask is not None:
            mask = mask.to(device=finite_mask.device, dtype=torch.bool)
            finite_mask = finite_mask & mask
        innov_vec = innov_vec[finite_mask]
        g_vec = g_vec[finite_mask]

        if innov_vec.numel() == 0:
            record.update({
                'innovation_mean': np.nan,
                'innovation_std': np.nan,
                'innovation_abs_mean': np.nan,
                'innovation_rms': np.nan,
                'gradient_mean': np.nan,
                'gradient_abs_mean': np.nan,
                'gradient_rms': np.nan,
                'projection_mean': np.nan,
                'alignment_cosine': np.nan,
                'alignment_frac': np.nan,
            })
            return record

        proj = (innov_vec * g_vec)
        dot = proj.sum()
        norm_innov = torch.norm(innov_vec)
        norm_grad = torch.norm(g_vec)
        cos = dot / (norm_innov * norm_grad + EPS)
        align_frac = (proj > 0).float().mean()

        record.update({
            'innovation_mean': innov_vec.mean().item(),
            'innovation_std': innov_vec.std(unbiased=False).item(),
            'innovation_abs_mean': innov_vec.abs().mean().item(),
            'innovation_rms': torch.sqrt((innov_vec ** 2).mean()).item(),
            'gradient_mean': g_vec.mean().item(),
            'gradient_abs_mean': g_vec.abs().mean().item(),
            'gradient_rms': torch.sqrt((g_vec ** 2).mean()).item(),
            'projection_mean': proj.mean().item(),
            'alignment_cosine': cos.item(),
            'alignment_frac': align_frac.item(),
        })
        return record

    for inst_name, fsoi_tensor in fsoi_values.items():
        # fsoi_tensor is [N, C]
        N, C = fsoi_tensor.shape

        inst_id = instrument_name_to_id.get(inst_name, -1)

        # Get pressure level info if available
        pressure_levels = None
        pressure_hpa_tensor = None
        if metadata is not None and inst_name in metadata:
            if metadata[inst_name].get('pressure_level') is not None:
                pressure_levels = metadata[inst_name]['pressure_level']
                if pressure_levels.numel() != N:
                    print(f"[WARNING] {inst_name}: pressure_level size mismatch ({pressure_levels.numel()} vs {N})")
                    pressure_levels = None
            if metadata[inst_name].get('pressure_hpa') is not None:
                pressure_hpa_tensor = metadata[inst_name]['pressure_hpa']

        # Fallback: use target pressure for instruments without native pressure
        if pressure_levels is None and metadata is not None:
            target_levels = metadata.get('_target_pressure_level')
            target_hpa = metadata.get('_target_pressure_hpa')

            if target_levels is not None:
                if target_levels.numel() == N:
                    pressure_levels = target_levels
                    if target_hpa is not None and target_hpa.numel() == N:
                        pressure_hpa_tensor = target_hpa
                elif target_levels.numel() == 1:
                    pressure_levels = target_levels.repeat(N)
                    if target_hpa is not None and target_hpa.numel() in (1, N):
                        pressure_hpa_tensor = target_hpa if target_hpa.numel() == N else target_hpa.repeat(N)
                else:
                    print(f"[WARNING] {inst_name}: cannot broadcast target pressure (len={target_levels.numel()} vs N={N})")

        valid_by_channel = None
        if innovations is not None and inst_name in innovations:
            innov_tensor = innovations[inst_name]
            if torch.is_tensor(innov_tensor) and innov_tensor.shape == fsoi_tensor.shape:
                valid_by_channel = torch.isfinite(innov_tensor).to(
                    device=fsoi_tensor.device,
                    dtype=torch.bool,
                )

        impact_array = fsoi_tensor.detach().cpu().double().numpy()
        valid_array = (valid_by_channel.detach().cpu().numpy() if valid_by_channel is not None
                       else np.ones(impact_array.shape, dtype=bool))

        # If we have pressure levels, stratify by them
        if pressure_levels is not None:
            # Group by pressure level and channel
            for ch in range(C):
                ch_impacts = fsoi_tensor[:, ch]
                if valid_by_channel is not None:
                    channel_valid = valid_by_channel[:, ch]
                else:
                    channel_valid = torch.ones(N, dtype=torch.bool, device=fsoi_tensor.device)

                # Get unique pressure levels
                unique_levels = torch.unique(pressure_levels)

                for press_level_idx in unique_levels:
                    # Mask for this pressure level
                    pressure_mask = (pressure_levels == press_level_idx)
                    mask = pressure_mask.to(
                        device=fsoi_tensor.device,
                        dtype=torch.bool,
                    )
                    raw_count = int(mask.sum().item())
                    mask = mask & channel_valid
                    # Filter impacts for this pressure level
                    level_impacts = ch_impacts[mask]

                    # Map pressure index to hPa value
                    press_idx_int = int(press_level_idx.item())
                    if pressure_hpa_tensor is not None:
                        press_vals = pressure_hpa_tensor[pressure_mask]
                        press_hpa = float(press_vals.flatten()[0].item()) if press_vals.numel() > 0 else np.nan
                    elif 0 <= press_idx_int < len(STANDARD_PRESSURE_LEVELS):
                        press_hpa = STANDARD_PRESSURE_LEVELS[press_idx_int]
                    else:
                        press_hpa = np.nan  # Invalid/unknown

                    record = {
                        'instrument': inst_name,
                        'instrument_id': inst_id,
                        'channel': ch + 1,
                        'pressure_level_idx': press_idx_int,
                        'pressure_hpa': press_hpa,
                        'mean_impact': level_impacts.mean().item(),
                        'sum_impact': level_impacts.sum().item(),
                        'positive_count': (level_impacts > 0).sum().item(),
                        'negative_count': (level_impacts < 0).sum().item(),
                        'zero_count': (level_impacts == 0).sum().item(),
                        'total_count': mask.sum().item(),
                        'raw_total_count': raw_count,
                        'positive_frac': (level_impacts > 0).float().mean().item(),
                    }

                    record = _attach_sampling(
                        record, inst_name, N, impact_array, valid_array,
                        pressure_mask.detach().cpu().numpy(), ch,
                    )
                    records.append(_attach_stats(record, inst_name, ch, mask))
        else:
            # No pressure stratification - aggregate over all observations
            for ch in range(C):
                if valid_by_channel is not None:
                    channel_valid = valid_by_channel[:, ch]
                else:
                    channel_valid = torch.ones(N, dtype=torch.bool, device=fsoi_tensor.device)
                ch_impacts = fsoi_tensor[:, ch][channel_valid]
                valid_count = int(channel_valid.sum().item())
                record = {
                    'instrument': inst_name,
                    'instrument_id': inst_id,
                    'channel': ch + 1,
                    'mean_impact': ch_impacts.mean().item(),
                    'sum_impact': ch_impacts.sum().item(),
                    'positive_count': (ch_impacts > 0).sum().item(),
                    'negative_count': (ch_impacts < 0).sum().item(),
                    'zero_count': (ch_impacts == 0).sum().item(),
                    'total_count': valid_count,
                    'raw_total_count': N,
                    'positive_frac': (ch_impacts > 0).float().mean().item(),
                }

                record = _attach_sampling(record, inst_name, N, impact_array, valid_array, channel=ch)
                records.append(_attach_stats(record, inst_name, ch, channel_valid))

    return pd.DataFrame(records)


def aggregate_fsoi_by_channel_latitude(
    fsoi_values: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    innovations: Optional[Dict[str, torch.Tensor]] = None,
    sampling_info: Optional[Dict[str, dict]] = None,
    latitude_edges: Tuple[float, ...] = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0),
) -> pd.DataFrame:
    """Aggregate channel impacts by fixed latitude bands.

    This is a diagnostic view of the same impacts, not a replacement for the
    primary global aggregate. Raw and HT-weighted totals remain separate.
    Rows without valid coordinates are excluded from the geographic table.
    """
    edges = np.asarray(latitude_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("latitude_edges must be strictly increasing")
    labels = [f"{edges[i]:g}_to_{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    records = []
    for inst_name, impacts in fsoi_values.items():
        if metadata is None or inst_name not in metadata:
            continue
        lat = metadata[inst_name].get("lat")
        if lat is None:
            continue
        lat = lat.detach().cpu().numpy() if torch.is_tensor(lat) else np.asarray(lat)
        if lat.ndim != 1 or lat.size != impacts.shape[0]:
            continue
        valid = np.isfinite(lat) & (lat >= edges[0]) & (lat <= edges[-1])
        band_idx = np.searchsorted(edges, lat, side="right") - 1
        band_idx = np.clip(band_idx, 0, len(labels) - 1)
        channel_valid = None
        if innovations is not None and inst_name in innovations:
            channel_valid = torch.isfinite(innovations[inst_name]).detach().cpu().numpy()
        design = (sampling_info or {}).get(inst_name, {})
        pi = np.asarray(design.get("inclusion_probability", np.ones(impacts.shape[0])), dtype=float)
        if pi.shape != (impacts.shape[0],):
            pi = np.ones(impacts.shape[0], dtype=float)
        values = impacts.detach().cpu().numpy()
        for ch in range(values.shape[1]):
            eligible = valid.copy()
            if channel_valid is not None and channel_valid.shape == values.shape:
                eligible &= channel_valid[:, ch]
            for band, label in enumerate(labels):
                mask = eligible & (band_idx == band)
                if not mask.any():
                    continue
                band_values = values[mask, ch]
                band_pi = pi[mask]
                records.append({
                    "instrument": inst_name,
                    "channel": ch + 1,
                    "latitude_band": label,
                    "latitude_min": edges[band],
                    "latitude_max": edges[band + 1],
                    "mean_impact": float(np.mean(band_values)),
                    "sum_impact": float(np.sum(band_values, dtype=np.float64)),
                    "sum_impact_ht": float(np.sum(band_values / band_pi, dtype=np.float64)),
                    "count": int(mask.sum()),
                    "positive_frac": float(np.mean(band_values > 0)),
                    "sampling_design": design.get("sampling_design", "unknown"),
                })
    return pd.DataFrame(records)


def aggregate_fsoi_by_grid(
    fsoi_values: Dict[str, torch.Tensor],
    obs_coords: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sampling_info: Optional[Dict[str, dict]] = None,
    grid_deg: float = 5.0,
) -> pd.DataFrame:
    """Sum every sampled row's contribution on a regular latitude-longitude grid.

    ``fsoi_values`` and ``obs_coords`` must be aligned with the sampled rows; a
    mismatch raises instead of dropping the instrument. Missing channels already
    contribute zero. ``fsoi_sum`` is the unexpanded sampled total and
    ``fsoi_sum_ht`` divides each row by its inclusion probability. Cells are
    labelled by their south-west corner.
    """
    frames = []
    for inst, impacts in fsoi_values.items():
        if inst not in obs_coords:
            raise RuntimeError(f"{inst}: no sampled-row coordinates for the grid aggregate")
        lat, lon = (np.asarray(a, dtype=float).reshape(-1) for a in obs_coords[inst])
        values = impacts.detach().cpu().numpy() if torch.is_tensor(impacts) else np.asarray(impacts)
        if values.ndim != 2 or lat.shape[0] != values.shape[0] or lon.shape[0] != values.shape[0]:
            raise RuntimeError(f"{inst}: coordinates {lat.shape} are not aligned with impacts {values.shape}")
        design = (sampling_info or {}).get(inst, {})
        pi = np.asarray(design.get("inclusion_probability", np.ones(values.shape[0])), dtype=float)
        if pi.shape != (values.shape[0],):
            raise RuntimeError(f"{inst}: inclusion probabilities are not aligned with sampled rows")
        row = np.nansum(values, axis=1, dtype=np.float64)
        ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
        ilat = np.minimum(np.floor((lat[ok] + 90) / grid_deg) * grid_deg - 90, 90 - grid_deg)
        ilon = np.floor(np.mod(lon[ok] + 180, 360) / grid_deg) * grid_deg - 180
        frames.append(pd.DataFrame(dict(instrument=inst, ilat=ilat.astype(int), ilon=ilon.astype(int),
                                        fsoi_sum=row[ok], fsoi_sum_ht=row[ok] / pi[ok], n_rows=1))
                      .groupby(["instrument", "ilat", "ilon"], as_index=False).sum())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def level_target_metadata(base_metadata, p_idx, p_hpa) -> dict:
    """Broadcast one target level to instruments without native pressure metadata.

    Surface targets have no level: leaving the keys out avoids torch.tensor([None])
    and keeps satellites aggregated per channel without a bogus target level. Any
    inherited broadcast is dropped first, so one source network's input pressures can
    never be attached to another instrument that happens to have the same row count.
    """
    metadata = {k: v for k, v in (base_metadata or {}).items() if not str(k).startswith('_target_')}
    if p_idx is not None:
        metadata['_target_pressure_level'] = torch.tensor([int(p_idx)])
        metadata['_target_pressure_hpa'] = torch.tensor([float(p_hpa)])
    return metadata


def level_geographic_frame(level_data: dict, metadata: dict,
                           sampling_info: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    """Latitude-band diagnostic for one target group.

    Variable-stratified runs aggregate while the tensors are alive and store the
    frame; other paths still carry ``fsoi_values``. Neither is required.
    """
    if 'fsoi_channel_latitude_aggregates' in level_data:
        return level_data['fsoi_channel_latitude_aggregates'].copy()
    if 'fsoi_values' not in level_data:
        return pd.DataFrame()
    return aggregate_fsoi_by_channel_latitude(
        level_data['fsoi_values'], metadata=metadata,
        innovations=level_data.get('innovations'), sampling_info=sampling_info,
    )


def collapse_target_variable_rows(
    df: pd.DataFrame,
    keys: Tuple[str, ...] = ("pair_idx", "instrument"),
) -> pd.DataFrame:
    """Collapse per-``target_variable`` rows back to one row per ``keys``.

    Variable-stratified target runs (e.g. surface_obs) write one row per
    ``target_variable`` — the *same* observations scored against each target
    metric. Averaging the impact columns over ``target_variable`` reproduces the
    single-metric scale of non-stratified runs (and keeps surface comparable to
    radiosonde/aircraft targets); observation counts, identical across those
    rows, are preserved. Stratification-only columns are dropped. Returns ``df``
    unchanged when no multi-valued ``target_variable`` column is present.
    """
    if "target_variable" not in df.columns or df["target_variable"].nunique(dropna=True) <= 1:
        return df
    key_cols = [k for k in keys if k in df.columns]
    if not key_cols:
        return df
    strat_cols = {
        "target_variable", "target_channel", "p_idx", "p_hpa",
        "ea_p", "eb_p", "ea_total", "eb_total",
    }
    count_cols = {
        "n_observations", "raw_n_observations", "sampled_n_observations",
        "n_channels", "instrument_id", "sample_scale", "is_subsampled",
        "n_valid_values", "n_total_values", "total_count", "raw_total_count",
        "total_count_scaled", "estimated_valid_values_ht", "population_valid_values",
        "sampling_seed",
    }
    agg: Dict[str, str] = {}
    for col in df.columns:
        if col in key_cols or col in strat_cols:
            continue
        if col in count_cols or df[col].dtype == object or df[col].dtype == bool:
            agg[col] = "first"
        else:
            agg[col] = "mean"
    return df.groupby(list(key_cols), dropna=False).agg(agg).reset_index()


def aggregate_fsoi_by_instrument(
    fsoi_values: Dict[str, torch.Tensor],
    instrument_name_to_id: Dict[str, int],
    innovations: Optional[Dict[str, torch.Tensor]] = None,
    gradient_sums: Optional[Dict[str, torch.Tensor]] = None,
    sampling_info: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    Aggregate FSOI values by instrument (sum over all channels).

    Args:
        fsoi_values: Dict mapping instrument names to FSOI tensors [N, C]
        instrument_name_to_id: Mapping from names to IDs

    Returns:
        DataFrame with columns: instrument, mean_impact, sum_impact, count
        Additional diagnostics (if innovations/gradient_sums provided):
          - innovation_mean/innovation_std/innovation_abs_mean/innovation_rms
          - gradient_mean/gradient_abs_mean/gradient_rms
          - projection_mean (δx·g)
          - alignment_cosine, alignment_frac
    """
    EPS = 1e-12
    records = []

    for inst_name, fsoi_tensor in fsoi_values.items():
        inst_id = instrument_name_to_id.get(inst_name, -1)
        n_obs = fsoi_tensor.shape[0]
        n_channels = fsoi_tensor.shape[1]
        sample = _sampling_record(sampling_info, inst_name, n_obs)
        innov = innovations.get(inst_name) if innovations is not None else None
        g_sum = gradient_sums.get(inst_name) if gradient_sums is not None else None

        if innov is not None and innov.shape == fsoi_tensor.shape:
            value_mask = torch.isfinite(innov)
        else:
            value_mask = torch.ones_like(fsoi_tensor, dtype=torch.bool)
        fsoi_valid = fsoi_tensor[value_mask]
        n_valid_values = int(value_mask.sum().item())
        n_total_values = int(fsoi_tensor.numel())

        if n_valid_values:
            total_impact = fsoi_valid.sum().item()
            mean_impact = fsoi_valid.mean().item()
            positive_frac = (fsoi_valid > 0).float().mean().item()
        else:
            total_impact = 0.0
            mean_impact = np.nan
            positive_frac = np.nan

        record = {
            'instrument': inst_name,
            'instrument_id': inst_id,
            'n_observations': n_obs,
            'n_channels': n_channels,
            'n_valid_values': n_valid_values,
            'n_total_values': n_total_values,
            'raw_n_observations': sample['raw_n_observations'],
            'sampled_n_observations': sample['sampled_n_observations'],
            'sample_scale': sample['sample_scale'],
            'is_subsampled': sample['is_subsampled'],
            'mean_impact': mean_impact,
            'sum_impact': total_impact,
            'sum_impact_scaled': total_impact * sample['sample_scale'],
            'positive_frac': positive_frac,
        }
        design = dict(sample)
        design.update((sampling_info or {}).get(inst_name, {}))
        record.update(population_summary(
            fsoi_tensor.detach().cpu().double().numpy(), value_mask.detach().cpu().numpy(), design,
        ))
        record['sampling_design'] = sample['sampling_design']
        record['sampling_seed'] = sample['sampling_seed']

        if innov is not None and g_sum is not None:
            stat_mask = (
                innov.shape == g_sum.shape
                and innov.shape == fsoi_tensor.shape
            )
            if stat_mask:
                finite_mask = value_mask & torch.isfinite(g_sum)
                innov_vec = innov[finite_mask]
                g_vec = g_sum[finite_mask]
            else:
                innov_vec = torch.empty(0, dtype=fsoi_tensor.dtype, device=fsoi_tensor.device)
                g_vec = torch.empty(0, dtype=fsoi_tensor.dtype, device=fsoi_tensor.device)

            if innov_vec.numel() == 0:
                record.update({
                    'innovation_mean': np.nan,
                    'innovation_std': np.nan,
                    'innovation_abs_mean': np.nan,
                    'innovation_rms': np.nan,
                    'gradient_mean': np.nan,
                    'gradient_abs_mean': np.nan,
                    'gradient_rms': np.nan,
                    'projection_mean': np.nan,
                    'alignment_cosine': np.nan,
                    'alignment_frac': np.nan,
                })
                records.append(record)
                continue

            proj = innov_vec * g_vec
            dot = proj.sum()
            norm_innov = torch.norm(innov_vec)
            norm_grad = torch.norm(g_vec)

            record.update({
                'innovation_mean': innov_vec.mean().item(),
                'innovation_std': innov_vec.std(unbiased=False).item(),
                'innovation_abs_mean': innov_vec.abs().mean().item(),
                'innovation_rms': torch.sqrt((innov_vec ** 2).mean()).item(),
                'gradient_mean': g_vec.mean().item(),
                'gradient_abs_mean': g_vec.abs().mean().item(),
                'gradient_rms': torch.sqrt((g_vec ** 2).mean()).item(),
                'projection_mean': proj.mean().item(),
                'alignment_cosine': (dot / (norm_innov * norm_grad + EPS)).item(),
                'alignment_frac': (proj > 0).float().mean().item(),
            })
        else:
            record.update({
                'innovation_mean': np.nan,
                'innovation_std': np.nan,
                'innovation_abs_mean': np.nan,
                'innovation_rms': np.nan,
                'gradient_mean': np.nan,
                'gradient_abs_mean': np.nan,
                'gradient_rms': np.nan,
                'projection_mean': np.nan,
                'alignment_cosine': np.nan,
                'alignment_frac': np.nan,
            })

        records.append(record)

    return pd.DataFrame(records)


def verify_alignment(
    xa: Dict[str, torch.Tensor],
    xb: Dict[str, torch.Tensor],
    batch_curr,
    verbose: bool = True,
    check_spatial: bool = True,
    skip_count_for: set = None,
) -> bool:
    """
    Verify that xa and xb refer to the same observation instances.

    This is critical for FSOI - the analysis and background must be aligned
    (same lat/lon/time/channel).

    Args:
        xa: Analysis observations
        xb: Background observations
        batch_curr: Current batch (for metadata)
        verbose: Print detailed diagnostics
        check_spatial: If True, verify lat/lon arrays match between INPUT nodes and predictions
        skip_count_for: Set of instrument names for which the batch-metadata count check
                        should be skipped.  Use this when xa/xb were intentionally
                        subsampled to fewer rows than the full batch (e.g. AVHRR 1.3M→30k).

    Returns:
        True if alignment is verified, False otherwise
    """
    if skip_count_for is None:
        skip_count_for = set()
    aligned = True
    checked_spatial = False

    for inst_name in xa.keys():
        if inst_name not in xb:
            if verbose:
                print(f"[ALIGNMENT ERROR] {inst_name} in xa but not in xb")
            aligned = False
            continue

        xa_vals = xa[inst_name]
        xb_vals = xb[inst_name]

        # For subsampled instruments xa is full-size while xb is a smaller subset.
        # Shape mismatch is expected by design — skip all checks for these.
        if inst_name in skip_count_for:
            if verbose:
                print(f"[ALIGNMENT OK (subsampled)] {inst_name}: "
                      f"xa={xa_vals.shape[0]} (full), xb={xb_vals.shape[0]} (subsampled)")
            continue

        # Shape check (only for non-subsampled instruments)
        if xa_vals.shape != xb_vals.shape:
            if verbose:
                print(f"[ALIGNMENT ERROR] {inst_name} shape mismatch: "
                      f"xa={xa_vals.shape}, xb={xb_vals.shape}")
            aligned = False
            continue

        # Check for metadata if available
        node_type = f"{inst_name}_input"
        if node_type in batch_curr.node_types:
            node_data = batch_curr[node_type]

            if hasattr(node_data, 'lat') and hasattr(node_data, 'lon'):
                # Verify we have correct number of observations.
                # Skip this check for instruments that were intentionally subsampled
                # (xa and xb have fewer rows than the full batch by design).
                n_obs = node_data.lat.shape[0]
                if inst_name not in skip_count_for and xa_vals.shape[0] != n_obs:
                    if verbose:
                        print(f"[ALIGNMENT ERROR] {inst_name} observation count mismatch: "
                              f"xa={xa_vals.shape[0]}, metadata={n_obs}")
                    aligned = False
                    continue
                elif inst_name in skip_count_for and verbose:
                    print(f"[ALIGNMENT OK (subsampled)] {inst_name}: "
                          f"xa={xa_vals.shape[0]} (subsampled from {n_obs})")

                # STRICTER CHECK: Verify lat/lon arrays used for predictions match INPUT metadata
                if check_spatial and not checked_spatial:
                    lat_input = node_data.lat
                    lon_input = node_data.lon

                    # Compute checksums for verification
                    lat_mean = lat_input.mean().item()
                    lon_mean = lon_input.mean().item()
                    lat_first5 = lat_input[:min(5, len(lat_input))].cpu().numpy()
                    lon_first5 = lon_input[:min(5, len(lon_input))].cpu().numpy()

                    if verbose:
                        print(f"\n[ALIGNMENT SPATIAL CHECK] {inst_name}:")
                        print(f"  INPUT lat: mean={lat_mean:.4f}, first_5={lat_first5}")
                        print(f"  INPUT lon: mean={lon_mean:.4f}, first_5={lon_first5}")
                        print(f"  NOTE: xb predictions should use these EXACT locations")
                        print(f"        (Verify in predict_at_targets() that pseudo-targets use curr_batch INPUT lat/lon)")

                    checked_spatial = True  # Only check once per alignment call

        if verbose and aligned:
            print(f"[ALIGNMENT OK] {inst_name}: shape={xa_vals.shape}")

    return aligned


def verify_gradients(
    adjoints: Dict[str, torch.Tensor],
    verbose: bool = True,
) -> bool:
    """
    Verify that computed gradients are valid (not None, not NaN, not all zeros).

    Args:
        adjoints: Dict of gradient tensors
        verbose: Print diagnostics

    Returns:
        True if all gradients are valid, False otherwise
    """
    valid = True

    for inst_name, grad in adjoints.items():
        if grad is None:
            if verbose:
                print(f"[GRADIENT ERROR] {inst_name}: gradient is None")
            valid = False
            continue

        if not torch.isfinite(grad).all():
            if verbose:
                n_nan = (~torch.isfinite(grad)).sum().item()
                print(f"[GRADIENT ERROR] {inst_name}: {n_nan} non-finite values")
            valid = False
            continue

        if (grad.abs() < 1e-20).all():
            if verbose:
                print(f"[GRADIENT WARNING] {inst_name}: all gradients near zero")
            # Not necessarily an error, but worth noting

        if verbose and valid:
            print(f"[GRADIENT OK] {inst_name}: mean={grad.abs().mean().item():.6e}, "
                  f"max={grad.abs().max().item():.6e}")

    return valid
