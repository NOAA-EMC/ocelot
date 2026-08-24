"""
FSOI OSE (Observing System Experiment) module.

Scientific purpose
------------------
FSOI is derived from a tangent-linear approximation of the forecast-error change.
An OSE directly measures the same change without any approximation. For matched
validation, define the OSE change with the FSOI-compatible sign convention:

    J_control      = J(xa full)
    J_denied       = J(xa with X denied)
    delta_J_actual = J_control - J_denied
    FSOI_X         ~= sum_k 0.5 * (x_control,k - x_denied,k) * (gc_k + gd_k)

Positive delta_J_actual means the control error was larger, so instrument X was
detrimental for this verification target. In the linear limit:

    FSOI_X ~= delta_J_actual

The legacy OSE column keeps the opposite sign:

    ose_impact = J_denied - J_control = -delta_J_actual

Therefore FSOI and legacy ose_impact should have opposite signs under these
definitions. Compare FSOI directly with delta_J_actual or with the derived
ose_fsoi_convention column, not with raw ose_impact.

Disagreement between them reveals where the GNN's nonlinearities break the
tangent-linear assumption used by FSOI.  A Pearson correlation r > 0.90 and a
slope close to 1 on the scatter plot indicate FSOI rankings are reliable.

Design
------
We reuse the already-computed xa and xb from the FSOI pipeline. The matched
obs-space validation uses two gradient-enabled endpoint passes, one at xa and
one at the denied endpoint. Those same two losses provide both FSOI and the
realized OSE error change. No retraining or additional background computation is
needed.

The default "denied" perturbation is background replacement:
    xa_ose[inst][valid]   = xb[inst][valid]   for denied instruments
    xa_ose[inst][missing] = xa[inst][missing] for sentinel-filled cells
    xa_ose[k]             = xa[k]             for other instruments

For true input-denial tests, the denied endpoint can instead mask observation
channels to the missing-value sentinel. ``sample_mask`` masks the same sampled
rows used in the matched FSOI calculation. ``full_mask`` masks every row for the
denied instrument in the batch; this is closest to a whole observing-system
input denial, but the sensitivity is along the path from xa to the missing-input
sentinel rather than the physical xa-xb analysis-background increment.

This is the OCELOT-appropriate single-cycle OSE.  In a cycling NWP context the
background would also degrade over time; here we measure the single-cycle impact.
The background-replacement mode matches the standard innovation FSOI path, while
mask-denial modes measure sensitivity to removing the input signal.

Usage
-----
Integrated into fsoi_inference.py via --ose_instruments flag.
Can also be run stand-alone via compute_ose_for_pair() if xa/xb are available.
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

ABSOLUTE_SIGNAL_FLOOR = 1e-12
REPRO_SIGNAL_MULTIPLIER = 10.0
NORMALIZED_MEAN_LOSS_REDUCTIONS = {
    "mean",
    "mse",
    "normalized",
    "average",
    "avg",
}
OSE_DENIAL_MODES = {
    "background_replacement",
    "sample_mask",
    "full_mask",
}
DENIAL_MODE_DESCRIPTIONS = {
    "background_replacement": "valid xa values are replaced by xb on the matched sampled rows",
    "sample_mask": "valid sampled rows are masked to the missing-observation sentinel",
    "full_mask": "all rows for the denied instrument are masked to the missing-observation sentinel",
}

# Matched OSE/FSOI validation uses:
#   delta_j_actual = J_control - J_denied
#   matched_fsoi   = 0.5 * dx^T * (grad J_control + grad J_denied)
# with positive values meaning the denied instrument was detrimental.
# The legacy ose_impact column keeps the opposite sign for backward
# compatibility: ose_impact = J_denied - J_control.


def _signal_threshold_from_repro(
    observed_control_reproducibility_error: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Return a numerical signal threshold and its reproducibility basis."""
    repro_error = float("nan")
    threshold = ABSOLUTE_SIGNAL_FLOOR
    basis = "absolute_floor_no_reproducibility_error_available"

    if observed_control_reproducibility_error is not None:
        try:
            candidate = abs(float(observed_control_reproducibility_error))
        except (TypeError, ValueError):
            candidate = float("nan")
        if np.isfinite(candidate):
            repro_error = candidate
            threshold = max(ABSOLUTE_SIGNAL_FLOOR, REPRO_SIGNAL_MULTIPLIER * candidate)
            basis = "max(1e-12, 10*control_reproducibility_error)"

    return threshold, repro_error, basis


def _observed_repro_error_from_frame(df: pd.DataFrame) -> Optional[float]:
    """Extract the largest finite observed control reproducibility error."""
    priority_cols = ("matched_control_reproducibility_error",)
    for col in priority_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values[np.isfinite(values)]
        if not finite.empty:
            return float(finite.abs().max())

    candidates = []
    for col in (
        "observed_control_reproducibility_error",
        "control_reproducibility_error",
        "repro_ea_diff",
        "ea_repro_diff",
        "reproducibility_ea_diff",
    ):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values[np.isfinite(values)]
        if not finite.empty:
            candidates.append(float(finite.abs().max()))
    return max(candidates) if candidates else None


def _serialize_provenance_value(value) -> str:
    """Serialize simple config provenance fields for CSV output."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return str(value)


def _normalize_denial_mode(denial_mode: Optional[str]) -> str:
    """Validate and normalize the OSE denied-endpoint construction."""
    mode = str(denial_mode or "background_replacement").strip().lower().replace("-", "_")
    aliases = {
        "background": "background_replacement",
        "replace": "background_replacement",
        "replacement": "background_replacement",
        "increment_denial": "background_replacement",
        "mask": "sample_mask",
        "masked": "sample_mask",
        "true_denial": "full_mask",
        "full": "full_mask",
    }
    mode = aliases.get(mode, mode)
    if mode not in OSE_DENIAL_MODES:
        raise ValueError(
            f"Unknown OSE denial mode {denial_mode!r}. "
            f"Expected one of {sorted(OSE_DENIAL_MODES)}."
        )
    return mode


def _input_channel_bounds(observation_config: dict, inst_name: str) -> Tuple[int, int]:
    """Return start/end column indices for observation channels in input .x."""
    for instruments in observation_config.values():
        if inst_name not in instruments:
            continue
        cfg = instruments[inst_name] or {}
        n_channels = len(cfg.get("features", []))
        if n_channels <= 0:
            raise ValueError(f"{inst_name}: no configured observation features")
        n_meta = len(cfg.get("metadata", []))
        start = 7 + n_meta
        return start, start + n_channels
    raise KeyError(f"{inst_name} not found in observation_config")


def _batch_input_channels(curr_batch, observation_config: dict, inst_name: str, device) -> torch.Tensor:
    """Extract full current input observation-channel tensor for one instrument."""
    node_type = f"{inst_name}_input"
    if node_type not in curr_batch.node_types:
        raise KeyError(f"{node_type} not present in batch")
    x_orig = getattr(curr_batch[node_type], "x", None)
    if x_orig is None or x_orig.numel() == 0:
        raise ValueError(f"{node_type}.x is missing or empty")
    start, end = _input_channel_bounds(observation_config, inst_name)
    if end > x_orig.shape[1]:
        raise ValueError(
            f"{inst_name}: configured channel slice {start}:{end} exceeds "
            f"input width {x_orig.shape[1]}"
        )
    return x_orig[:, start:end].detach().clone().to(device)


def _make_denied_channels(
    control_tensor: torch.Tensor,
    background_tensor: Optional[torch.Tensor],
    denial_mode: str,
) -> torch.Tensor:
    """Construct denied endpoint channels for one instrument."""
    if denial_mode == "background_replacement":
        if background_tensor is None:
            raise ValueError("background_replacement requires xb for the denied instrument")
        from fsoi_utils import observation_valid_mask

        valid_obs = observation_valid_mask(control_tensor)
        return torch.where(valid_obs, background_tensor, control_tensor)

    from fsoi_utils import SENTINEL_OBS

    return torch.full_like(control_tensor, float(SENTINEL_OBS))


def _finite_signal(a: float, b: float, signal_threshold: float) -> bool:
    """Return True only when both values are finite and above the noise floor."""
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(a) > signal_threshold and abs(b) > signal_threshold


def _finite_sign_agree(
    a: float,
    b: float,
    signal_threshold: float = ABSOLUTE_SIGNAL_FLOOR,
) -> bool:
    """Compare signs only when both values have useful signal."""
    if not _finite_signal(a, b, signal_threshold):
        return False
    return np.sign(a) == np.sign(b)


def _mesh_channel_names(mesh_instrument: str, n_channels: int) -> np.ndarray:
    """Human-readable channel labels for saved mesh OSE fields."""
    try:
        from fsoi_utils import _default_target_channel_names
        mapping = _default_target_channel_names(mesh_instrument, n_channels)
    except Exception:
        mapping = {i: f"channel_{i + 1}" for i in range(n_channels)}
    return np.asarray([mapping.get(i, f"channel_{i + 1}") for i in range(n_channels)])


def _save_ose_spatial_fields(
    output_dir,
    control_diag: dict,
    denied_diag: dict,
    pair_idx: int,
    prev_bin: str,
    curr_bin: str,
    lead_step: int,
    denied_instruments: List[str],
    mesh_instrument: str,
    mesh_pressure_level_idx: Optional[int],
    ea_control: float,
    ea_denied: float,
    ose_impact: float,
) -> str:
    """Save per-node mesh OSE error-difference fields for physical case studies."""
    control_sq = np.asarray(control_diag.get("sq_error"), dtype=np.float32)
    denied_sq = np.asarray(denied_diag.get("sq_error"), dtype=np.float32)
    if control_sq.shape != denied_sq.shape or control_sq.size == 0:
        raise ValueError(
            f"Control/denied spatial error shape mismatch: "
            f"{control_sq.shape} vs {denied_sq.shape}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pressure_hpa = np.nan
    pressure_tag = "plevNA"
    if mesh_pressure_level_idx is not None:
        pressure_tag = f"pidx{int(mesh_pressure_level_idx):02d}"
        try:
            from fsoi_utils import STANDARD_PRESSURE_LEVELS
            pressure_hpa = float(STANDARD_PRESSURE_LEVELS[int(mesh_pressure_level_idx)])
            pressure_tag = f"{int(pressure_hpa)}hPa"
        except Exception:
            pass

    denied_tag = "_".join(sorted(denied_instruments)) or "unknown"
    safe_denied = "".join(c if c.isalnum() or c in "_-" else "_" for c in denied_tag)
    out_file = out_dir / (
        f"ose_spatial_pair{pair_idx:04d}_{safe_denied}_"
        f"{mesh_instrument}_{pressure_tag}.npz"
    )

    valid_mask = control_diag.get("valid_mask")
    if valid_mask is None:
        valid_mask = np.isfinite(control_sq) & np.isfinite(denied_sq)
    lat = control_diag.get("lat")
    lon = control_diag.get("lon")

    np.savez_compressed(
        out_file,
        lat=np.asarray(lat, dtype=np.float32) if lat is not None else np.asarray([]),
        lon=np.asarray(lon, dtype=np.float32) if lon is not None else np.asarray([]),
        control_sq_error=control_sq,
        denied_sq_error=denied_sq,
        error_diff=(control_sq - denied_sq).astype(np.float32),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        area_weight=np.asarray(control_diag.get("area_weight"), dtype=np.float32)
                    if control_diag.get("area_weight") is not None else np.asarray([]),
        channel_names=_mesh_channel_names(mesh_instrument, control_sq.shape[1]),
        pair_idx=np.asarray(pair_idx),
        prev_bin=np.asarray(prev_bin),
        curr_bin=np.asarray(curr_bin),
        lead_step=np.asarray(lead_step),
        denied_instruments=np.asarray(",".join(sorted(denied_instruments))),
        mesh_instrument=np.asarray(mesh_instrument),
        mesh_pressure_level_idx=np.asarray(
            -1 if mesh_pressure_level_idx is None else int(mesh_pressure_level_idx)
        ),
        mesh_pressure_hpa=np.asarray(pressure_hpa, dtype=np.float32),
        ea_control=np.asarray(ea_control, dtype=np.float64),
        ea_denied=np.asarray(ea_denied, dtype=np.float64),
        ose_impact=np.asarray(ose_impact, dtype=np.float64),
        error_diff_convention=np.asarray(
            "full/control squared error minus denied squared error; positive means denial improves locally"
        ),
    )
    print(f"[OSE] Saved spatial error-difference fields: {out_file}")
    return str(out_file)


def compute_ose_for_pair(
    model,
    curr_batch,
    xa: Dict[str, torch.Tensor],
    xb: Dict[str, torch.Tensor],
    denied_instruments: List[str],
    ea_control: float,
    observation_config: dict,
    subsample_indices: Dict[str, Optional[torch.Tensor]],
    target_instruments: Optional[List[str]],
    target_variables: Optional[List[str]],
    target_pressure_levels: Optional[List[float]],
    instrument_weights: dict,
    channel_weights: dict,
    use_area_weights: bool,
    loss_reduction: str,
    forecast_lead_step: int,
    pair_idx: int,
    curr_bin: str,
    prev_bin: str,
    gfs_reference: Optional[torch.Tensor] = None,
    mesh_instrument: str = "radiosonde",
    mesh_pressure_level_idx: Optional[int] = None,
    init_time_unix: Optional[int] = None,
    spatial_output_dir: Optional[str] = None,
    denial_mode: str = "background_replacement",
) -> dict:
    """Compute single-cycle OSE impact for one time pair.

    Parameters
    ----------
    xa, xb : already-aligned observation dicts from the FSOI pipeline.
    ea_control : forecast error with full xa passed in by caller.  This value
        is logged but NOT used for ose_impact — ea_control is always recomputed
        here with the same single compute_forecast_error call used for ea_denied,
        guaranteeing consistent scale regardless of whether the caller used a
        stratified (sum-of-strata) or unstratified error.
    denied_instruments : list of instrument names to withhold.
    denial_mode : how to construct the denied endpoint:
        background_replacement, sample_mask, or full_mask.

    Returns
    -------
    dict with per-(pair, instrument) OSE impact and comparison metadata.
    """
    from fsoi_utils import (
        replace_batch_inputs,
        compute_forecast_error,
        compute_forecast_error_on_mesh,
    )
    from fsoi_utils import prune_batch_targets_inplace

    if model.training:
        print("[OSE] WARNING: model was in training mode; switching to eval()")
    model.eval()

    device = next(model.parameters()).device
    denial_mode = _normalize_denial_mode(denial_mode)

    # Check which denied instruments are available for the requested endpoint.
    present_denied = []
    missing_denied = []
    for inst in denied_instruments:
        if denial_mode == "full_mask":
            if f"{inst}_input" in curr_batch.node_types:
                present_denied.append(inst)
            else:
                missing_denied.append(inst)
        elif denial_mode == "sample_mask":
            if inst in xa:
                present_denied.append(inst)
            else:
                missing_denied.append(inst)
        elif inst in xa and inst in xb:
            present_denied.append(inst)
        else:
            missing_denied.append(inst)
    if missing_denied:
        print(f"[OSE] WARNING: {missing_denied} unavailable for denial mode {denial_mode} - skipping")
    if not present_denied:
        print(f"[OSE] No denied instruments present - skipping pair {pair_idx}")
        return {}

    shared_kwargs = dict(
        forecast_lead_step=forecast_lead_step,
        instrument_weights=instrument_weights,
        channel_weights=channel_weights,
        use_area_weights=use_area_weights,
        target_instruments=target_instruments,
        target_variables=target_variables,
        target_pressure_levels=target_pressure_levels,
        loss_reduction=loss_reduction,
    )
    use_mesh_ose = gfs_reference is not None and init_time_unix is not None
    save_spatial = bool(spatial_output_dir) and use_mesh_ose

    def _compute_error(batch_for_error, return_spatial: bool = False):
        if use_mesh_ose:
            out = compute_forecast_error_on_mesh(
                model=model,
                batch=batch_for_error,
                gfs_reference=gfs_reference,
                mesh_instrument=mesh_instrument,
                forecast_lead_step=forecast_lead_step,
                init_time_unix=init_time_unix,
                use_area_weights=use_area_weights,
                loss_reduction=loss_reduction,
                return_diagnostics=return_spatial,
                enable_gradients=False,
            )
            if return_spatial:
                loss, diag = out
                return float(loss.item()), diag
            return float(out.item()), None

        return float(compute_forecast_error(
            model, batch_for_error, **shared_kwargs).item()), None

    # ── Control run: ea with full xa (recomputed for scale consistency) ──────
    # The caller may pass an ea_control derived from a stratified sum (e.g. sum
    # of 64 per-level mean losses), which is a different scale than the single
    # compute_forecast_error call used for ea_denied.  Always recompute here so
    # both numbers come from identical aggregation.
    def _make_ose_inputs(denied: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        inputs = {}
        replace_idx = dict(subsample_indices or {})

        for inst, tensor in xa.items():
            if inst in present_denied and denial_mode == "full_mask":
                control_tensor = _batch_input_channels(curr_batch, observation_config, inst, device)
                replace_idx[inst] = None
            else:
                control_tensor = tensor.detach().clone().to(device)

            if denied and inst in present_denied:
                background_tensor = xb[inst].detach().clone().to(device) if inst in xb else None
                inputs[inst] = _make_denied_channels(control_tensor, background_tensor, denial_mode)
            else:
                inputs[inst] = control_tensor

        for inst in present_denied:
            if inst in inputs:
                continue
            control_tensor = _batch_input_channels(curr_batch, observation_config, inst, device)
            replace_idx[inst] = None
            inputs[inst] = (
                _make_denied_channels(control_tensor, None, denial_mode)
                if denied else control_tensor
            )

        return inputs, replace_idx

    control_inputs, control_replace_idx = _make_ose_inputs(denied=False)
    curr_batch_ctrl = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(curr_batch_ctrl, target_instruments, forecast_lead_step)
    replace_batch_inputs(curr_batch_ctrl, control_inputs, observation_config,
                         replace_indices=control_replace_idx)
    with torch.no_grad():
        ea_control_fresh, control_diag = _compute_error(
            curr_batch_ctrl,
            return_spatial=save_spatial,
        )
    torch.cuda.empty_cache()

    if abs(ea_control) > 1e-12:
        ratio = ea_control_fresh / ea_control
        if ratio > 2.0 or ratio < 0.5:
            print(f"[OSE] NOTE: caller ea_control={ea_control:.4e}, "
                  f"recomputed={ea_control_fresh:.4e} (ratio={ratio:.2f}). "
                  f"Using recomputed value (caller used stratified aggregation).")

    # ── Denied run: ea with xa[denied] replaced by xb[denied] ───────────────
    denied_inputs, denied_replace_idx = _make_ose_inputs(denied=True)
    curr_batch_ose = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(curr_batch_ose, target_instruments, forecast_lead_step)
    replace_batch_inputs(curr_batch_ose, denied_inputs, observation_config,
                         replace_indices=denied_replace_idx)
    with torch.no_grad():
        ea_denied, denied_diag = _compute_error(
            curr_batch_ose,
            return_spatial=save_spatial,
        )
    torch.cuda.empty_cache()

    # OSE impact: negative means denied instrument was detrimental (its removal
    # reduced error).  Positive means it was beneficial (removal increased error).
    # Sign convention: ose_impact < 0 = detrimental (matches FSOI > 0 = detrimental
    # after negation — see compare_ose_vs_fsoi).
    ose_impact = ea_denied - ea_control_fresh

    spatial_npz = ""
    if save_spatial and control_diag and denied_diag:
        try:
            spatial_npz = _save_ose_spatial_fields(
                output_dir=spatial_output_dir,
                control_diag=control_diag,
                denied_diag=denied_diag,
                pair_idx=pair_idx,
                prev_bin=prev_bin,
                curr_bin=curr_bin,
                lead_step=forecast_lead_step,
                denied_instruments=present_denied,
                mesh_instrument=mesh_instrument,
                mesh_pressure_level_idx=mesh_pressure_level_idx,
                ea_control=ea_control_fresh,
                ea_denied=ea_denied,
                ose_impact=ose_impact,
            )
        except Exception as save_err:
            print(f"[OSE] WARNING: spatial field save failed for pair {pair_idx}: {save_err}")

    return {
        'pair_idx': pair_idx,
        'prev_bin': prev_bin,
        'curr_bin': curr_bin,
        'lead_step': forecast_lead_step,
        'denied_instruments': ','.join(sorted(present_denied)),
        'ose_denial_mode': denial_mode,
        'ose_denial_description': DENIAL_MODE_DESCRIPTIONS[denial_mode],
        'ea_control': ea_control_fresh,
        'ea_denied': ea_denied,
        'ose_impact': ose_impact,
        'ose_sign': 'helpful' if ose_impact > 0 else 'detrimental',
        'ose_relative_impact': ose_impact / (abs(ea_control_fresh) + 1e-12),
        'verification_target': 'mesh' if use_mesh_ose else 'obs',
        'mesh_instrument': mesh_instrument if use_mesh_ose else '',
        'mesh_pressure_level_idx': mesh_pressure_level_idx if use_mesh_ose else '',
        'ose_spatial_npz': spatial_npz,
        'loss_reduction': str(loss_reduction),
        'target_instruments': _serialize_provenance_value(target_instruments),
        'target_variables': _serialize_provenance_value(target_variables),
        'target_pressure_levels': _serialize_provenance_value(target_pressure_levels),
        'use_area_weights': bool(use_area_weights),
    }


def compute_matched_conditional_fsoi_for_pair(
    model,
    curr_batch,
    xa: Dict[str, torch.Tensor],
    xb: Dict[str, torch.Tensor],
    denied_instruments: List[str],
    observation_config: dict,
    subsample_indices: Dict[str, Optional[torch.Tensor]],
    target_instruments: Optional[List[str]],
    target_variables: Optional[List[str]],
    target_pressure_levels: Optional[List[float]],
    instrument_weights: dict,
    channel_weights: dict,
    use_area_weights: bool,
    loss_reduction: str,
    forecast_lead_step: int,
    pair_idx: int,
    curr_bin: str,
    prev_bin: str,
    impact_factor: float = 0.5,
    run_control_repro_check: bool = False,
    control_reproducibility_error: Optional[float] = None,
    denial_mode: str = "background_replacement",
) -> dict:
    """Compute apples-to-apples conditional FSOI for an OSE denial.

    This validation uses one combined forecast-error metric J, not the
    per-variable/per-pressure stratified losses used for channel diagnostics.
    For background_replacement and sample_mask it compares the same sampled
    denied rows on both sides:

        x_control = xa
        x_denied  = xa with denied-instrument cells replaced by xb or masked

    For full_mask, all current-batch rows for the denied instrument are masked
    to the missing-observation sentinel. This is a stronger input-denial
    experiment, but the path is xa to missing-input sentinel rather than the
    physical xa-xb innovation path. Sentinel-filled missing channels contribute
    zero because they are unchanged.

        I_matched = 0.5 * (x_control - x_denied)^T
                    [grad J(x_control) + grad J(x_denied)]

        delta_j_actual = J(x_control) - J(x_denied)

    Positive values mean the denied instrument was detrimental, because the
    control error is larger than the denied error. No population scaling is
    applied to either side.
    """
    from fsoi_utils import (
        replace_batch_inputs,
        compute_forecast_error,
        prune_batch_targets_inplace,
    )

    if model.training:
        print("[OSE Matched] WARNING: model was in training mode; switching to eval()")
    model.eval()

    device = next(model.parameters()).device
    denial_mode = _normalize_denial_mode(denial_mode)
    present_denied = []
    missing_denied = []
    for inst in denied_instruments:
        if denial_mode == "full_mask":
            if f"{inst}_input" in curr_batch.node_types:
                present_denied.append(inst)
            else:
                missing_denied.append(inst)
        elif denial_mode == "sample_mask":
            if inst in xa:
                present_denied.append(inst)
            else:
                missing_denied.append(inst)
        elif inst in xa and inst in xb:
            present_denied.append(inst)
        else:
            missing_denied.append(inst)
    if missing_denied:
        print(
            f"[OSE Matched] WARNING: {missing_denied} unavailable for "
            f"denial mode {denial_mode} - skipping"
        )
    if not present_denied:
        print(f"[OSE Matched] No denied instruments present for pair {pair_idx}")
        return {}
    if not np.isclose(impact_factor, 0.5):
        raise ValueError("Matched endpoint FSOI requires impact_factor=0.5")
    loss_reduction_key = str(loss_reduction).strip().lower()
    if loss_reduction_key not in NORMALIZED_MEAN_LOSS_REDUCTIONS:
        raise ValueError("Matched observation-space OSE validation requires a normalized-mean J")

    shared_kwargs = dict(
        forecast_lead_step=forecast_lead_step,
        instrument_weights=instrument_weights,
        channel_weights=channel_weights,
        use_area_weights=use_area_weights,
        target_instruments=target_instruments,
        target_variables=target_variables,
        target_pressure_levels=target_pressure_levels,
        loss_reduction=loss_reduction,
    )

    def _make_inputs(denied: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        inputs = {}
        replace_idx = dict(subsample_indices or {})
        for inst, tensor in xa.items():
            if inst in present_denied:
                if denial_mode == "full_mask":
                    control_tensor = _batch_input_channels(curr_batch, observation_config, inst, device)
                    replace_idx[inst] = None
                else:
                    control_tensor = tensor.detach().clone().to(device)
                if denied:
                    background_tensor = xb[inst].detach().clone().to(device) if inst in xb else None
                    src = _make_denied_channels(control_tensor, background_tensor, denial_mode)
                else:
                    src = control_tensor
                inputs[inst] = src.requires_grad_(True)
            else:
                inputs[inst] = tensor.detach().clone().to(device)
        for inst in present_denied:
            if inst in inputs:
                continue
            control_tensor = _batch_input_channels(curr_batch, observation_config, inst, device)
            replace_idx[inst] = None
            src = (
                _make_denied_channels(control_tensor, None, denial_mode)
                if denied else control_tensor
            )
            inputs[inst] = src.requires_grad_(True)
        return inputs, replace_idx

    def _loss_and_grads(
        inputs: Dict[str, torch.Tensor],
        replace_idx: Dict[str, Optional[torch.Tensor]],
    ):
        batch_for_error = curr_batch.clone()
        if target_instruments is not None:
            prune_batch_targets_inplace(batch_for_error, target_instruments, forecast_lead_step)
        replace_batch_inputs(batch_for_error, inputs, observation_config, replace_indices=replace_idx)
        loss = compute_forecast_error(model, batch_for_error, **shared_kwargs)
        grad_inputs = [inputs[inst] for inst in present_denied]
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=grad_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        grad_map = {}
        for inst, grad in zip(present_denied, grads):
            if grad is None:
                raise RuntimeError(f"Missing matched FSOI gradient for {inst}")
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"Non-finite matched FSOI gradient for {inst}")
            grad_map[inst] = grad
        return loss, grad_map

    def _loss_value_no_grad(
        inputs: Dict[str, torch.Tensor],
        replace_idx: Dict[str, Optional[torch.Tensor]],
    ) -> float:
        batch_for_error = curr_batch.clone()
        if target_instruments is not None:
            prune_batch_targets_inplace(batch_for_error, target_instruments, forecast_lead_step)
        replace_batch_inputs(batch_for_error, inputs, observation_config, replace_indices=replace_idx)
        with torch.no_grad():
            loss = compute_forecast_error(model, batch_for_error, **shared_kwargs)
        return float(loss.detach().item())

    control_inputs, control_replace_idx = _make_inputs(denied=False)
    denied_inputs, denied_replace_idx = _make_inputs(denied=True)

    j_control, grad_control = _loss_and_grads(control_inputs, control_replace_idx)
    torch.cuda.empty_cache()
    j_denied, grad_denied = _loss_and_grads(denied_inputs, denied_replace_idx)
    torch.cuda.empty_cache()

    matched_by_instrument = {}
    sampled_rows = {}
    raw_rows = {}
    sample_scales = {}

    for inst in present_denied:
        g_c = grad_control.get(inst)
        g_d = grad_denied.get(inst)
        if g_c is None or g_d is None:
            raise RuntimeError(f"Missing matched FSOI gradient for {inst}")

        dx = control_inputs[inst].detach() - denied_inputs[inst].detach()
        if dx.shape != g_c.shape or dx.shape != g_d.shape:
            raise RuntimeError(
                f"[OSE Matched] WARNING: Shape mismatch for {inst}: "
                f"dx={tuple(dx.shape)}, g_control={tuple(g_c.shape)}, "
                f"g_denied={tuple(g_d.shape)}"
            )

        matched_by_instrument[inst] = float((0.5 * dx * (g_c + g_d)).sum().item())
        sampled_rows[inst] = int(dx.shape[0])

        node_type = f"{inst}_input"
        raw_n = sampled_rows[inst]
        if node_type in curr_batch.node_types and getattr(curr_batch[node_type], "x", None) is not None:
            raw_n = int(curr_batch[node_type].x.shape[0])
        raw_rows[inst] = raw_n
        sample_scales[inst] = float(raw_n / sampled_rows[inst]) if sampled_rows[inst] > 0 else 1.0

    matched_fsoi = float(sum(matched_by_instrument.values()))
    j_control_value = float(j_control.detach().item())
    j_denied_value = float(j_denied.detach().item())
    delta_j_actual = float(j_control_value - j_denied_value)
    ose_impact = float(j_denied_value - j_control_value)
    j_control_repeat_value = float("nan")
    matched_control_repro_error = float("nan")
    matched_control_repro_source = "none"
    if control_reproducibility_error is not None:
        try:
            candidate_repro_error = abs(float(control_reproducibility_error))
        except (TypeError, ValueError):
            candidate_repro_error = float("nan")
        if np.isfinite(candidate_repro_error):
            matched_control_repro_error = candidate_repro_error
            matched_control_repro_source = "representative_pair"
    if run_control_repro_check:
        j_control_repeat_value = _loss_value_no_grad(control_inputs, control_replace_idx)
        matched_control_repro_error = abs(j_control_repeat_value - j_control_value)
        matched_control_repro_source = "this_pair_repeat"
    signal_threshold, repro_error, threshold_basis = _signal_threshold_from_repro(
        matched_control_repro_error if np.isfinite(matched_control_repro_error) else None
    )
    signal_valid = _finite_signal(matched_fsoi, delta_j_actual, signal_threshold)
    closure_ratio = (
        matched_fsoi / delta_j_actual
        if signal_valid else float("nan")
    )

    return {
        'pair_idx': pair_idx,
        'prev_bin': prev_bin,
        'curr_bin': curr_bin,
        'lead_step': forecast_lead_step,
        'denied_instruments': ','.join(sorted(present_denied)),
        'ose_denial_mode': denial_mode,
        'ose_denial_description': DENIAL_MODE_DESCRIPTIONS[denial_mode],
        'ea_control': j_control_value,
        'ea_denied': j_denied_value,
        'ose_impact': ose_impact,
        'ose_sign': 'helpful' if ose_impact > 0 else 'detrimental',
        'ose_relative_impact': ose_impact / (abs(j_control_value) + 1e-12),
        'verification_target': 'obs',
        'mesh_instrument': '',
        'mesh_pressure_level_idx': '',
        'ose_spatial_npz': '',
        'loss_reduction': str(loss_reduction),
        'target_instruments': _serialize_provenance_value(target_instruments),
        'target_variables': _serialize_provenance_value(target_variables),
        'target_pressure_levels': _serialize_provenance_value(target_pressure_levels),
        'use_area_weights': bool(use_area_weights),
        'matched_comparison_mode': {
            'background_replacement': 'conditional_endpoint_same_sample_same_J',
            'sample_mask': 'conditional_endpoint_sample_mask_same_J',
            'full_mask': 'conditional_endpoint_full_mask_same_J',
        }[denial_mode],
        'matched_sign_convention': 'positive=detrimental; delta_j_actual=J_control-J_denied',
        'matched_fsoi': matched_fsoi,
        'matched_fsoi_by_instrument': ';'.join(
            f"{inst}:{matched_by_instrument[inst]:.17g}"
            for inst in sorted(matched_by_instrument)
        ),
        'delta_j_actual': delta_j_actual,
        'j_control': j_control_value,
        'j_denied': j_denied_value,
        'matched_control_repeated': bool(run_control_repro_check),
        'matched_control_repeat': j_control_repeat_value,
        'matched_control_reproducibility_error': matched_control_repro_error,
        'matched_control_reproducibility_source': matched_control_repro_source,
        'matched_closure_ratio': closure_ratio,
        'matched_signal_threshold': signal_threshold,
        'matched_signal_threshold_basis': threshold_basis,
        'matched_observed_control_reproducibility_error': repro_error,
        'matched_signal_valid': signal_valid,
        'matched_sign_agree': _finite_sign_agree(
            matched_fsoi,
            delta_j_actual,
            signal_threshold,
        ),
        'matched_population_scaled': False,
        'matched_sampled_rows': ';'.join(
            f"{inst}:{sampled_rows[inst]}" for inst in sorted(sampled_rows)
        ),
        'matched_raw_rows': ';'.join(
            f"{inst}:{raw_rows[inst]}" for inst in sorted(raw_rows)
        ),
        'matched_sample_scale': ';'.join(
            f"{inst}:{sample_scales[inst]:.8g}" for inst in sorted(sample_scales)
        ),
    }


def compare_ose_vs_fsoi(
    ose_csv: "Path",
    fsoi_inst_csv: "Path",
) -> pd.DataFrame:
    """Merge OSE results with FSOI predictions per denied instrument.

    When matched endpoint columns are present in ``ose_results.csv``, this
    function uses them directly:

      delta_j_actual = J_control - J_denied
      fsoi_predicted = matched_fsoi

    Both use the convention positive = detrimental and neither side is
    population-scaled. Older outputs without matched endpoint columns now fail
    instead of falling back to the legacy aggregate instrument CSV.
    """
    from pathlib import Path

    if not Path(ose_csv).is_file():
        print(f"[OSE Compare] {ose_csv} not found")
        return pd.DataFrame()

    ose = pd.read_csv(ose_csv)

    matched_cols = {'matched_fsoi', 'delta_j_actual'}
    missing = matched_cols.difference(ose.columns)
    if missing:
        raise ValueError(
            "Final OSE/FSOI validation requires matched endpoint columns in "
            f"{ose_csv}. Missing: {sorted(missing)}. Rerun OSE with the "
            "matched conditional endpoint code; the legacy stratified/"
            "population-scaled comparison is disabled."
        )
    signal_threshold, repro_error, threshold_basis = _signal_threshold_from_repro(
        _observed_repro_error_from_frame(ose)
    )
    if matched_cols.issubset(ose.columns) and ose['matched_fsoi'].notna().any():
        ose = ose.copy()
        ose['matched_fsoi'] = pd.to_numeric(ose['matched_fsoi'], errors='coerce')
        ose['delta_j_actual'] = pd.to_numeric(ose['delta_j_actual'], errors='coerce')
        rows = []
        for _, row in ose.iterrows():
            if not np.isfinite(row.get('matched_fsoi', np.nan)):
                raise ValueError(f"Non-finite matched_fsoi in {ose_csv}, pair_idx={row.get('pair_idx')}")
            if not np.isfinite(row.get('delta_j_actual', np.nan)):
                raise ValueError(f"Non-finite delta_j_actual in {ose_csv}, pair_idx={row.get('pair_idx')}")
            denied = str(row.get('denied_instruments', '')).strip()
            instruments = [i.strip() for i in denied.split(',') if i.strip()]
            denied_label = instruments[0] if len(instruments) == 1 else denied
            rec = row.to_dict()
            rec['denied_instrument'] = denied_label
            rec['instrument'] = denied_label
            rec['fsoi_predicted'] = row.get('matched_fsoi')
            rec['ose_fsoi_convention'] = row.get('delta_j_actual')
            rec['comparison_mode'] = row.get(
                'matched_comparison_mode',
                'conditional_endpoint_same_sample_same_J',
            )
            rows.append(rec)

        merged = pd.DataFrame(rows)
        if merged.empty:
            return merged
        merged['fsoi_predicted'] = pd.to_numeric(merged['fsoi_predicted'], errors='coerce')
        merged['ose_fsoi_convention'] = pd.to_numeric(merged['ose_fsoi_convention'], errors='coerce')
        signal_valid = (
            merged['fsoi_predicted'].abs().gt(signal_threshold) &
            merged['ose_fsoi_convention'].abs().gt(signal_threshold)
        )
        sign_agree = (
            np.sign(merged['fsoi_predicted']) ==
            np.sign(merged['ose_fsoi_convention'])
        )
        merged['signal_threshold'] = signal_threshold
        merged['signal_threshold_basis'] = threshold_basis
        merged['observed_control_reproducibility_error'] = repro_error
        merged['signal_valid'] = signal_valid
        merged['near_zero_excluded'] = ~signal_valid
        merged['closure_ratio'] = np.where(
            signal_valid,
            merged['fsoi_predicted'] / merged['ose_fsoi_convention'],
            np.nan,
        )
        merged['abs_magnitude_ratio'] = np.where(
            signal_valid,
            merged['fsoi_predicted'].abs() / merged['ose_fsoi_convention'].abs(),
            np.nan,
        )
        merged['sign_agree'] = np.where(signal_valid, sign_agree, np.nan)
        merged['n_total_cycles'] = len(merged)
        merged['n_signal_valid_cycles'] = int(signal_valid.sum())
        merged['n_near_zero_excluded_cycles'] = int((~signal_valid).sum())
        return merged

    raise ValueError(
        "Final OSE/FSOI validation requires finite matched_fsoi and "
        "delta_j_actual values. The legacy stratified/population-scaled "
        "comparison is disabled."
    )
