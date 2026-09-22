"""
FSOI OSE (Observing System Experiment) module.

Scientific purpose
------------------
FSOI is derived from a tangent-linear approximation of the forecast-error change.
An OSE directly measures the same change without any approximation:

    OSE_X  =  ea( xa with X replaced by xb )  -  ea( xa full )
    FSOI_X ≈  sum_k  0.5 * (xa_k - xb_k) * (ga_k + gb_k)

In the linear limit:  OSE_X ≈ FSOI_X

Disagreement between them reveals where the GNN's nonlinearities break the
tangent-linear assumption used by FSOI.  A Pearson correlation r > 0.90 and a
slope close to 1 on the scatter plot indicate FSOI rankings are reliable.

Design
------
We reuse the already-computed xa and xb from the FSOI pipeline so that the OSE
adds only ONE extra no-grad forward pass per (pair, denied instrument) beyond the
FSOI cost.  No retraining or additional background computation is needed.

The "denied" perturbation is:
    xa_ose[inst] = xb[inst]   for inst in denied_instruments
    xa_ose[k]    = xa[k]      for k not in denied_instruments

This is the OCELOT-appropriate single-cycle OSE.  In a cycling NWP context the
background would also degrade over time; here we measure the single-cycle impact,
which is the same quantity FSOI estimates.

Usage
-----
Integrated into fsoi_inference.py via --ose_instruments flag.
Can also be run stand-alone via compute_ose_for_pair() if xa/xb are available.
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


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
    denied_instruments : list of instrument names to withhold (replace xa with xb).

    Returns
    -------
    dict with per-(pair, instrument) OSE impact and comparison metadata.
    """
    from fsoi_utils import replace_batch_inputs, compute_forecast_error
    from fsoi_utils import prune_batch_targets_inplace

    device = next(model.parameters()).device

    # Check which denied instruments are actually present in xa
    present_denied = [i for i in denied_instruments if i in xa and i in xb]
    missing_denied = [i for i in denied_instruments if i not in xa or i not in xb]
    if missing_denied:
        print(f"[OSE] WARNING: {missing_denied} not in xa/xb — skipping those in denial")
    if not present_denied:
        print(f"[OSE] No denied instruments present in xa — skipping pair {pair_idx}")
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

    # ── Control run: ea with full xa (recomputed for scale consistency) ──────
    # The caller may pass an ea_control derived from a stratified sum (e.g. sum
    # of 64 per-level mean losses), which is a different scale than the single
    # compute_forecast_error call used for ea_denied.  Always recompute here so
    # both numbers come from identical aggregation.
    curr_batch_ctrl = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(curr_batch_ctrl, target_instruments, forecast_lead_step)
    replace_batch_inputs(curr_batch_ctrl, xa, observation_config,
                         replace_indices=subsample_indices)
    with torch.no_grad():
        ea_control_fresh = float(compute_forecast_error(
            model, curr_batch_ctrl, **shared_kwargs).item())
    torch.cuda.empty_cache()

    if abs(ea_control) > 1e-12:
        ratio = ea_control_fresh / ea_control
        if ratio > 2.0 or ratio < 0.5:
            print(f"[OSE] NOTE: caller ea_control={ea_control:.4e}, "
                  f"recomputed={ea_control_fresh:.4e} (ratio={ratio:.2f}). "
                  f"Using recomputed value (caller used stratified aggregation).")

    # ── Denied run: ea with xa[denied] replaced by xb[denied] ───────────────
    xa_ose = {}
    for inst, tensor in xa.items():
        if inst in present_denied and inst in xb:
            xa_ose[inst] = xb[inst].detach().clone()
        else:
            xa_ose[inst] = tensor.detach().clone()

    curr_batch_ose = curr_batch.clone()
    if target_instruments is not None:
        prune_batch_targets_inplace(curr_batch_ose, target_instruments, forecast_lead_step)
    replace_batch_inputs(curr_batch_ose, xa_ose, observation_config,
                         replace_indices=subsample_indices)
    with torch.no_grad():
        ea_denied = float(compute_forecast_error(
            model, curr_batch_ose, **shared_kwargs).item())
    torch.cuda.empty_cache()

    # OSE impact: negative means denied instrument was detrimental (its removal
    # reduced error).  Positive means it was beneficial (removal increased error).
    # Sign convention: ose_impact < 0 = detrimental (matches FSOI > 0 = detrimental
    # after negation — see compare_ose_vs_fsoi).
    ose_impact = ea_denied - ea_control_fresh

    return {
        'pair_idx': pair_idx,
        'prev_bin': prev_bin,
        'curr_bin': curr_bin,
        'lead_step': forecast_lead_step,
        'denied_instruments': ','.join(sorted(present_denied)),
        'ea_control': ea_control_fresh,
        'ea_denied': ea_denied,
        'ose_impact': ose_impact,
        'ose_sign': 'helpful' if ose_impact > 0 else 'detrimental',
        'ose_relative_impact': ose_impact / (abs(ea_control_fresh) + 1e-12),
    }


def compare_ose_vs_fsoi(
    ose_csv: "Path",
    fsoi_inst_csv: "Path",
) -> pd.DataFrame:
    """Merge OSE results with FSOI predictions per (pair, denied instrument).

    For each (pair_idx, denied_instrument), matches:
      OSE:   ose_impact  = ea_denied - ea_control
      FSOI:  fsoi_predicted = sum_impact_scaled  (for that instrument and pair)

    Returns merged DataFrame with closure_ratio = fsoi_predicted / ose_impact.
    A ratio close to 1.0 means FSOI accurately predicts the actual impact.
    """
    from pathlib import Path

    if not Path(ose_csv).is_file():
        print(f"[OSE Compare] {ose_csv} not found")
        return pd.DataFrame()
    if not Path(fsoi_inst_csv).is_file():
        print(f"[OSE Compare] {fsoi_inst_csv} not found")
        return pd.DataFrame()

    ose = pd.read_csv(ose_csv)
    fsoi = pd.read_csv(fsoi_inst_csv)

    impact_col = 'sum_impact_scaled' if 'sum_impact_scaled' in fsoi.columns else 'sum_impact'

    # Aggregate FSOI per (pair_idx, instrument) — sum across levels/variables
    fsoi_agg = (
        fsoi.groupby(['pair_idx', 'instrument'])[impact_col]
        .sum()
        .reset_index()
        .rename(columns={impact_col: 'fsoi_predicted'})
    )

    # Explode multi-instrument denial rows in OSE
    rows = []
    for _, row in ose.iterrows():
        for inst in str(row['denied_instruments']).split(','):
            inst = inst.strip()
            rows.append({**row.to_dict(), 'denied_instrument': inst})
    ose_long = pd.DataFrame(rows)

    if ose_long.empty or fsoi_agg.empty:
        return pd.DataFrame()

    merged = ose_long.merge(
        fsoi_agg,
        left_on=['pair_idx', 'denied_instrument'],
        right_on=['pair_idx', 'instrument'],
        how='left',
    )

    # Sign conventions differ between FSOI and OSE:
    #   FSOI > 0  → instrument is detrimental (increases forecast error)
    #   ose_impact = ea_denied - ea_control
    #     ose_impact < 0 → denying instrument helped → instrument was detrimental
    #     ose_impact > 0 → denying instrument hurt  → instrument was beneficial
    # So FSOI and ose_impact have OPPOSITE signs for the same physical conclusion.
    # Negate ose_impact to put both in the "positive = detrimental" convention
    # before computing sign agreement and closure ratio.
    eps = 1e-12
    ose_fsoi_convention = -merged['ose_impact']   # now positive = detrimental
    merged['closure_ratio'] = np.where(
        ose_fsoi_convention.abs() > eps,
        merged['fsoi_predicted'] / ose_fsoi_convention,
        np.nan,
    )
    merged['sign_agree'] = np.sign(merged['fsoi_predicted']) == np.sign(ose_fsoi_convention)

    return merged
