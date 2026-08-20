"""
FSOI Inference - Main script for computing Forecast Sensitivity to Observations Impact.

This script:
1. Loads a trained GNN model
2. Creates a sequential FSOI dataloader
3. For each (prev, curr) pair:
   - Computes analysis adjoint (ga)
   - Computes background adjoint (gb)
   - Computes FSOI = δx ⊙ (ga + gb)
4. Aggregates and saves results

Usage:
    python FSOI/fsoi_inference.py --checkpoint path/to/model.ckpt --config FSOI/configs/fsoi_config.yaml
    python FSOI/fsoi_inference.py --checkpoint checkpoints/ --config FSOI/configs/fsoi_config.yaml

Author: Azadeh Gholoubi
"""

import argparse
import glob
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Allow running this script from gnn_model/FSOI/ while importing from parent.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lightning.pytorch as pl  # noqa: E402

from gnn_model import GNNLightning  # noqa: E402
from gnn_datamodule import GNNDataModule, BinDataset  # noqa: E402
from fsoi_dataset import (  # noqa: E402
    FSOIDataset,
    create_fsoi_bin_list,
    verify_sequential_consistency,
)
from fsoi_utils import (  # noqa: E402
    get_fsoi_inputs,
    get_fsoi_metadata,
    replace_batch_inputs,
    zero_feature_columns,
    compute_forecast_error,
    compute_adjoints,
    compute_fsoi_per_observation,
    compute_per_level_fsoi,
    compute_per_level_fsoi_by_variable,
    aggregate_fsoi_by_channel,
    aggregate_fsoi_by_instrument,
    sample_innovation_vs_fsoi,
    verify_alignment,
    verify_gradients,
    validate_gradients,  # Hard check for ga/gb
)
from fsoi_model_extensions import (  # noqa: E402
    predict_at_targets,  # Use the CORRECT graph construction method
    freeze_model_for_fsoi,
)
from weight_utils import load_weights_from_yaml  # noqa: E402
from torch_geometric.loader import DataLoader as PyGDataLoader  # noqa: E402


def find_checkpoint(checkpoint_path):
    """
    Find checkpoint file from path or directory.

    Args:
        checkpoint_path: Path to .ckpt file or directory containing checkpoints

    Returns:
        Path to checkpoint file

    Raises:
        ValueError if no checkpoint found
    """
    checkpoint_path = Path(checkpoint_path)

    # If it's a file, use it directly
    if checkpoint_path.is_file():
        if checkpoint_path.suffix == '.ckpt':
            return str(checkpoint_path)
        else:
            raise ValueError(f"File {checkpoint_path} is not a .ckpt file")

    # If it's a directory, find the latest .ckpt file
    elif checkpoint_path.is_dir():
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))

        if not ckpt_files:
            raise ValueError(f"No .ckpt files found in {checkpoint_path}")

        # Sort by modification time, take the latest
        latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
        print(f"Found {len(ckpt_files)} checkpoint(s), using latest: {latest_ckpt.name}")
        return str(latest_ckpt)

    else:
        raise ValueError(f"Path {checkpoint_path} does not exist")


def load_fsoi_config(config_path: str) -> dict:
    """Load FSOI configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_directory(output_dir: str) -> Path:
    """Create output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_path / "csv").mkdir(exist_ok=True)
    (output_path / "zarr").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)

    return output_path


def _as_scalar_bin(x):
    """
    Unwrap batch attribute to scalar value.

    PyTorch Geometric DataLoader collates batch attributes into lists.
    For example, bin_name becomes ['bin1'] instead of 'bin1'.

    This helper unwraps single-element lists/tuples to their scalar value.

    Args:
        x: Value from batch attribute (may be list, tuple, or scalar)

    Returns:
        Scalar value (unwrapped if needed)
    """
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def _expand_seviri_instrument_aliases(names):
    """Expand legacy 'seviri' instrument alias to both v7 streams."""
    if names is None:
        return None
    if isinstance(names, str):
        names = [names]

    expanded = []
    for name in names:
        if name == 'seviri':
            expanded.extend(['seviri_asr', 'seviri_csr'])
        else:
            expanded.append(name)

    # Stable de-duplication while preserving order.
    out = []
    seen = set()
    for name in expanded:
        if name in seen:
            continue
        out.append(name)
        seen.add(name)
    return out


def _compute_innovation_diagnostics(
    xa: dict,
    xb: dict,
    pair_idx: int,
    curr_bin: str,
    lead_step: int,
) -> list[dict]:
    """Compute per-instrument, per-channel innovation statistics.

    Returns a list of records (one per instrument×channel) with:
      mean, std, skewness, median, IQR-scaled spread, Bowley skewness,
      rmse, obs_range, normalized_rmse
    A normalized_rmse < 0.05 (~5% of obs range) indicates xb is a good
    background; > 0.20 suggests the background is unreliable.
    """
    records = []
    for inst in xa:
        if inst not in xb:
            continue
        xa_cpu = xa[inst].detach().cpu().float()
        dx = (xa_cpu - xb[inst].detach().cpu().float())  # [N, C]
        n_obs, n_ch = dx.shape
        for ch in range(n_ch):
            d = dx[:, ch].numpy()
            x = xa_cpu[:, ch].numpy()
            if len(d) == 0:
                continue

            # Mask sentinel observations: process_timeseries.py fills missing
            # channels with exactly -9.0 in normalised space.  Exclude these
            # from all statistics so they don't contaminate means / RMSE.
            valid = ~np.isclose(x, -9.0, atol=1e-3)
            d = d[valid]
            x = x[valid]
            n_valid = int(valid.sum())
            if n_valid < 2:
                continue

            mu = float(np.mean(d))
            sigma = float(np.std(d))
            rmse = float(np.sqrt(np.mean(d ** 2)))
            obs_range = float(np.ptp(x)) if len(x) > 1 else float(np.abs(x).max() + 1e-9)
            skew = float(np.mean(((d - mu) / (sigma + 1e-12)) ** 3)) if sigma > 1e-12 else 0.0
            q1, q2, q3 = np.percentile(d, [25.0, 50.0, 75.0])
            iqr = float(q3 - q1)
            iqr_scaled = float(iqr / 1.349) if iqr > 1e-12 else 0.0
            bowley_skew = float((q3 + q1 - 2.0 * q2) / iqr) if iqr > 1e-12 else 0.0
            records.append({
                'pair_idx': pair_idx,
                'curr_bin': curr_bin,
                'lead_step': lead_step,
                'instrument': inst,
                'channel': ch + 1,
                'n_obs': n_valid,
                'innovation_mean': mu,
                'innovation_std': sigma,
                'innovation_skewness': skew,
                'innovation_median': float(q2),
                'innovation_iqr_scaled': iqr_scaled,
                'innovation_bowley_skewness': bowley_skew,
                'innovation_rmse': rmse,
                'obs_range': obs_range,
                'normalized_rmse': rmse / (obs_range + 1e-12),
            })
    return records


def _run_ose_check(
    results: dict,
    xa: dict,
    xb: dict,
    ea_control: float,
    subsample_indices: dict,
    model,
    curr_batch,
    observation_config: dict,
    ose_instruments: list,
    target_instruments,
    target_variables,
    target_pressure_levels,
    instrument_weights: dict,
    channel_weights: dict,
    use_area_weights: bool,
    loss_reduction: str,
    impact_factor: float,
    lead_step: int,
    pair_idx: int,
    gfs_reference=None,
    mesh_instrument: str = "radiosonde",
    mesh_pressure_level_idx: int = None,
    init_time_unix: int = None,
    spatial_output_dir: str = None,
):
    """Compute OSE impact and append to results['ose_records'].""" 
    from fsoi_ose import (
        compute_matched_conditional_fsoi_for_pair as _matched_fsoi_pair,
        compute_ose_for_pair as _ose_pair,
    )
    print(f"[OSE] Computing denial experiment for: {ose_instruments} (pair {pair_idx})")
    try:
        if isinstance(gfs_reference, dict):
            gfs_tensor_for_ose = gfs_reference.get(mesh_instrument)
        else:
            gfs_tensor_for_ose = gfs_reference
        rec = _ose_pair(
            model=model,
            curr_batch=curr_batch,
            xa=xa,
            xb=xb,
            denied_instruments=ose_instruments,
            ea_control=ea_control,
            observation_config=observation_config,
            subsample_indices=subsample_indices,
            target_instruments=target_instruments,
            target_variables=target_variables,
            target_pressure_levels=target_pressure_levels,
            instrument_weights=instrument_weights,
            channel_weights=channel_weights,
            use_area_weights=use_area_weights,
            loss_reduction=loss_reduction,
            forecast_lead_step=lead_step,
            pair_idx=pair_idx,
            curr_bin=results.get('curr_bin', ''),
            prev_bin=results.get('prev_bin', ''),
            gfs_reference=gfs_tensor_for_ose,
            mesh_instrument=mesh_instrument,
            mesh_pressure_level_idx=mesh_pressure_level_idx,
            init_time_unix=init_time_unix,
            spatial_output_dir=spatial_output_dir,
        )
        if rec:
            if gfs_tensor_for_ose is None:
                try:
                    matched = _matched_fsoi_pair(
                        model=model,
                        curr_batch=curr_batch,
                        xa=xa,
                        xb=xb,
                        denied_instruments=ose_instruments,
                        observation_config=observation_config,
                        subsample_indices=subsample_indices,
                        target_instruments=target_instruments,
                        target_variables=target_variables,
                        target_pressure_levels=target_pressure_levels,
                        instrument_weights=instrument_weights,
                        channel_weights=channel_weights,
                        use_area_weights=use_area_weights,
                        loss_reduction=loss_reduction,
                        forecast_lead_step=lead_step,
                        pair_idx=pair_idx,
                        curr_bin=results.get('curr_bin', ''),
                        prev_bin=results.get('prev_bin', ''),
                        impact_factor=impact_factor,
                    )
                    if matched:
                        rec.update(matched)
                except Exception as matched_err:
                    print(f"[OSE Matched] WARNING on pair {pair_idx}: {matched_err}")
            else:
                print("[OSE Matched] Skipping obs-space matched FSOI for mesh OSE")
            results['ose_records'].append(rec)
            print(f"[OSE]   ea_control={rec['ea_control']:.4e}  "
                  f"ea_denied={rec['ea_denied']:.4e}  "
                  f"ose_impact={rec['ose_impact']:+.4e}  "
                  f"({rec['ose_sign']})")
            if 'matched_fsoi' in rec:
                print(f"[OSE Matched]   delta_j_actual={rec['delta_j_actual']:+.4e}  "
                      f"matched_fsoi={rec['matched_fsoi']:+.4e}  "
                      f"closure={rec['matched_closure_ratio']:.3f}  "
                      f"sign_agree={rec['matched_sign_agree']}")
    except Exception as ose_err:
        print(f"[OSE] ERROR on pair {pair_idx}: {ose_err}")


def compute_fsoi_for_pair(
    model,
    prev_batch,
    curr_batch,
    fsoi_config: dict,
    observation_config: dict,
    instrument_weights: dict,
    channel_weights: dict,
    pair_idx: int,
    verbose: bool = True,
    run_fd_check: bool = False,
    run_directional_check: bool = False,
    run_float64_check: bool = False,
    run_diagnostics: bool = False,
    run_repro_check: bool = False,
    ose_instruments: list = None,
    gfs_reference: dict = None,
    mesh_instrument: str = "radiosonde",
    mesh_pressure_level_idx: int = 4,
    ose_spatial_output_dir: str = None,
    ose_spatial_pair_indices: set = None,
):
    """
    Compute FSOI for a single (prev, curr) batch pair.

    This is the core FSOI computation implementing:
    FSOI(k) = δx(k) ⊙ (ga(k) + gb(k))

    Args:
        model: Trained GNN model (frozen)
        prev_batch: Previous window batch (k-1)
        curr_batch: Current window batch (k)
        fsoi_config: FSOI configuration dict
        observation_config: Observation configuration dict
        instrument_weights: Weight per instrument
        channel_weights: Weight per channel
        pair_idx: Index of this pair (for logging)
        verbose: Print detailed diagnostics

    Returns:
        Dict with FSOI results and metadata
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Computing FSOI for pair {pair_idx}")
        # Note: bin_name will be unwrapped to scalar in results dict
        print(f"Previous: {prev_batch.bin_name} (type: {type(prev_batch.bin_name)})")
        print(f"Current:  {curr_batch.bin_name} (type: {type(curr_batch.bin_name)})")
        print(f"{'=' * 80}\n")

    device = next(model.parameters()).device

    # Move batches to device
    prev_batch = prev_batch.to(device)
    curr_batch = curr_batch.to(device)

    # Get forecast lead step(s) to score
    forecast_lead_steps = fsoi_config['forecast']['lead_steps']
    if not isinstance(forecast_lead_steps, list):
        forecast_lead_steps = [forecast_lead_steps]

    # Get target instruments filter
    target_instruments = fsoi_config['forecast'].get('target_instruments', 'all')
    if target_instruments == 'all':
        target_instruments = None
    else:
        expanded_targets = _expand_seviri_instrument_aliases(target_instruments)
        if expanded_targets != target_instruments:
            print(f"[FSOI Config] Expanded forecast.target_instruments {target_instruments} -> {expanded_targets}")
        target_instruments = expanded_targets

    # Get target variables filter (e.g., ["temperature"])
    target_variables = fsoi_config['forecast'].get('target_variables', 'all')
    if target_variables == 'all':
        target_variables = None

    # Optional: additionally stratify the FORECAST ERROR metric by target variable
    # (i.e., one backward pass per (pressure, variable) slice).
    stratify_by_variable = fsoi_config['forecast'].get('stratify_by_variable', False)

    # Get target pressure levels filter (e.g., [850, 500, 250])
    target_pressure_levels = fsoi_config['forecast'].get('target_pressure_levels', 'all')
    if target_pressure_levels == 'all':
        target_pressure_levels = None

    use_area_weights = fsoi_config['forecast'].get('use_area_weights', True)
    loss_reduction = fsoi_config['forecast'].get('loss_reduction', 'sum')
    impact_factor = float(fsoi_config['forecast'].get('impact_factor', 0.5))
    print(f"[FSOI Config] loss_reduction={loss_reduction}, impact_factor={impact_factor}")

    # Optional: save innovation-vs-FSOI scatter sample for plotting
    save_scatter_samples = fsoi_config.get('plots', {}).get('save_scatter_samples', True)
    scatter_max_points = int(fsoi_config.get('plots', {}).get('scatter_max_points', 200000))

    # Results storage
    # CRITICAL: Unwrap bin_name from PyG DataLoader collation (list → scalar)
    results = {
        'pair_idx': pair_idx,
        'prev_bin': _as_scalar_bin(prev_batch.bin_name),
        'curr_bin': _as_scalar_bin(curr_batch.bin_name),
        'fsoi_by_step': {},
        'scatter_samples': [],
        'fd_records': [],          # finite-difference validation records for this pair
        'directional_records': [],  # Rademacher directional-derivative records
        'float64_records': [],     # float64 per-obs FD records
        'diag_records': [],        # innovation diagnostic records for this pair
        'ose_records': [],         # OSE cross-check records for this pair
    }

    # Process each forecast lead step
    for lead_step in forecast_lead_steps:
        if verbose:
            print(f"\n--- Processing Lead Step {lead_step} ---\n")

        # ========================================
        # STEP 1: Extract analysis inputs (xa) and metadata
        # ========================================
        print("[1/6] Extracting analysis inputs (xa) and metadata...")
        print("[FSOI Strategy] Extracting observation CHANNELS from current INPUT nodes")
        xa = get_fsoi_inputs(
            curr_batch,
            observation_config,
            model.instrument_name_to_id,
            match_targets=False,  # Unused parameter (kept for compatibility)
        )

        if not xa:
            print("[WARNING] No analysis inputs found, skipping this pair")
            continue

        # Extract metadata (pressure levels, lat/lon, etc.)
        metadata = get_fsoi_metadata(curr_batch, observation_config)

        # Propagate target pressure to all instruments for downstream plotting/aggregation
        target_pressure_level = None
        target_pressure_hpa = None
        for meta in metadata.values():
            if isinstance(meta, dict) and meta.get('pressure_level') is not None:
                target_pressure_level = meta['pressure_level']
                target_pressure_hpa = meta.get('pressure_hpa')
                break

        metadata['_target_pressure_level'] = target_pressure_level
        metadata['_target_pressure_hpa'] = target_pressure_hpa

        # Feature masks applied at inference (zero selected channels by name)
        raw_mask_map = fsoi_config.get('data', {}).get('feature_masks', {}) or {}
        feature_mask_map = {}
        for inst_name, feats in raw_mask_map.items():
            if feats is None:
                continue
            expanded_names = _expand_seviri_instrument_aliases(inst_name)
            feat_list = [feats] if isinstance(feats, str) else list(feats)
            for expanded_name in expanded_names:
                feature_mask_map[expanded_name] = list(feat_list)

        # ========================================
        # STEP 2: Compute background inputs (xb)
        # ========================================
        print("[2/6] Computing background inputs (xb) from previous window...")
        print("[Background Strategy] Predict at current INPUT locations using prev window obs")

        # MEMORY OPTIMIZATION: Only create pseudo-targets for instruments in xa
        # This prevents creating heavy pseudo-targets for instruments we don't need
        keep_x_instruments = list(xa.keys())
        print(f"[MEMORY] Creating xb predictions for {len(keep_x_instruments)} instruments: {keep_x_instruments}")

        # Per-instrument cap on decoder nodes. Avhrr has 1.3M nodes which
        # causes the GAT attention allocation to exceed 11 GiB on its own.
        # Reasonable defaults: satellite swaths ~20-50k, conventional obs are
        # small and do not need capping.
        mem_config = fsoi_config.get('memory', {}) or {}
        max_decoder_nodes_cfg = mem_config.get('max_decoder_nodes', {}) or {}
        if 'seviri' in max_decoder_nodes_cfg:
            legacy_cap = max_decoder_nodes_cfg.get('seviri')
            max_decoder_nodes_cfg = dict(max_decoder_nodes_cfg)
            max_decoder_nodes_cfg.pop('seviri', None)
            max_decoder_nodes_cfg.setdefault('seviri_asr', legacy_cap)
            max_decoder_nodes_cfg.setdefault('seviri_csr', legacy_cap)
            print(
                "[FSOI Config] Expanded memory.max_decoder_nodes.seviri "
                f"-> seviri_asr={max_decoder_nodes_cfg.get('seviri_asr')}, "
                f"seviri_csr={max_decoder_nodes_cfg.get('seviri_csr')}"
            )
        # Defaults for known heavy instruments (override in fsoi_config.yaml)
        default_caps = {
            'avhrr': 30000,
            'atms': 50000,
            'amsua': 30000,
            'ssmis': 30000,
            'seviri_asr': 30000,
            'seviri_csr': 30000,
        }
        max_decoder_nodes = {**default_caps, **max_decoder_nodes_cfg}

        # Predict current window observations from previous window
        # Graph construction:
        #   - Encoder: prev_batch INPUT obs → mesh
        #   - Decoder: mesh → curr_batch INPUT locations (pseudo-targets)
        #   - Result: xb predictions at same locations as xa
        xb_raw, subsample_indices = predict_at_targets(
            model,
            prev_batch,
            curr_batch,  # Pass curr_batch for INPUT locations and metadata
            observation_config,  # Pass config for proper channel/metadata handling
            forecast_step=lead_step,
            keep_instruments=keep_x_instruments,  # Only predict for instruments in xa
            max_decoder_nodes=max_decoder_nodes,  # Cap heavy decoder instruments
        )

        # Track raw vs sampled row counts so aggregate CSVs can include
        # scaled total-impact columns for instruments capped by max_decoder_nodes.
        sampling_info = {}
        for inst_name, tensor in xa.items():
            raw_n = int(tensor.shape[0])
            idx = subsample_indices.get(inst_name)
            sampled_n = int(idx.numel()) if idx is not None else raw_n
            sample_scale = float(raw_n / sampled_n) if sampled_n > 0 else 1.0
            sampling_info[inst_name] = {
                'raw_n_observations': raw_n,
                'sampled_n_observations': sampled_n,
                'sample_scale': sample_scale,
                'is_subsampled': idx is not None,
            }
            if idx is not None:
                print(
                    f"[SAMPLING] {inst_name}: raw={raw_n}, sampled={sampled_n}, "
                    f"scale={sample_scale:.3f}"
                )

        # Clear CUDA cache after forward pass to free memory
        torch.cuda.empty_cache()
        print(f"[MEMORY] Cleared CUDA cache after xb computation")

        # ALIGNMENT: subsample xa to the same rows that were decoded for xb.
        # After this block both xa[inst] and xb[inst] have the same shape so
        # δx = xa - xb is well-defined.  replace_batch_inputs will place these
        # subsampled tensors at the correct row positions inside the full batch
        # tensor via indexed assignment (replace_indices).
        for inst_name, idx in subsample_indices.items():
            if idx is None:
                continue
            if inst_name in xa:
                xa[inst_name] = xa[inst_name][idx].clone().detach()
                xa[inst_name].requires_grad_(True)
                print(f"[ALIGN] Subsampled xa[{inst_name}] → {xa[inst_name].shape[0]} obs "
                      f"(matched to xb subsample)")

        # Build obs_coords: per-instrument (lat, lon) numpy arrays already aligned
        # with the (possibly subsampled) xa tensors, for use in scatter samples.
        obs_coords: dict = {}
        for inst_name, meta in metadata.items():
            if not isinstance(meta, dict):
                continue
            lat_t = meta.get('lat')
            lon_t = meta.get('lon')
            if lat_t is None or lon_t is None:
                continue
            lat_np = lat_t.detach().cpu().numpy() if torch.is_tensor(lat_t) else np.asarray(lat_t)
            lon_np = lon_t.detach().cpu().numpy() if torch.is_tensor(lon_t) else np.asarray(lon_t)
            # Apply the same subsampling that was applied to xa
            idx = subsample_indices.get(inst_name)
            if idx is not None:
                cpu_idx = idx.cpu().numpy()
                lat_np = lat_np[cpu_idx]
                lon_np = lon_np[cpu_idx]
            obs_coords[inst_name] = (lat_np, lon_np)

        # Optionally zero specific channels by feature name at inference time
        zero_feature_columns(xa, observation_config, feature_mask_map)

        # Enable gradients for xb
        # NOTE: xb is treated as an independent variable for computing ∂e/∂xb
        # We are NOT differentiating through the previous-window model
        # This computes FSOI impact in current observation space
        xb = {}
        for inst_name, tensor in xb_raw.items():
            if inst_name in xa:  # Only keep instruments that are also in xa
                xb_tensor = tensor.clone().detach()
                tmp = {inst_name: xb_tensor}
                zero_feature_columns(tmp, observation_config, feature_mask_map)
                xb_tensor = tmp[inst_name]  # pick up any new tensor zero_feature_columns may have returned
                xb_tensor.requires_grad_(True)
                xb[inst_name] = xb_tensor

        if not xb:
            print("[WARNING] No background predictions, skipping this pair")
            continue

        # ========================================
        # STEP 3: Verify alignment
        # ========================================
        if fsoi_config['validation'].get('check_alignment', True):
            print("[3/6] Verifying xa/xb alignment...")
            # Instruments that were subsampled: xa/xb row counts intentionally differ
            # from the full batch node counts, so skip the batch-size count check.
            subsampled_insts = {k for k, v in subsample_indices.items() if v is not None}
            aligned = verify_alignment(
                xa, xb, curr_batch,
                verbose=verbose,
                check_spatial=len(subsampled_insts) == 0,
                skip_count_for=subsampled_insts,
            )
            if not aligned:
                print("[ERROR] Alignment check failed, skipping this pair")
                continue
        else:
            print("[3/6] Skipping alignment check (disabled in config)")

        # ── Resolve init_time_unix from batch (needed for mesh decoder) ────────
        init_time_unix = None
        try:
            init_time_unix = model._extract_init_time_unix(curr_batch)
        except Exception:
            pass

        # ── Mesh-space verification path ─────────────────────────────────────
        # When gfs_reference is provided, replace the obs-space error function
        # with mesh-space MSE(OCELOT_mesh_pred, GFS_analysis).
        # The innovation δx = xa - xb and the FSOI formula are unchanged —
        # only the error metric (and its gradients) changes.
        if gfs_reference is not None and init_time_unix is not None:
            from fsoi_utils import compute_forecast_error_on_mesh
            gfs_tensor = gfs_reference.get(mesh_instrument)
            if gfs_tensor is None:
                print(f"[MESH FSOI] No GFS tensor for '{mesh_instrument}', "
                      f"falling back to obs-space error")
            else:
                # Monkey-patch: wrap compute_forecast_error to use mesh error
                # for ea/eb inside the non-stratified path below.
                # We store as a closure; the stratified path is not supported
                # in mesh mode (would need one GFS load per level, very expensive).
                def _mesh_ea_fn(batch_): return compute_forecast_error_on_mesh(
                    model=model,
                    batch=batch_,
                    gfs_reference=gfs_tensor,
                    mesh_instrument=mesh_instrument,
                    forecast_lead_step=lead_step,
                    init_time_unix=init_time_unix,
                    use_area_weights=use_area_weights,
                    loss_reduction=loss_reduction,
                )
                print(
                    f"[MESH FSOI] Using OCELOT mesh verification "
                    f"with GFS reference at pair {pair_idx}"
                )

        # Optional: innovation diagnostics (background quality check)
        if run_diagnostics:
            diag = _compute_innovation_diagnostics(
                xa, xb,
                pair_idx=pair_idx,
                curr_bin=results['curr_bin'],
                lead_step=lead_step,
            )
            results['diag_records'].extend(diag)
            bad = [r for r in diag if r['normalized_rmse'] > 0.20]
            if bad:
                insts = sorted({r['instrument'] for r in bad})
                print(f"[DIAG] WARNING: {len(bad)} channels have normalized_rmse > 20%: "
                      f"{insts} — xb may be a poor background for these instruments")

        # Optional: finite-difference gradient validation
        if run_fd_check:
            print("\n[VALIDATION] Running finite-difference gradient check "
                  f"(pair {pair_idx})...")
            from fsoi_validation import validate_fsoi_gradients, finite_difference_check as fd_check_one

            num_samples = fsoi_config['validation'].get('fd_num_samples', 5)
            epsilon = fsoi_config['validation'].get('fd_epsilon', 1e-2)

            fd_passed = validate_fsoi_gradients(
                model, prev_batch, curr_batch,
                num_samples=num_samples, epsilon=epsilon,
            )

            if not fd_passed:
                print("[WARNING] Finite-difference validation failed!")
                if fsoi_config['validation'].get('require_fd_pass', False):
                    print("[ERROR] Stopping due to failed validation")
                    return results
            else:
                print(f"[VALIDATION] Finite-difference check PASSED (pair {pair_idx})")

            # Accumulate per-sample FD records keyed by pair and bin
            try:
                fd_xa = get_fsoi_inputs(curr_batch, observation_config,
                                        model.instrument_name_to_id)
                rng_fd = np.random.default_rng(pair_idx * 7919 + 42)
                inst_list = [k for k in fd_xa if fd_xa[k].shape[0] > 0]
                for _ in range(num_samples):
                    inst_fd = inst_list[rng_fd.integers(len(inst_list))]
                    x = fd_xa[inst_fd]
                    obs_i = int(rng_fd.integers(x.shape[0]))
                    ch_i = int(rng_fd.integers(x.shape[1]))
                    rec = fd_check_one(
                        model, curr_batch, inst_fd, obs_i, ch_i,
                        epsilon=epsilon, forecast_step=lead_step,
                    )
                    if rec:
                        rec['pair_idx'] = pair_idx
                        rec['curr_bin'] = results['curr_bin']
                        results['fd_records'].append(rec)
            except Exception as fd_err:
                print(f"[WARNING] FD sample collection failed for pair {pair_idx}: {fd_err}")

        # Optional: directional-derivative check (Rademacher, float32) for satellite instruments
        if run_directional_check:
            print("\n[VALIDATION] Running directional-derivative check "
                  f"(Rademacher, float32, pair {pair_idx})...")
            try:
                from fsoi_validation import directional_derivative_check
                from fsoi_utils import get_fsoi_inputs as _get_inputs
                _HIGH_N = {'atms', 'avhrr', 'ssmis', 'ascat', 'amsua', 'seviri_asr', 'seviri_csr'}
                n_trials = fsoi_config['validation'].get('directional_n_trials', 5)
                eps_dir = fsoi_config['validation'].get('directional_epsilon',
                                                        fsoi_config['validation'].get('fd_epsilon', 1e-2))
                _xa_dir = _get_inputs(curr_batch, observation_config, model.instrument_name_to_id)
                for inst in sorted(_xa_dir.keys()):
                    if inst not in _HIGH_N:
                        continue
                    dr = directional_derivative_check(
                        model, curr_batch, inst,
                        epsilon=eps_dir,
                        n_trials=n_trials,
                        seed=pair_idx * 31 + 7,
                        forecast_step=lead_step,
                    )
                    if dr:
                        for t in dr.get('trials', []):
                            row = {
                                'pair_idx': pair_idx,
                                'curr_bin': results['curr_bin'],
                                'inst_name': inst,
                                'trial': t['trial'],
                                'autograd_dd': t['autograd_dd'],
                                'fd_dd': t['fd_dd'],
                                'rel_error': t['rel_error'],
                                'pearson_r': dr['pearson_r'],
                                'l1_norm': dr['l1_norm'],
                                'n_ulp': dr['n_ulp'],
                                'status': dr['status'],
                            }
                            results['directional_records'].append(row)
            except Exception as dir_err:
                print(f"[WARNING] Directional derivative check failed for pair {pair_idx}: {dir_err}")

        # Optional: float64 per-obs FD check for satellite instruments
        if run_float64_check:
            print("\n[VALIDATION] Running float64 FD check "
                  f"(pair {pair_idx})...")
            try:
                from fsoi_validation import finite_difference_check_float64
                from fsoi_utils import get_fsoi_inputs as _get_inputs
                _HIGH_N = {'atms', 'avhrr', 'ssmis', 'ascat', 'amsua', 'seviri_asr', 'seviri_csr'}
                n_f64 = fsoi_config['validation'].get('float64_num_samples', 3)
                eps_f64 = fsoi_config['validation'].get('float64_epsilon', 1e-4)
                _xa_f64 = _get_inputs(curr_batch, observation_config, model.instrument_name_to_id)
                rng_f64 = np.random.default_rng(pair_idx * 1009 + 13)
                for inst in sorted(_xa_f64.keys()):
                    if inst not in _HIGH_N:
                        continue
                    x = _xa_f64[inst]
                    for _ in range(n_f64):
                        obs_i = int(rng_f64.integers(x.shape[0]))
                        ch_i = int(rng_f64.integers(x.shape[1]))
                        rec = finite_difference_check_float64(
                            model, curr_batch, inst, obs_i, ch_i,
                            epsilon=eps_f64,
                            forecast_step=lead_step,
                        )
                        if rec:
                            rec['pair_idx'] = pair_idx
                            rec['curr_bin'] = results['curr_bin']
                            results['float64_records'].append(rec)
            except Exception as f64_err:
                print(f"[WARNING] Float64 FD check failed for pair {pair_idx}: {f64_err}")

        # Whether to compute a separate loss per radiosonde pressure level
        # (enables pressure_hpa column for ALL instruments including satellites)
        stratify_by_pressure = fsoi_config['forecast'].get('stratify_by_pressure', False)

        # Variable stratification also uses the stratified backward-pass path.
        # Surface targets have no vertical structure (stratify_by_pressure=False)
        # but still need per-variable attribution (T, Td, u, v, ps); the
        # per-variable kernel handles the no-pressure case as a single surface
        # level, so enable the stratified path whenever either flag is set.
        use_stratified_path = stratify_by_pressure or stratify_by_variable

        if use_stratified_path:
            # ====================================================
            # STEPS 4-6 (STRATIFIED): one backward pass per level
            # ====================================================
            print("[4-6/6] Pressure-stratified FSOI: one backward pass per target level...")

            if stratify_by_variable:
                raw_results = compute_per_level_fsoi_by_variable(
                    model=model,
                    curr_batch=curr_batch,
                    xa=xa,
                    xb=xb,
                    observation_config=observation_config,
                    forecast_lead_step=lead_step,
                    instrument_weights=instrument_weights,
                    channel_weights=channel_weights,
                    use_area_weights=use_area_weights,
                    target_instruments=target_instruments,
                    replace_indices=subsample_indices,
                    requested_target_variables=target_variables,
                    target_pressure_levels=target_pressure_levels,
                    loss_reduction=loss_reduction,
                    impact_factor=impact_factor,
                )

                per_level_results = []
                scatter_saved = False
                for mr in raw_results:
                    # Broadcast metric target pressure for all instruments.
                    # Surface targets have no pressure level (p_idx is None), so
                    # skip the broadcast: satellites then aggregate per-channel
                    # without a bogus target level (avoids torch.tensor([None])).
                    meta_m = dict(metadata)
                    if mr.get('p_idx') is not None:
                        meta_m['_target_pressure_level'] = torch.tensor([mr['p_idx']])
                        meta_m['_target_pressure_hpa'] = torch.tensor([mr['p_hpa']])

                    df_ch = aggregate_fsoi_by_channel(
                        mr['fsoi_values'],
                        model.instrument_name_to_id,
                        metadata=meta_m,
                        innovations=mr.get('innovations'),
                        gradient_sums=mr.get('gradient_sums'),
                        sampling_info=sampling_info,
                    )
                    df_ch['target_variable'] = mr.get('target_variable')
                    df_ch['target_channel'] = mr.get('target_channel')
                    df_ch['p_idx'] = mr.get('p_idx')
                    df_ch['p_hpa'] = mr.get('p_hpa')

                    df_inst = aggregate_fsoi_by_instrument(
                        mr['fsoi_values'],
                        model.instrument_name_to_id,
                        innovations=mr.get('innovations'),
                        gradient_sums=mr.get('gradient_sums'),
                        sampling_info=sampling_info,
                    )
                    df_inst['target_variable'] = mr.get('target_variable')
                    df_inst['target_channel'] = mr.get('target_channel')
                    df_inst['p_idx'] = mr.get('p_idx')
                    df_inst['p_hpa'] = mr.get('p_hpa')

                    per_level_results.append(
                        {
                            'p_idx': mr['p_idx'],
                            'p_hpa': mr['p_hpa'],
                            'target_variable': mr.get('target_variable'),
                            'target_channel': mr.get('target_channel'),
                            'ea_p': mr.get('ea_p', 0.0),
                            'eb_p': mr.get('eb_p', 0.0),
                            'fsoi_channel_aggregates': df_ch,
                            'fsoi_instrument_aggregates': df_inst,
                        }
                    )

                    if save_scatter_samples and (not scatter_saved) and scatter_max_points > 0:
                        # Save ONE representative metric's scatter sample per (pair, lead_step)
                        # to avoid huge files.
                        scatter_df = sample_innovation_vs_fsoi(
                            mr['fsoi_values'],
                            mr.get('innovations') or {},
                            max_points=scatter_max_points,
                            seed=pair_idx * 1000 + int(lead_step),
                            obs_coords=obs_coords,
                            xa=xa,  # enables exact sentinel mask (xa==-9.0) instead of range heuristic
                        )
                        if not scatter_df.empty:
                            scatter_df['pair_idx'] = pair_idx
                            scatter_df['lead_step'] = lead_step
                            scatter_df['target_variable'] = mr.get('target_variable')
                            scatter_df['p_hpa'] = mr.get('p_hpa')
                            results['scatter_samples'].append(scatter_df)
                            scatter_saved = True

            else:
                per_level_results = compute_per_level_fsoi(
                    model=model,
                    curr_batch=curr_batch,
                    xa=xa,
                    xb=xb,
                    observation_config=observation_config,
                    forecast_lead_step=lead_step,
                    instrument_weights=instrument_weights,
                    channel_weights=channel_weights,
                    use_area_weights=use_area_weights,
                    target_instruments=target_instruments,
                    target_variables=target_variables,
                    replace_indices=subsample_indices,
                    target_pressure_levels=target_pressure_levels,
                    loss_reduction=loss_reduction,
                    impact_factor=impact_factor,
                )

            if not per_level_results:
                print("[WARNING] No per-level FSOI results; skipping pair")
                continue

            ea_total = sum(lr['ea_p'] for lr in per_level_results)
            eb_total = sum(lr['eb_p'] for lr in per_level_results)
            print(f"  Total ea (sum over levels): {ea_total:.6e}")
            print(f"  Total eb (sum over levels): {eb_total:.6e}")

            results['fsoi_by_step'][lead_step] = {
                'ea': ea_total,
                'eb': eb_total,
                'pressure_stratified': True,
                'variable_stratified': bool(stratify_by_variable),
                'per_level': per_level_results,
                'metadata': metadata,
                'sampling_info': sampling_info,
            }

            # ── OSE cross-check (stratified path) ────────────────────────
            if ose_instruments:
                _run_ose_check(
                    results, xa, xb, ea_total, subsample_indices,
                    model, curr_batch, observation_config, ose_instruments,
                    target_instruments, target_variables, target_pressure_levels,
                    instrument_weights, channel_weights, use_area_weights,
                    loss_reduction, impact_factor, lead_step, pair_idx,
                    gfs_reference=gfs_reference,
                    mesh_instrument=mesh_instrument,
                    mesh_pressure_level_idx=mesh_pressure_level_idx,
                    init_time_unix=init_time_unix,
                    spatial_output_dir=(
                        ose_spatial_output_dir
                        if ose_spatial_output_dir
                        and (ose_spatial_pair_indices is None or pair_idx in ose_spatial_pair_indices)
                        else None
                    ),
                )

        else:
            # ========================================
            # STEP 4: Compute analysis adjoint (ga)
            # ========================================
            print("[4/6] Computing analysis adjoint (ga)...")

            # Replace batch inputs with xa (indexed for subsampled instruments)
            curr_batch_xa = curr_batch.clone()
            replace_batch_inputs(curr_batch_xa, xa, observation_config,
                                 replace_indices=subsample_indices)

            # Compute forecast error for analysis (obs-space or mesh-space)
            if gfs_reference is not None and init_time_unix is not None \
                    and gfs_reference.get(mesh_instrument) is not None:
                ea = _mesh_ea_fn(curr_batch_xa)
                print(f"  [MESH] Analysis mesh error (ea): {ea.item():.6e}")
            else:
                ea = compute_forecast_error(
                    model,
                    curr_batch_xa,
                    forecast_lead_step=lead_step,
                    instrument_weights=instrument_weights,
                    channel_weights=channel_weights,
                    use_area_weights=use_area_weights,
                    target_instruments=target_instruments,
                    target_variables=target_variables,
                    target_pressure_levels=target_pressure_levels,
                    loss_reduction=loss_reduction,
                )

            print(f"  Analysis error (ea): {ea.item():.6e}")

            # Clear CUDA cache after forward pass
            torch.cuda.empty_cache()
            print(f"[MEMORY] Cleared CUDA cache after ea computation")

            # EARLY CHECK: No verification targets
            if ea.item() == 0.0:
                print("\n" + "=" * 80)
                print("SKIPPING FSOI PAIR: No verification targets found")
                print("=" * 80)
                print(f"  Lead step: {lead_step}")
                print(f"  Target instruments: {target_instruments}")
                print(f"  Target variables: {target_variables}")
                print(f"  Target pressure levels: {target_pressure_levels}")
                print(f"  This is normal if verification data is sparse or filtered.")
                print(f"  Gradients will be zero, but this is NOT a gradient error.")
                print("=" * 80 + "\n")
                continue  # Skip this lead step

            # Compute gradients
            ga = compute_adjoints(ea, xa, create_graph=False)

            if fsoi_config['validation'].get('check_gradients', True):
                if not verify_gradients(ga, verbose=verbose):
                    print("[WARNING] Analysis gradient check failed")

            # ========================================
            # STEP 5: Compute background adjoint (gb)
            # ========================================
            print("[5/6] Computing background adjoint (gb)...")

            # Replace batch inputs with xb (indexed for subsampled instruments)
            curr_batch_xb = curr_batch.clone()
            replace_batch_inputs(curr_batch_xb, xb, observation_config,
                                 replace_indices=subsample_indices)

            # Compute forecast error for background (obs-space or mesh-space)
            if gfs_reference is not None and init_time_unix is not None \
                    and gfs_reference.get(mesh_instrument) is not None:
                eb = _mesh_ea_fn(curr_batch_xb)
                print(f"  [MESH] Background mesh error (eb): {eb.item():.6e}")
            else:
                eb = compute_forecast_error(
                    model,
                    curr_batch_xb,
                    forecast_lead_step=lead_step,
                    instrument_weights=instrument_weights,
                    channel_weights=channel_weights,
                    use_area_weights=use_area_weights,
                    target_instruments=target_instruments,
                    target_variables=target_variables,
                    target_pressure_levels=target_pressure_levels,
                    loss_reduction=loss_reduction,
                )

            print(f"  Background error (eb): {eb.item():.6e}")

            # Clear CUDA cache after forward pass
            torch.cuda.empty_cache()
            print(f"[MEMORY] Cleared CUDA cache after eb computation")

            if eb.item() == 0.0:
                print("\n[WARNING] Background error is also 0 (consistent with no verification targets)")

            # Compute gradients
            gb = compute_adjoints(eb, xb, create_graph=False)

            torch.cuda.empty_cache()
            print(f"[MEMORY] Cleared CUDA cache after gradient computation")

            if fsoi_config['validation'].get('check_gradients', True):
                if not verify_gradients(gb, verbose=verbose):
                    print("[WARNING] Background gradient check failed")

            # ========================================
            # CRITICAL VALIDATION: Check ga and gb
            # ========================================
            print("[VALIDATION] Hard check: ga and gb must be finite and non-zero...")

            require_instruments = fsoi_config['validation'].get('require_instruments', None)
            require_satellite = fsoi_config['validation'].get('require_satellite', False)

            try:
                validate_gradients(
                    ga, gb,
                    require_instruments=require_instruments,
                    require_satellite=require_satellite
                )
            except ValueError as e:
                print(f"\n{'=' * 80}")
                print("FATAL: Gradient validation FAILED!")
                print(f"{'=' * 80}")
                print(str(e))
                print(f"\nSkipping FSOI pair: pair_idx={pair_idx}")
                print(f"{'=' * 80}\n")
                continue  # Skip this FSOI pair

            # ========================================
            # STEP 6: Compute FSOI
            # ========================================
            print("[6/6] Computing FSOI = δx ⊙ (ga + gb)...")

            # xa and xb are shape-aligned after the ALIGNMENT block above
            fsoi_values, innovations, gradient_sums = compute_fsoi_per_observation(
                xa, xb, ga, gb, return_components=True, impact_factor=impact_factor
            )

            if save_scatter_samples and scatter_max_points > 0:
                scatter_df = sample_innovation_vs_fsoi(
                    fsoi_values,
                    innovations,
                    max_points=scatter_max_points,
                    seed=pair_idx * 1000 + int(lead_step),
                    obs_coords=obs_coords,
                    xa=xa,  # enables exact sentinel mask (xa==-9.0) instead of range heuristic
                )
                if not scatter_df.empty:
                    scatter_df['pair_idx'] = pair_idx
                    scatter_df['lead_step'] = lead_step
                    results['scatter_samples'].append(scatter_df)

            # ── Reproducibility check (non-stratified path only) ──────────
            if run_repro_check:
                print("[REPRO] Re-running ea + ga to verify determinism...")
                try:
                    xa_r = {k: v.clone().detach().requires_grad_(True)
                            for k, v in xa.items()}
                    bat_r = curr_batch.clone()
                    replace_batch_inputs(bat_r, xa_r, observation_config,
                                         replace_indices=subsample_indices)
                    ea_r = compute_forecast_error(
                        model, bat_r,
                        forecast_lead_step=lead_step,
                        instrument_weights=instrument_weights,
                        channel_weights=channel_weights,
                        use_area_weights=use_area_weights,
                        target_instruments=target_instruments,
                        target_variables=target_variables,
                        target_pressure_levels=target_pressure_levels,
                        loss_reduction=loss_reduction,
                    )
                    ga_r = compute_adjoints(ea_r, xa_r, create_graph=False)
                    torch.cuda.empty_cache()

                    ea_diff = abs(ea_r.item() - ea.item())
                    ga_diffs = [
                        (ga_r[k] - ga[k]).abs().max().item()
                        for k in ga if k in ga_r
                    ]
                    max_ga_diff = max(ga_diffs) if ga_diffs else float('nan')
                    ea_rel = ea_diff / (abs(ea.item()) + 1e-12)

                    if ea_diff < 1e-5 and max_ga_diff < 1e-5:
                        print(f"[REPRO] PASS — ea_diff={ea_diff:.2e}, "
                              f"max_ga_diff={max_ga_diff:.2e}")
                        results['repro_status'] = 'PASS'
                    else:
                        print(f"[REPRO] WARN — ea_diff={ea_diff:.2e} "
                              f"(rel={ea_rel:.2e}), max_ga_diff={max_ga_diff:.2e}")
                        print("  Non-deterministic ops (e.g. atomic scatter) may "
                              "cause small diffs; check if diff > 1e-3")
                        results['repro_status'] = 'WARN'
                    results['repro_ea_diff'] = ea_diff
                    results['repro_max_ga_diff'] = max_ga_diff
                except Exception as repro_err:
                    print(f"[REPRO] ERROR: {repro_err}")
                    results['repro_status'] = 'ERROR'

            # ── OSE cross-check (non-stratified path) ────────────────────
            if ose_instruments:
                _run_ose_check(
                    results, xa, xb, ea.item(), subsample_indices,
                    model, curr_batch, observation_config, ose_instruments,
                    target_instruments, target_variables, target_pressure_levels,
                    instrument_weights, channel_weights, use_area_weights,
                    loss_reduction, impact_factor, lead_step, pair_idx,
                    gfs_reference=gfs_reference,
                    mesh_instrument=mesh_instrument,
                    mesh_pressure_level_idx=mesh_pressure_level_idx,
                    init_time_unix=init_time_unix,
                    spatial_output_dir=(
                        ose_spatial_output_dir
                        if ose_spatial_output_dir
                        and (ose_spatial_pair_indices is None or pair_idx in ose_spatial_pair_indices)
                        else None
                    ),
                )

        # Memory control: only store full tensors if enabled (non-stratified path)
        store_full_tensors = fsoi_config['data'].get('store_full_tensors', False)

        if use_stratified_path:
            pass  # already stored above via per_level_results

        elif store_full_tensors:
            # Store all tensors for detailed analysis
            results['fsoi_by_step'][lead_step] = {
                'ea': ea.item(),
                'eb': eb.item(),
                'fsoi_values': fsoi_values,
                'innovations': {k: v.detach().cpu() for k, v in innovations.items()},
                'gradient_sums': {k: v.detach().cpu() for k, v in gradient_sums.items()},
                'xa': {k: v.detach().cpu() for k, v in xa.items()},
                'xb': {k: v.detach().cpu() for k, v in xb.items()},
                'ga': {k: v.detach().cpu() for k, v in ga.items()},
                'gb': {k: v.detach().cpu() for k, v in gb.items()},
                'sampling_info': sampling_info,
            }
            print(f"  [Memory] Stored full xa/xb/ga/gb tensors")

        else:
            # Only store aggregated statistics (memory efficient)
            # Compute aggregates immediately before discarding tensors
            channel_agg = aggregate_fsoi_by_channel(
                fsoi_values,
                model.instrument_name_to_id,  # Use model mapping for aggregation
                metadata=metadata,  # Pass metadata for pressure level stratification
                innovations=innovations,
                gradient_sums=gradient_sums,
                sampling_info=sampling_info,
            )

            inst_agg = aggregate_fsoi_by_instrument(
                fsoi_values,
                model.instrument_name_to_id,  # Use model mapping for aggregation
                innovations=innovations,
                gradient_sums=gradient_sums,
                sampling_info=sampling_info,
            )

            results['fsoi_by_step'][lead_step] = {
                'ea': ea.item(),
                'eb': eb.item(),
                'fsoi_channel_aggregates': channel_agg,
                'fsoi_instrument_aggregates': inst_agg,
                # fsoi_values still stored as it's needed for final aggregation
                'fsoi_values': {k: v.detach().cpu() for k, v in fsoi_values.items()},
                'innovations': {k: v.detach().cpu() for k, v in innovations.items()},
                'gradient_sums': {k: v.detach().cpu() for k, v in gradient_sums.items()},
                # Store metadata for final aggregation with pressure levels
                'metadata': metadata,
                'sampling_info': sampling_info,
            }
            print(f"  [Memory] Stored only aggregated FSOI statistics (saved memory)")

        print(f"\n✓ FSOI computation complete for lead step {lead_step}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FSOI Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt file or directory containing checkpoints)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="FSOI/configs/fsoi_config.yaml",
        help="Path to FSOI configuration file",
    )
    parser.add_argument(
        "--obs_config",
        type=str,
        default="configs/observation_config.yaml",
        help="Path to observation configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Override start date from config (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Override end date from config (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data directory (default: use train/val global v7 path)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        default=False,
        help="Compute and save innovation diagnostics (mean/std/skewness/RMSE of "
             "xa - xb) per instrument per pair to evaluation/innovation_diagnostics.csv",
    )
    parser.add_argument(
        "--ose_instruments",
        nargs="+",
        default=None,
        metavar="INST",
        help="Run OSE cross-check by replacing xa[X] with xb[X] for these "
             "instruments (e.g. --ose_instruments atms amsua). Also computes "
             "matched conditional endpoint FSOI using the same sampled rows and "
             "the same combined J. Results saved to evaluation/ose_results.csv.",
    )
    parser.add_argument(
        "--verification_target",
        choices=["obs", "mesh"],
        default="obs",
        help="Verification target for FSOI forecast error:\n"
             "  obs  (default): obs-space targets (e.g. radiosonde locations)\n"
             "  mesh: OCELOT icosahedral mesh vs GFS analysis (global reference)\n"
             "Mesh mode requires --gfs_root and --mesh_instrument.",
    )
    parser.add_argument(
        "--gfs_root",
        type=str,
        default="/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25",
        help="Root directory of GFS GRIB2 analysis files (used with --verification_target mesh).",
    )
    parser.add_argument(
        "--mesh_instrument",
        type=str,
        default="radiosonde",
        choices=["radiosonde", "surface_obs"],
        help="Which mesh decoder to use for verification (default: radiosonde).",
    )
    parser.add_argument(
        "--mesh_pressure_level_idx",
        type=int,
        default=4,
        help="Pressure level index 0-15 for radiosonde mesh verification "
             "(0=1000hPa … 4=500hPa … 15=10hPa, default: 4=500hPa).",
    )
    parser.add_argument(
        "--ose_save_spatial_fields",
        action="store_true",
        default=False,
        help="When running OSE with --verification_target mesh, save per-node "
             "control, denied, and full-minus-denied squared-error fields for "
             "selected pairs. Use for physical case-study maps.",
    )
    parser.add_argument(
        "--ose_spatial_pair_indices",
        nargs="*",
        type=int,
        default=None,
        help="Pair indices for --ose_save_spatial_fields. If spatial saving is "
             "enabled without this option, pair 0 is saved.",
    )
    parser.add_argument(
        "--encoding_order",
        type=str,
        default="default",
        choices=["default", "reversed"],
        help=(
            "Observation encoding order into the shared mesh.\n"
            "  default:  YAML config order — satellite instruments first (ATMS, AMSU-A, ...),\n"
            "            then conventional (surface_obs, radiosonde, aircraft last).\n"
            "  reversed: Reverse of config order — conventional first, satellite last.\n"
            "Use 'reversed' to test whether encoding position drives FSOI rankings\n"
            "(ENCODING_ORDER_INVESTIGATION.md, Hypothesis H1). No retraining needed."
        ),
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FSOI INFERENCE - Forecast Sensitivity to Observations Impact")
    print("=" * 80 + "\n")

    # Find checkpoint file (handles both files and directories)
    try:
        checkpoint_path = find_checkpoint(args.checkpoint)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load configurations
    print("Loading configurations...")
    fsoi_config = load_fsoi_config(args.config)
    observation_config, feature_stats, instrument_weights, channel_weights, name_to_id = \
        load_weights_from_yaml(args.obs_config)

    # ── Honor use_instrument_weights / use_channel_weights config flags ───────
    # These flags were defined in all YAML configs but were never read — weights
    # were always applied regardless.  This is now fixed.
    _fc = fsoi_config.get('forecast', {})
    if not _fc.get('use_instrument_weights', True):
        instrument_weights = {}
        print("[WEIGHTS] use_instrument_weights=false: uniform instrument weights")
    else:
        print(f"[WEIGHTS] use_instrument_weights=true: {len(instrument_weights)} instrument weights loaded")
    if not _fc.get('use_channel_weights', True):
        channel_weights = {}
        print("[WEIGHTS] use_channel_weights=false: uniform channel weights")
    else:
        print(f"[WEIGHTS] use_channel_weights=true: {len(channel_weights)} channel weight sets loaded")

    # Override config with command-line args
    if args.output_dir:
        fsoi_config['data']['output_dir'] = args.output_dir
    if args.start_date:
        fsoi_config['data']['start_date'] = args.start_date
    if args.end_date:
        fsoi_config['data']['end_date'] = args.end_date

    # Setup output directory
    output_path = setup_output_directory(fsoi_config['data']['output_dir'])
    print(f"Output directory: {output_path}")

    ose_spatial_output_dir = None
    ose_spatial_pair_indices = None
    if args.ose_save_spatial_fields:
        ose_spatial_output_dir = str(output_path / "evaluation" / "ose_spatial_fields")
        if args.ose_spatial_pair_indices:
            ose_spatial_pair_indices = set(args.ose_spatial_pair_indices)
        else:
            ose_spatial_pair_indices = {0}
        print(f"[OSE] Spatial field saving enabled for pairs: "
              f"{sorted(ose_spatial_pair_indices)}")
        print(f"[OSE] Spatial field output: {ose_spatial_output_dir}")

    # Save configuration
    config_save_path = output_path / "logs" / "fsoi_config_used.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(fsoi_config, f)
    print(f"Configuration saved to: {config_save_path}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    model = GNNLightning.load_from_checkpoint(checkpoint_path)
    model.to(device)

    # Set encoding order BEFORE freezing — this is a runtime-only attribute,
    # not a trained parameter. It changes the order instruments write to the
    # shared mesh during the forward pass without altering any weights.
    model.encoding_order = args.encoding_order
    if args.encoding_order != "default":
        print(f"\n[ENCODING ORDER] Using '{args.encoding_order}' encoding order.")
        print("  This is the H1 test from ENCODING_ORDER_INVESTIGATION.md:")
        print("  If FSOI rankings shift substantially, encoding position is a confound.")
        if args.encoding_order == "reversed":
            print("  Reversed: conventional obs encode FIRST, satellites encode LAST.")
    else:
        print("\n[ENCODING ORDER] Using default (config YAML) encoding order.")
        print("  Satellite instruments encode first, conventional obs encode last.")

    # Freeze model for FSOI
    freeze_model_for_fsoi(model)

    # CRITICAL FIX: Preserve both ID mappings instead of overwriting
    # Model checkpoint may have learned parameters indexed by instrument ID
    # (e.g., instrument embeddings). Overwriting breaks those.
    #
    # Solution: Keep both mappings:
    #   - model.instrument_name_to_id_ckpt: Original from checkpoint (for model internals)
    #   - weights_name_to_id: From YAML (for weights lookup)
    #
    # Then inside compute_forecast_error(), use weights_name_to_id for weights only.

    print(f"\n[ID MAPPING] Checking instrument ID consistency...")

    # Preserve original checkpoint mapping
    model.instrument_name_to_id_ckpt = dict(model.instrument_name_to_id)

    # Create separate weights mapping from YAML
    weights_name_to_id = name_to_id

    # Check for mismatches (diagnostic only)
    mismatches = []
    for inst_name in sorted(set(model.instrument_name_to_id_ckpt.keys()) & set(weights_name_to_id.keys())):
        if model.instrument_name_to_id_ckpt[inst_name] != weights_name_to_id[inst_name]:
            mismatches.append(
                f"  {inst_name}: ckpt_id={model.instrument_name_to_id_ckpt[inst_name]} "
                f"vs yaml_id={weights_name_to_id[inst_name]}"
            )

    if mismatches:
        print(f"[INFO] Found {len(mismatches)} ID mapping differences between checkpoint and YAML:")
        for m in mismatches[:5]:
            print(m)
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")
        print("[SOLUTION] Using checkpoint IDs for model, YAML IDs for weights (safe)")
    else:
        print(f"[OK] Checkpoint and YAML ID mappings are consistent")

    # Store weights mapping on model for use in compute_forecast_error()
    model.weights_name_to_id = weights_name_to_id

    print(f"[ID MAPPING] Checkpoint: {len(model.instrument_name_to_id_ckpt)} instruments")
    print(f"[ID MAPPING] Weights: {len(weights_name_to_id)} instruments")

    print(f"Model loaded and frozen for FSOI")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Mesh type: {model.mesh_type}")
    print(f"  Instruments: {list(model.instrument_name_to_id.keys())}")

    # Create datamodule
    print("\nSetting up data...")

    # Determine data path (same logic as train_gnn.py)
    region = "global"
    if args.data_path:
        data_path = args.data_path
    elif region == "conus":
        data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
    else:
        data_path = "/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7"

    print(f"Data path: {data_path}")

    # Create datamodule for accessing data
    datamodule = GNNDataModule(
        data_path=data_path,
        start_date=fsoi_config['data']['start_date'],
        end_date=fsoi_config['data']['end_date'],
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=1,  # Must be 1 for FSOI
        feature_stats=feature_stats,
        num_neighbors=3,
        window_size="12h",
        pipeline=None,  # No special pipeline for FSOI - use default processing
    )

    # Setup data
    datamodule.setup(stage='test')

    # Create FSOI dataset (sequential pairs)
    print("\nCreating FSOI sequential dataset...")

    # Get bin names for FSOI period
    from process_timeseries import organize_bins_times
    fsoi_summary = organize_bins_times(
        datamodule.z,
        fsoi_config['data']['start_date'],
        fsoi_config['data']['end_date'],
        observation_config,
        pipeline_cfg={},
        window_size="12h",
    )

    fsoi_bin_names = sorted(fsoi_summary.keys())
    print(f"Found {len(fsoi_bin_names)} bins for FSOI computation")

    # Verify sequential consistency
    verify_sequential_consistency(fsoi_bin_names, expected_interval_hours=12)

    # Create base dataset using datamodule's graph creation method
    def create_graph_fn(bin_data):
        return datamodule._create_graph_structure(bin_data)

    base_dataset = BinDataset(
        bin_names=fsoi_bin_names,
        data_summary=fsoi_summary,
        zarr_store=datamodule.z,
        create_graph_fn=create_graph_fn,
        observation_config=observation_config,
        feature_stats=feature_stats,
        tag="FSOI",
    )

    # Create FSOI paired dataset
    fsoi_dataset = FSOIDataset(
        base_dataset=base_dataset,
        bin_names=fsoi_bin_names,
    )

    # Create dataloader (NO SHUFFLE!)
    fsoi_loader = PyGDataLoader(
        fsoi_dataset,
        batch_size=1,
        shuffle=False,  # CRITICAL: must be sequential
        num_workers=0,  # Single worker for stability
    )

    print(f"FSOI dataloader created with {len(fsoi_loader)} pairs")
    min_pairs_warning = int(fsoi_config.get('validation', {}).get('min_pairs_warning', 20))
    if len(fsoi_loader) < min_pairs_warning:
        print(
            f"[WARNING] Only {len(fsoi_loader)} sequential FSOI pairs. "
            f"Use >= {min_pairs_warning} pairs for stable instrument rankings."
        )

    # ── Finite-difference pair schedule ──────────────────────────────────────
    # Determine which pair indices get a FD gradient check.  The goal is to
    # cover at least 3 distinct dates with ~5 samples each (15 total), spread
    # across early, middle, and late in the FSOI window.
    n_pairs = len(fsoi_loader)
    fd_enabled = fsoi_config['validation'].get('finite_difference_check', False)
    if fd_enabled:
        fd_pairs_cfg = fsoi_config['validation'].get('fd_check_pairs', None)
        if fd_pairs_cfg is not None:
            fd_pair_set = set(int(p) for p in fd_pairs_cfg)
        else:
            # Auto: three equally-spaced pairs (early / middle / late)
            pts = [0, max(0, n_pairs // 3), max(0, 2 * n_pairs // 3)]
            fd_pair_set = set(pts)
        print(f"[FD] Will run gradient validation on pair indices: "
              f"{sorted(fd_pair_set)} (total dates: {len(fd_pair_set)})")
    else:
        fd_pair_set = set()

    # Directional-derivative test (Rademacher, float32) — for high-N satellites
    directional_enabled = fsoi_config['validation'].get('directional_derivative_check', False)
    directional_pair_set = fd_pair_set if directional_enabled else set()
    if directional_enabled:
        print("[FD] Directional-derivative check enabled (Rademacher, float32) — "
              "runs on same pair indices as scalar FD")

    # Float64 per-obs FD test — for high-N satellites
    float64_enabled = fsoi_config['validation'].get('float64_fd_check', False)
    float64_pair_set = fd_pair_set if float64_enabled else set()
    if float64_enabled:
        print("[FD] Float64 FD check enabled — runs on same pair indices as scalar FD")

    run_diagnostics = bool(args.diagnostics)
    if run_diagnostics:
        print("[DIAG] Innovation diagnostics enabled — saving to evaluation/innovation_diagnostics.csv")

    repro_enabled = fsoi_config['validation'].get('check_reproducibility', False)
    if repro_enabled:
        print("[REPRO] Reproducibility check enabled — will re-run pair 0 forward+backward")

    # ── Mesh-space verification setup ─────────────────────────────────────────
    use_mesh_verification = (args.verification_target == "mesh")
    mesh_pred_edges_cache = None
    mesh_lats = mesh_lons = None

    if use_mesh_verification:
        import numpy as np
        from FSOI.fsoi_gfs_loader import load_gfs_on_mesh
        from fsoi_utils import compute_forecast_error_on_mesh

        print(f"\n[MESH FSOI] Mesh-space verification enabled")
        print(f"  Mesh instrument   : {args.mesh_instrument}")
        hpa = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
        print(f"  Pressure level    : idx={args.mesh_pressure_level_idx} "
              f"({hpa[args.mesh_pressure_level_idx]} hPa)")
        print(f"  GFS root          : {args.gfs_root}")

        # Enable mesh prediction on the model
        model.enable_mesh_pred = True
        model.mesh_pressure_level_idx = args.mesh_pressure_level_idx

        # Load mesh prediction edges once (expensive, cached on model)
        mesh_pred_edges_cache = model._get_mesh_pred_edges()
        if args.mesh_instrument not in mesh_pred_edges_cache:
            raise ValueError(
                f"[MESH FSOI] No edges for '{args.mesh_instrument}'. "
                f"Available: {list(mesh_pred_edges_cache.keys())}. "
                "Run precompute_mesh_edges.py if needed."
            )
        edges = mesh_pred_edges_cache[args.mesh_instrument]
        mesh_lats = edges['lats']  # [N_mesh] numpy array
        mesh_lons = edges['lons']  # [N_mesh] numpy array
        print(f"  Mesh nodes        : {len(mesh_lats):,} global icosahedral nodes")

        if not os.path.isdir(args.gfs_root):
            raise FileNotFoundError(
                f"[MESH FSOI] GFS root not found: {args.gfs_root}. "
                "Set via --gfs_root."
            )

    # OSE instruments: expand seviri aliases and validate
    ose_instruments = None
    if args.ose_instruments:
        ose_instruments = _expand_seviri_instrument_aliases(args.ose_instruments)
        print(f"[OSE] Observation-withholding experiment enabled for: {ose_instruments}")
        print("[OSE] Adds 1 no-grad forward pass per pair per denial group.")

    # Main FSOI computation loop
    print("\n" + "=" * 80)
    print("Starting FSOI Computation")
    print("=" * 80 + "\n")

    all_results = []
    scatter_frames = []
    all_fd_records = []
    all_directional_records = []
    all_float64_records = []
    all_diag_records = []
    all_ose_records = []

    for pair_idx, (prev_batch, curr_batch) in enumerate(tqdm(fsoi_loader, desc="Computing FSOI")):
        try:
            # ── Load GFS reference for mesh-space verification ───────────────
            gfs_reference_pair = None
            if use_mesh_verification:
                from FSOI.fsoi_gfs_loader import load_gfs_on_mesh
                from datetime import datetime, timezone
                import numpy as np

                # Parse valid time from curr_bin (end of current observation window)
                curr_bin_name = _as_scalar_bin(curr_batch.bin_name)
                try:
                    if str(curr_bin_name).startswith('bin') and len(str(curr_bin_name)) >= 13:
                        ts_str = str(curr_bin_name)[3:13]  # YYYYMMDDHH
                        valid_dt = datetime.strptime(ts_str, "%Y%m%d%H")
                    else:
                        valid_dt = pd.to_datetime(str(curr_bin_name)).to_pydatetime()

                    gfs_tensor = load_gfs_on_mesh(
                        valid_dt=valid_dt,
                        mesh_lats=mesh_lats,
                        mesh_lons=mesh_lons,
                        inst_name=args.mesh_instrument,
                        feature_stats=feature_stats,
                        gfs_root=args.gfs_root,
                        pressure_level_idx=args.mesh_pressure_level_idx,
                        device=device,
                    )
                    gfs_reference_pair = {args.mesh_instrument: gfs_tensor}
                except Exception as gfs_err:
                    print(f"[MESH FSOI] GFS load failed for pair {pair_idx}: {gfs_err}")
                    print("  Falling back to obs-space verification for this pair")
                    gfs_reference_pair = None

            # Compute FSOI for this pair
            result = compute_fsoi_for_pair(
                model=model,
                prev_batch=prev_batch,
                curr_batch=curr_batch,
                fsoi_config=fsoi_config,
                observation_config=observation_config,
                instrument_weights=instrument_weights,
                channel_weights=channel_weights,
                pair_idx=pair_idx,
                verbose=fsoi_config['validation'].get('verbose', True),
                run_fd_check=(pair_idx in fd_pair_set),
                run_directional_check=(pair_idx in directional_pair_set),
                run_float64_check=(pair_idx in float64_pair_set),
                run_diagnostics=run_diagnostics,
                run_repro_check=(repro_enabled and pair_idx == 0),
                ose_instruments=ose_instruments,
                gfs_reference=gfs_reference_pair,
                mesh_instrument=args.mesh_instrument if use_mesh_verification else "radiosonde",
                mesh_pressure_level_idx=args.mesh_pressure_level_idx if use_mesh_verification else 4,
                ose_spatial_output_dir=ose_spatial_output_dir,
                ose_spatial_pair_indices=ose_spatial_pair_indices,
            )

            all_results.append(result)

            if isinstance(result, dict) and result.get('scatter_samples'):
                scatter_frames.extend(result['scatter_samples'])
            if isinstance(result, dict) and result.get('fd_records'):
                all_fd_records.extend(result['fd_records'])
            if isinstance(result, dict) and result.get('directional_records'):
                all_directional_records.extend(result['directional_records'])
            if isinstance(result, dict) and result.get('float64_records'):
                all_float64_records.extend(result['float64_records'])
            if isinstance(result, dict) and result.get('diag_records'):
                all_diag_records.extend(result['diag_records'])
            if isinstance(result, dict) and result.get('ose_records'):
                all_ose_records.extend(result['ose_records'])

        except Exception as e:
            print(f"\n[ERROR] Failed to compute FSOI for pair {pair_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Reclaim GPU memory after a failure (e.g. CUDA OOM) so fragmentation
            # does not cascade into subsequent pairs.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    print(f"\n✓ FSOI computation complete for {len(all_results)} pairs")

    # Aggregate results
    print("\n" + "=" * 80)
    print("Aggregating FSOI Results")
    print("=" * 80 + "\n")

    # Aggregate across all pairs and steps
    aggregated_by_channel = []
    aggregated_by_instrument = []

    for result in all_results:
        for lead_step, step_data in result['fsoi_by_step'].items():

            if step_data.get('pressure_stratified'):
                # ── Pressure-stratified path ──────────────────────────────
                # Each level gets its own aggregate_fsoi_by_channel call with
                # a single-level metadata so all instruments get pressure_hpa.
                base_metadata = step_data.get('metadata', {}) or {}

                variable_stratified = bool(step_data.get('variable_stratified', False))

                for level_data in step_data['per_level']:
                    metric_ea = level_data.get('ea_p', step_data['ea'])
                    metric_eb = level_data.get('eb_p', step_data['eb'])
                    if 'fsoi_channel_aggregates' in level_data:
                        df_ch = level_data['fsoi_channel_aggregates'].copy()
                    else:
                        # Build single-level metadata: broadcast this one p_idx/hPa
                        # to all instruments via the _target_pressure_level fallback.
                        meta_p = dict(base_metadata)
                        meta_p['_target_pressure_level'] = torch.tensor([level_data['p_idx']])
                        meta_p['_target_pressure_hpa'] = torch.tensor([level_data['p_hpa']])

                        df_ch = aggregate_fsoi_by_channel(
                            level_data['fsoi_values'],
                            model.instrument_name_to_id,
                            metadata=meta_p,
                            innovations=level_data.get('innovations'),
                            gradient_sums=level_data.get('gradient_sums'),
                            sampling_info=step_data.get('sampling_info'),
                        )
                        # Preserve target identity if present
                        if 'target_variable' in level_data:
                            df_ch['target_variable'] = level_data.get('target_variable')
                        if 'target_channel' in level_data:
                            df_ch['target_channel'] = level_data.get('target_channel')
                        df_ch['p_idx'] = level_data.get('p_idx')
                        df_ch['p_hpa'] = level_data.get('p_hpa')

                    df_ch['pair_idx'] = result['pair_idx']
                    df_ch['prev_bin'] = result['prev_bin']
                    df_ch['curr_bin'] = result['curr_bin']
                    df_ch['lead_step'] = lead_step
                    df_ch['ea'] = metric_ea
                    df_ch['eb'] = metric_eb
                    df_ch['ea_p'] = metric_ea
                    df_ch['eb_p'] = metric_eb
                    df_ch['ea_total'] = step_data['ea']
                    df_ch['eb_total'] = step_data['eb']
                    aggregated_by_channel.append(df_ch)

                if variable_stratified:
                    # Instrument aggregates already represent a specific (p, variable) metric.
                    for level_data in step_data['per_level']:
                        if 'fsoi_instrument_aggregates' not in level_data:
                            continue
                        metric_ea = level_data.get('ea_p', step_data['ea'])
                        metric_eb = level_data.get('eb_p', step_data['eb'])
                        df_inst = level_data['fsoi_instrument_aggregates'].copy()
                        df_inst['pair_idx'] = result['pair_idx']
                        df_inst['prev_bin'] = result['prev_bin']
                        df_inst['curr_bin'] = result['curr_bin']
                        df_inst['lead_step'] = lead_step
                        df_inst['ea'] = metric_ea
                        df_inst['eb'] = metric_eb
                        df_inst['ea_p'] = metric_ea
                        df_inst['eb_p'] = metric_eb
                        df_inst['ea_total'] = step_data['ea']
                        df_inst['eb_total'] = step_data['eb']
                        aggregated_by_instrument.append(df_inst)
                else:
                    # For instrument-level: sum FSOI across all pressure levels.
                    combined_fsoi = {}
                    combined_innov = {}
                    combined_gsum = {}
                    for level_data in step_data['per_level']:
                        for inst, v in level_data['fsoi_values'].items():
                            if inst not in combined_fsoi:
                                combined_fsoi[inst] = v.clone()
                                combined_innov[inst] = level_data['innovations'].get(inst)
                                combined_gsum[inst] = level_data['gradient_sums'].get(inst)
                            else:
                                combined_fsoi[inst] = combined_fsoi[inst] + v

                    df_inst = aggregate_fsoi_by_instrument(
                        combined_fsoi,
                        model.instrument_name_to_id,
                        innovations=combined_innov,
                        gradient_sums=combined_gsum,
                        sampling_info=step_data.get('sampling_info'),
                    )
                    df_inst['pair_idx'] = result['pair_idx']
                    df_inst['prev_bin'] = result['prev_bin']
                    df_inst['curr_bin'] = result['curr_bin']
                    df_inst['lead_step'] = lead_step
                    df_inst['ea'] = step_data['ea']
                    df_inst['eb'] = step_data['eb']
                    aggregated_by_instrument.append(df_inst)

            else:
                # ── Standard (non-stratified) path ────────────────────────
                fsoi_values = step_data['fsoi_values']
                innovations = step_data.get('innovations', None)
                gradient_sums = step_data.get('gradient_sums', None)
                metadata = step_data.get('metadata', None)

                df_channel = aggregate_fsoi_by_channel(
                    fsoi_values,
                    model.instrument_name_to_id,
                    metadata=metadata,
                    innovations=innovations,
                    gradient_sums=gradient_sums,
                    sampling_info=step_data.get('sampling_info'),
                )
                df_channel['pair_idx'] = result['pair_idx']
                df_channel['prev_bin'] = result['prev_bin']
                df_channel['curr_bin'] = result['curr_bin']
                df_channel['lead_step'] = lead_step
                df_channel['ea'] = step_data['ea']
                df_channel['eb'] = step_data['eb']
                aggregated_by_channel.append(df_channel)

                df_inst = aggregate_fsoi_by_instrument(
                    fsoi_values,
                    model.instrument_name_to_id,
                    innovations=innovations,
                    gradient_sums=gradient_sums,
                    sampling_info=step_data.get('sampling_info'),
                )
                df_inst['pair_idx'] = result['pair_idx']
                df_inst['prev_bin'] = result['prev_bin']
                df_inst['curr_bin'] = result['curr_bin']
                df_inst['lead_step'] = lead_step
                df_inst['ea'] = step_data['ea']
                df_inst['eb'] = step_data['eb']
                aggregated_by_instrument.append(df_inst)

    # Combine into DataFrames with empty-result guards
    if len(aggregated_by_channel) == 0:
        print("[WARNING] No FSOI results to aggregate (all pairs failed). Creating empty output.")
        df_all_channel = pd.DataFrame()
    else:
        df_all_channel = pd.concat(aggregated_by_channel, ignore_index=True)

    if len(aggregated_by_instrument) == 0:
        print("[WARNING] No FSOI results to aggregate (all pairs failed). Creating empty output.")
        df_all_instrument = pd.DataFrame()
    else:
        df_all_instrument = pd.concat(aggregated_by_instrument, ignore_index=True)

    # Save results
    print("\nSaving results...")

    if fsoi_config['output'].get('save_csv', True):
        csv_dir = output_path / "csv"

        # Per-channel results
        if not df_all_channel.empty:
            channel_csv = csv_dir / "fsoi_by_channel.csv"
            df_all_channel.to_csv(channel_csv, index=False)
            print(f"  Channel-level FSOI: {channel_csv}")
        else:
            print("  [SKIPPED] Channel-level FSOI: No data to save")

        # Per-instrument results
        if not df_all_instrument.empty:
            inst_csv = csv_dir / "fsoi_by_instrument.csv"
            df_all_instrument.to_csv(inst_csv, index=False)
            print(f"  Instrument-level FSOI: {inst_csv}")
        else:
            print("  [SKIPPED] Instrument-level FSOI: No data to save")

        # Summary statistics (only if we have data)
        if not df_all_instrument.empty and 'instrument' in df_all_instrument.columns:
            summary_csv = csv_dir / "fsoi_summary.csv"
            summary_aggs = {
                'sum_impact': ['mean', 'std', 'sum'],
                'mean_impact': ['mean', 'std'],
                'positive_frac': 'mean',
            }
            if 'sum_impact_scaled' in df_all_instrument.columns:
                summary_aggs['sum_impact_scaled'] = ['mean', 'std', 'sum']
            if 'raw_n_observations' in df_all_instrument.columns:
                summary_aggs['raw_n_observations'] = 'sum'
            if 'sample_scale' in df_all_instrument.columns:
                summary_aggs['sample_scale'] = 'mean'

            optional_aggs = {
                'innovation_abs_mean': 'mean',
                'innovation_rms': 'mean',
                'gradient_abs_mean': 'mean',
                'gradient_rms': 'mean',
                'alignment_cosine': 'mean',
                'alignment_frac': 'mean',
            }

            for col, agg in optional_aggs.items():
                if col in df_all_instrument.columns:
                    summary_aggs[col] = agg

            summary = df_all_instrument.groupby('instrument').agg(summary_aggs).reset_index()
            # Flatten multi-index columns for CSV friendliness
            summary.columns = [
                '_'.join([c for c in col if c]).strip('_') if isinstance(col, tuple) else col
                for col in summary.columns.values
            ]
            summary.to_csv(summary_csv, index=False)
            print(f"  Summary statistics: {summary_csv}")
        else:
            print("  [SKIPPED] Summary statistics: No data to aggregate")

        # Innovation vs FSOI scatter samples
        if scatter_frames and fsoi_config.get('plots', {}).get('save_scatter_samples', True):
            scatter_csv = csv_dir / "scatter_samples.csv"
            df_scatter = pd.concat(scatter_frames, ignore_index=True)
            df_scatter.to_csv(scatter_csv, index=False)
            print(f"  Scatter samples: {scatter_csv}")
        else:
            print("  [SKIPPED] Scatter samples: None collected")

    # ── Write OSE cross-check results ────────────────────────────────────
    if all_ose_records:
        from fsoi_ose import compare_ose_vs_fsoi
        eval_dir = output_path / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        ose_df = pd.DataFrame(all_ose_records)
        ose_csv = eval_dir / "ose_results.csv"
        ose_df.to_csv(ose_csv, index=False)
        print(f"\n[OSE] Saved {len(ose_df)} OSE records to {ose_csv}")

        # Immediate comparison with FSOI if instrument CSV exists
        fsoi_inst_csv = output_path / "csv" / "fsoi_by_instrument.csv"
        if fsoi_inst_csv.is_file():
            comp = compare_ose_vs_fsoi(ose_csv, fsoi_inst_csv)
            if not comp.empty:
                comp_csv = eval_dir / "ose_vs_fsoi_comparison.csv"
                comp.to_csv(comp_csv, index=False)
                valid = comp.dropna(subset=["closure_ratio"])
                if not valid.empty:
                    med_r = float(valid["closure_ratio"].median())
                    sign_pct = float(valid["sign_agree"].mean() * 100)
                    r_corr = float(valid[["ose_impact", "fsoi_predicted"]]
                                   .corr().iloc[0, 1]) if len(valid) > 2 else float("nan")
                    print(f"[OSE] Comparison: median closure_ratio={med_r:.3f} "
                          f"(target 1.0)  sign_agree={sign_pct:.0f}%  "
                          f"Pearson r={r_corr:.3f}")
                    if abs(med_r - 1.0) > 0.30:
                        print("[OSE] WARNING: closure_ratio far from 1.0 — "
                              "nonlinear GNN effects may be significant for "
                              f"{ose_instruments}")
                print(f"[OSE] Comparison saved to {comp_csv}")

    # ── Write reproducibility result ─────────────────────────────────────
    if repro_enabled and all_results:
        r0 = all_results[0]
        repro_rec = {
            'pair_idx': 0,
            'curr_bin': r0.get('curr_bin', ''),
            'status': r0.get('repro_status', 'NOT_RUN'),
            'ea_diff': r0.get('repro_ea_diff', float('nan')),
            'max_ga_diff': r0.get('repro_max_ga_diff', float('nan')),
        }
        eval_dir = output_path / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([repro_rec]).to_csv(eval_dir / "reproducibility_check.csv", index=False)
        print(f"\n[REPRO] Result: {repro_rec['status']} "
              f"(ea_diff={repro_rec['ea_diff']:.2e}, "
              f"ga_diff={repro_rec['max_ga_diff']:.2e})")

    # ── Write accumulated FD validation records ───────────────────────────
    if all_fd_records:
        eval_dir = output_path / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        fd_csv = eval_dir / "fd_validation.csv"
        pd.DataFrame(all_fd_records).to_csv(fd_csv, index=False)
        n_pass = sum(1 for r in all_fd_records if r.get('status') == 'PASS')
        n_warn = sum(1 for r in all_fd_records if r.get('status') == 'WARNING')
        n_fail = sum(1 for r in all_fd_records if r.get('status') == 'FAIL')
        print(f"\n[FD] Saved {len(all_fd_records)} scalar FD records to {fd_csv}")
        print(f"     PASS={n_pass}  WARN={n_warn}  FAIL={n_fail} "
              f"(across {len(fd_pair_set)} dates)")
        if n_fail > 0:
            print("[FD] WARNING: some gradient checks failed — review fd_validation.csv")

    # ── Write directional-derivative records ──────────────────────────────
    if all_directional_records:
        eval_dir = output_path / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        dir_csv = eval_dir / "fd_directional_validation.csv"
        pd.DataFrame(all_directional_records).to_csv(dir_csv, index=False)
        n_pass = sum(1 for r in all_directional_records if r.get('status') == 'PASS')
        n_warn = sum(1 for r in all_directional_records if r.get('status') == 'WARN')
        n_fail = sum(1 for r in all_directional_records if r.get('status') == 'FAIL')
        n_insuf = sum(1 for r in all_directional_records if r.get('status') == 'INSUF_SIGNAL')
        print(f"\n[FD] Saved {len(all_directional_records)} directional-deriv records to {dir_csv}")
        print(f"     PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  INSUF_SIGNAL={n_insuf}")
        if n_fail > 0:
            print("[FD] WARNING: directional gradient direction test failed — "
                  "review fd_directional_validation.csv")

    # ── Write float64 FD records ──────────────────────────────────────────
    if all_float64_records:
        eval_dir = output_path / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        f64_csv = eval_dir / "fd_float64_validation.csv"
        pd.DataFrame(all_float64_records).to_csv(f64_csv, index=False)
        n_pass = sum(1 for r in all_float64_records if r.get('status') == 'PASS')
        n_warn = sum(1 for r in all_float64_records if r.get('status') == 'WARN')
        n_fail = sum(1 for r in all_float64_records if r.get('status') == 'FAIL')
        print(f"\n[FD] Saved {len(all_float64_records)} float64 FD records to {f64_csv}")
        print(f"     PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}")
        if n_fail > 0:
            print("[FD] WARNING: float64 gradient check failed — review fd_float64_validation.csv")

    # ── Write innovation diagnostics ──────────────────────────────────────
    if all_diag_records:
        eval_dir = output_path / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        diag_csv = eval_dir / "innovation_diagnostics.csv"
        df_diag = pd.DataFrame(all_diag_records)
        df_diag.to_csv(diag_csv, index=False)
        print(f"\n[DIAG] Saved innovation diagnostics to {diag_csv}")
        # Quick quality summary
        bad_bg = df_diag[df_diag['normalized_rmse'] > 0.20]
        if not bad_bg.empty:
            bad_insts = sorted(bad_bg['instrument'].unique())
            print(f"[DIAG] WARNING: {len(bad_bg)} (instrument×channel×pair) entries "
                  f"have normalized_rmse > 20%: {bad_insts}")
            print("       xb background forecast quality may be poor for these instruments.")
        good = df_diag[df_diag['normalized_rmse'] <= 0.05]
        print(f"[DIAG] {len(good)}/{len(df_diag)} entries have normalized_rmse ≤ 5% "
              f"(good background quality)")

    print("\n" + "=" * 80)
    print("FSOI Inference Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print(f"Total pairs processed: {len(all_results)}")

    # Only print stats if we have data
    if not df_all_instrument.empty and 'instrument' in df_all_instrument.columns:
        print(f"Total instruments: {df_all_instrument['instrument'].nunique()}")
        impact_col = 'sum_impact_scaled' if 'sum_impact_scaled' in df_all_instrument.columns else 'sum_impact'
        print(f"\nTop 5 instruments by mean {impact_col}:")
        print(df_all_instrument.groupby('instrument')[impact_col].mean().sort_values(ascending=False).head())
    else:
        print("\n[WARNING] No FSOI data computed - check logs for errors")


if __name__ == "__main__":
    main()
