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
        print(f"\n{'='*80}")
        print(f"Computing FSOI for pair {pair_idx}")
        # Note: bin_name will be unwrapped to scalar in results dict
        print(f"Previous: {prev_batch.bin_name} (type: {type(prev_batch.bin_name)})")
        print(f"Current:  {curr_batch.bin_name} (type: {type(curr_batch.bin_name)})")
        print(f"{'='*80}\n")

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

        # Feature masks applied at inference (e.g., zero aircraft humidity)
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

        # Optionally zero specific channels (e.g., aircraft humidity) at inference time
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

        # Optional: Finite-difference gradient validation
        if fsoi_config['validation'].get('finite_difference_check', False) and pair_idx == 0:
            print("\n[VALIDATION] Running finite-difference gradient check...")
            from fsoi_validation import validate_fsoi_gradients

            num_samples = fsoi_config['validation'].get('fd_num_samples', 3)
            epsilon = fsoi_config['validation'].get('fd_epsilon', 1e-4)

            fd_passed = validate_fsoi_gradients(
                model,
                prev_batch,
                curr_batch,
                num_samples=num_samples,
                epsilon=epsilon,
            )

            if not fd_passed:
                print("[WARNING] Finite-difference validation failed!")
                print("  Gradients may be incorrect. Review implementation.")
                if fsoi_config['validation'].get('require_fd_pass', False):
                    print("[ERROR] Stopping due to failed validation")
                    return results
            else:
                print("[VALIDATION] Finite-difference check PASSED ✓")

        # Whether to compute a separate loss per radiosonde pressure level
        # (enables pressure_hpa column for ALL instruments including satellites)
        stratify_by_pressure = fsoi_config['forecast'].get('stratify_by_pressure', False)

        if stratify_by_pressure:
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
                )

                per_level_results = []
                scatter_saved = False
                for mr in raw_results:
                    # Broadcast metric target pressure for all instruments
                    meta_m = dict(metadata)
                    meta_m['_target_pressure_level'] = torch.tensor([mr['p_idx']])
                    meta_m['_target_pressure_hpa'] = torch.tensor([mr['p_hpa']])

                    df_ch = aggregate_fsoi_by_channel(
                        mr['fsoi_values'],
                        model.instrument_name_to_id,
                        metadata=meta_m,
                        innovations=mr.get('innovations'),
                        gradient_sums=mr.get('gradient_sums'),
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
                        # Save ONE representative metric’s scatter sample per (pair, lead_step)
                        # to avoid huge files.
                        scatter_df = sample_innovation_vs_fsoi(
                            mr['fsoi_values'],
                            mr.get('innovations') or {},
                            max_points=scatter_max_points,
                            seed=pair_idx * 1000 + int(lead_step),
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
            }

        else:
            # ========================================
            # STEP 4: Compute analysis adjoint (ga)
            # ========================================
            print("[4/6] Computing analysis adjoint (ga)...")

            # Replace batch inputs with xa (indexed for subsampled instruments)
            curr_batch_xa = curr_batch.clone()
            replace_batch_inputs(curr_batch_xa, xa, observation_config,
                                 replace_indices=subsample_indices)

            # Compute forecast error for analysis
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
            )

            print(f"  Analysis error (ea): {ea.item():.6e}")

            # Clear CUDA cache after forward pass
            torch.cuda.empty_cache()
            print(f"[MEMORY] Cleared CUDA cache after ea computation")

            # EARLY CHECK: No verification targets
            if ea.item() == 0.0:
                print("\n" + "="*80)
                print("SKIPPING FSOI PAIR: No verification targets found")
                print("="*80)
                print(f"  Lead step: {lead_step}")
                print(f"  Target instruments: {target_instruments}")
                print(f"  Target variables: {target_variables}")
                print(f"  Target pressure levels: {target_pressure_levels}")
                print(f"  This is normal if verification data is sparse or filtered.")
                print(f"  Gradients will be zero, but this is NOT a gradient error.")
                print("="*80 + "\n")
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

            # Compute forecast error for background
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
                print(f"\n{'='*80}")
                print("FATAL: Gradient validation FAILED!")
                print(f"{'='*80}")
                print(str(e))
                print(f"\nSkipping FSOI pair: pair_idx={pair_idx}")
                print(f"{'='*80}\n")
                continue  # Skip this FSOI pair

            # ========================================
            # STEP 6: Compute FSOI
            # ========================================
            print("[6/6] Computing FSOI = δx ⊙ (ga + gb)...")

            # xa and xb are shape-aligned after the ALIGNMENT block above
            fsoi_values, innovations, gradient_sums = compute_fsoi_per_observation(
                xa, xb, ga, gb, return_components=True
            )

            if save_scatter_samples and scatter_max_points > 0:
                scatter_df = sample_innovation_vs_fsoi(
                    fsoi_values,
                    innovations,
                    max_points=scatter_max_points,
                    seed=pair_idx * 1000 + int(lead_step),
                )
                if not scatter_df.empty:
                    scatter_df['pair_idx'] = pair_idx
                    scatter_df['lead_step'] = lead_step
                    results['scatter_samples'].append(scatter_df)

        # Memory control: only store full tensors if enabled (non-stratified path)
        store_full_tensors = fsoi_config['data'].get('store_full_tensors', False)

        if stratify_by_pressure:
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
            )

            inst_agg = aggregate_fsoi_by_instrument(
                fsoi_values,
                model.instrument_name_to_id,  # Use model mapping for aggregation
                innovations=innovations,
                gradient_sums=gradient_sums,
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

    args = parser.parse_args()

    print("\n" + "="*80)
    print("FSOI INFERENCE - Forecast Sensitivity to Observations Impact")
    print("="*80 + "\n")

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

    # Main FSOI computation loop
    print("\n" + "="*80)
    print("Starting FSOI Computation")
    print("="*80 + "\n")

    all_results = []
    scatter_frames = []

    for pair_idx, (prev_batch, curr_batch) in enumerate(tqdm(fsoi_loader, desc="Computing FSOI")):
        try:
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
            )

            all_results.append(result)

            if isinstance(result, dict) and result.get('scatter_samples'):
                scatter_frames.extend(result['scatter_samples'])

        except Exception as e:
            print(f"\n[ERROR] Failed to compute FSOI for pair {pair_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ FSOI computation complete for {len(all_results)} pairs")

    # Aggregate results
    print("\n" + "="*80)
    print("Aggregating FSOI Results")
    print("="*80 + "\n")

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
                    df_ch['ea'] = step_data['ea']
                    df_ch['eb'] = step_data['eb']
                    aggregated_by_channel.append(df_ch)

                if variable_stratified:
                    # Instrument aggregates already represent a specific (p, variable) metric.
                    for level_data in step_data['per_level']:
                        if 'fsoi_instrument_aggregates' not in level_data:
                            continue
                        df_inst = level_data['fsoi_instrument_aggregates'].copy()
                        df_inst['pair_idx'] = result['pair_idx']
                        df_inst['prev_bin'] = result['prev_bin']
                        df_inst['curr_bin'] = result['curr_bin']
                        df_inst['lead_step'] = lead_step
                        df_inst['ea'] = step_data['ea']
                        df_inst['eb'] = step_data['eb']
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

    print("\n" + "="*80)
    print("FSOI Inference Complete")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print(f"Total pairs processed: {len(all_results)}")

    # Only print stats if we have data
    if not df_all_instrument.empty and 'instrument' in df_all_instrument.columns:
        print(f"Total instruments: {df_all_instrument['instrument'].nunique()}")
        print("\nTop 5 instruments by mean impact:")
        print(df_all_instrument.groupby('instrument')['sum_impact'].mean().sort_values(ascending=False).head())
    else:
        print("\n[WARNING] No FSOI data computed - check logs for errors")


if __name__ == "__main__":
    main()
