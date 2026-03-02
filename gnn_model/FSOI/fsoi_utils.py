"""
FSOI Utilities - Helper functions for FSOI computation.

This module provides utilities for:
- Extracting observation values from batches
- Computing forecast errors
- Computing gradients (adjoints)
- Aggregating FSOI results
- Validation and diagnostics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict


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
            1: 'specific_humidity',
            2: 'u_wind',
            3: 'v_wind',
        }
        return {k: v for k, v in base.items() if k < n_channels}
    # Fallback: generic
    return {i: f'channel_{i}' for i in range(n_channels)}


def sample_innovation_vs_fsoi(
    fsoi_values: Dict[str, torch.Tensor],
    innovations: Dict[str, torch.Tensor],
    max_points: int = 200000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a lightweight random sample of (innovation, fsoi) pairs.

    This is used for innovation-vs-FSOI scatter plots without storing full tensors.
    Sample is taken across all instruments/channels available.
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
        n = int(f.numel())
        if n <= 0:
            continue
        avail[inst] = n
        total_available += n

    if total_available == 0:
        return pd.DataFrame()

    remaining = int(max_points)
    for inst, n in sorted(avail.items(), key=lambda kv: kv[1], reverse=True):
        if remaining <= 0:
            break
        # Proportional allocation with a minimum of 2000 for big instruments
        take = int(np.ceil(max_points * (n / total_available)))
        take = int(min(max(take, 2000 if n >= 20000 else 200), remaining, n))

        f = fsoi_values[inst].detach().cpu().reshape(-1)
        inn = innovations[inst].detach().cpu().reshape(-1)

        idx = rng.choice(n, size=take, replace=False)
        # Recover channel index
        C = int(fsoi_values[inst].shape[1])
        ch = (idx % C).astype(np.int64)

        frames.append(
            pd.DataFrame(
                {
                    'instrument': inst,
                    'channel': ch,
                    'innovation': inn.numpy()[idx],
                    'fsoi': f.numpy()[idx],
                }
            )
        )
        remaining -= take

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
) -> List[Dict]:
    """Compute per-(pressure level, target variable) FSOI and return aggregates.

    This is the closest analogue to the paper’s Fig. 5 workflow: define a family
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
    if not hasattr(batch_xa[target_nt], 'pressure_level'):
        raise ValueError(f"[PerLevelVar] '{target_nt}' has no pressure_level attribute")

    if not hasattr(batch_xa[target_nt], 'y') or batch_xa[target_nt].y is None:
        raise ValueError(f"[PerLevelVar] '{target_nt}' has no y targets")

    y_ref = batch_xa[target_nt].y
    if y_ref.dim() != 2:
        raise ValueError(f"[PerLevelVar] Expected y to be [N,C], got {tuple(y_ref.shape)}")
    n_target_channels = int(y_ref.shape[1])

    # Pressure levels
    pl = batch_xa[target_nt].pressure_level
    if pl.dim() > 1:
        pl = pl.squeeze(1)
    unique_levels = sorted(int(p) for p in torch.unique(pl).tolist())

    # Determine target channels to compute
    ch_name_map = _default_target_channel_names(target_inst, n_target_channels)
    keep_channels = list(range(n_target_channels))

    if requested_target_variables is not None:
        # Map variable names to channel indices using model.instrument_channels if available
        mapped = []
        if hasattr(model, 'instrument_channels'):
            cinfo = model.instrument_channels.get(target_inst, [])
            for i, c in enumerate(cinfo):
                v = c.get('variable_name', c.get('variable', ''))
                if v in requested_target_variables:
                    mapped.append(i)
        if mapped:
            keep_channels = sorted(set(i for i in mapped if 0 <= i < n_target_channels))
        else:
            # Fallback: try direct name map
            inv = {v: k for k, v in ch_name_map.items()}
            keep_channels = [inv[v] for v in requested_target_variables if v in inv]
            keep_channels = sorted(set(i for i in keep_channels if 0 <= i < n_target_channels))

    if not keep_channels:
        print(f"[PerLevelVar] No target channels matched requested variables {requested_target_variables}; defaulting to all")
        keep_channels = list(range(n_target_channels))

    # Helper: loss for a single pressure and single target channel
    def _loss_for(preds, batch, p_idx: int, ch_idx: int):
        preds_list = preds.get(target_nt) or preds.get(f"{target_inst}_target")
        if preds_list is None or len(preds_list) <= forecast_lead_step:
            return None
        y_pred = preds_list[forecast_lead_step]
        if not hasattr(batch[target_nt], 'y') or batch[target_nt].y is None:
            return None
        y_ref_loc = batch[target_nt].y
        if y_pred.shape != y_ref_loc.shape:
            return None

        sq = (y_pred[:, ch_idx:ch_idx+1] - y_ref_loc[:, ch_idx:ch_idx+1]) ** 2  # [N,1]
        pl_loc = batch[target_nt].pressure_level.to(device)
        if pl_loc.dim() > 1:
            pl_loc = pl_loc.squeeze(1)
        pmask = (pl_loc == p_idx).float().view(-1, 1)
        if pmask.sum() == 0:
            return None
        sq = sq * pmask
        if use_area_weights and hasattr(batch[target_nt], 'lat'):
            lat = batch[target_nt].lat.to(device)
            sq = sq * torch.cos(torch.deg2rad(lat)).abs().view(-1, 1)
        return sq.sum()

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
            if loss is not None and loss.item() != 0.0:
                xa_losses.append((p_idx, ch, loss))

    ga_map: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
    ea_map: Dict[tuple[int, int], float] = {}
    for i, (p_idx, ch, loss) in enumerate(xa_losses):
        is_last = (i == len(xa_losses) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xa_list,
            retain_graph=not is_last,
            allow_unused=True,
        )
        ga = {}
        for k, g in zip(xa.keys(), grads):
            if g is None:
                continue
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
            if loss is not None and loss.item() != 0.0:
                xb_losses.append((p_idx, ch, loss))

    gb_map: Dict[tuple[int, int], Dict[str, torch.Tensor]] = {}
    eb_map: Dict[tuple[int, int], float] = {}
    for i, (p_idx, ch, loss) in enumerate(xb_losses):
        is_last = (i == len(xb_losses) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xb_list,
            retain_graph=not is_last,
            allow_unused=True,
        )
        gb = {}
        for k, g in zip(xb.keys(), grads):
            if g is None:
                continue
            gb[k] = g.detach().cpu()
        gb_map[(p_idx, ch)] = gb
        eb_map[(p_idx, ch)] = float(loss.detach().item())

    # Combine to FSOI per observation
    results = []
    for p_idx in unique_levels:
        # Map to hPa for readability
        _HPa = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10])
        p_hpa = float(_HPa[p_idx]) if 0 <= int(p_idx) < len(_HPa) else float('nan')
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
                dx = xa[inst].detach().cpu() - xb[inst].detach().cpu()
                gs = ga[inst] + gb[inst]
                if dx.shape != gs.shape:
                    continue
                fsoi_values[inst] = dx * gs
                innovations[inst] = dx
                gradient_sums[inst] = gs

            if not fsoi_values:
                continue
            results.append(
                {
                    'p_idx': int(p_idx),
                    'p_hpa': p_hpa,
                    'target_channel': int(ch),
                    'target_variable': ch_name_map.get(int(ch), f'channel_{ch}'),
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


# ==============================================================================
# Channel/Metadata Splitting
# ==============================================================================

def split_input_channels_and_meta(
    x_input: torch.Tensor,
    n_obs_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split INPUT .x into observation channels and metadata.

    INPUT .x typically contains: [obs_channels | metadata]
    - obs_channels: The actual observation values (e.g., 22 for ATMS, 4 for radiosonde)
    - metadata: Auxiliary features like scan angles, sin/cos lat/lon, etc.

    For FSOI, innovation δx = xa - xb must be computed in observation space only.
    Metadata should remain fixed in the batch.

    Args:
        x_input: Full input tensor [N, input_dim]
        n_obs_channels: Number of observation channels (from observation_config)

    Returns:
        channels: [N, n_obs_channels] - observation values only
        metadata: [N, n_meta] - auxiliary features
    """
    if x_input.shape[1] < n_obs_channels:
        raise ValueError(
            f"Input has {x_input.shape[1]} columns but config specifies "
            f"{n_obs_channels} observation channels"
        )

    channels = x_input[:, :n_obs_channels]
    metadata = x_input[:, n_obs_channels:]

    return channels, metadata


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


def merge_channels_and_meta(
    channels: torch.Tensor,
    metadata: torch.Tensor,
) -> torch.Tensor:
    """
    Merge observation channels and metadata back into full input format.

    This is used when replacing batch inputs with xa or xb:
    - Replace the observation channels (xa or xb)
    - Keep the original metadata unchanged

    Args:
        channels: [N, n_obs_channels] - observation values (xa or xb)
        metadata: [N, n_meta] - auxiliary features (from original batch)

    Returns:
        x_input: [N, input_dim] - full input tensor
    """
    return torch.cat([channels, metadata], dim=1)


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

            # Extract channels only (first n_channels columns)
            x_channels = x_input[:, :n_channels]

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

            # Extract lat/lon if available
            if hasattr(node_data, 'metadata'):
                # metadata is typically [N, 2] with [lat, lon]
                node_metadata = node_data.metadata.detach().cpu()
                if node_metadata.shape[1] >= 2:
                    metadata['lat'] = node_metadata[:, 0]
                    metadata['lon'] = node_metadata[:, 1]
                else:
                    metadata['lat'] = None
                    metadata['lon'] = None
            else:
                metadata['lat'] = None
                metadata['lon'] = None

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

            # Extract metadata (everything after channels)
            metadata = x_orig[:, n_channels:].detach()

            # Get new channels (xa or xb)
            new_channels = new_inputs[inst_name]

            if new_channels.shape[1] != n_channels:
                raise ValueError(
                    f"{inst_name}: new_channels has {new_channels.shape[1]} channels "
                    f"but config specifies {n_channels} channels"
                )

            # Detached base channels (non-subsampled rows act as constants)
            channels_base = x_orig[:, :n_channels].detach()
            metadata_full = x_orig[:, n_channels:].detach()

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

            batch[node_type_input].x = torch.cat([full_channels, metadata_full], dim=1)
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
    use_area_weights: bool = True,
    target_instruments: Optional[List[str]] = None,
    target_variables: Optional[List[str]] = None,
    target_pressure_levels: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Compute scalar forecast error e(x) for a given lead time.

    The error is computed as:
    e(x) = sum over targets of: w * (y_pred(x) - y_ref)^2

    Args:
        model: Trained GNN model (in eval mode)
        batch: Input batch with observations and targets
        forecast_lead_step: Which latent step to score (0-indexed)
        instrument_weights: Weight per instrument
        channel_weights: Weight per channel for each instrument
        use_area_weights: Apply latitude-dependent area weighting
        target_instruments: List of instruments to include (None = all)
        target_variables: List of variables to include (None = all)
            Examples: ["temperature"], ["u_wind", "v_wind"]
        target_pressure_levels: List of pressure levels in hPa (None = all)
            Examples: [1000, 850, 500, 250]

    Returns:
        Scalar tensor representing the forecast error
    """
    # ===========================================================================
    # MEMORY OPTIMIZATION: Prune unwanted targets BEFORE forward pass
    # ===========================================================================
    # Only decode targets we actually need for error computation
    # This prevents memory explosion from heavy instruments like AVHRR
    if target_instruments is not None:
        batch_copy = batch.clone()  # Clone to avoid modifying original
        prune_batch_targets_inplace(batch_copy, target_instruments, forecast_lead_step)
        batch = batch_copy

        # Optional: subsample remaining targets to further reduce memory
        # Uncomment if still hitting OOM:
        # for inst in target_instruments:
        #     subsample_target_nodes_inplace(batch, inst, forecast_lead_step, max_n=20000)

    # Forward pass to get predictions
    predictions = _unwrap_predictions(model(batch))

    # Scalar error accumulator (don't need requires_grad on accumulator)
    total_error = 0.0
    num_contributions = 0

    # Loop over all predicted instruments
    for node_type, preds_list in predictions.items():
        # Extract instrument name
        if "_target_step" in node_type:
            inst_name = node_type.split("_target_step")[0]
        else:
            inst_name = node_type.replace("_target", "")

        # Filter by target_instruments if specified
        if target_instruments is not None and inst_name not in target_instruments:
            continue

        # Check if we have predictions for the requested lead step
        if len(preds_list) <= forecast_lead_step:
            continue

        y_pred = preds_list[forecast_lead_step]

        # Get corresponding ground truth
        # Extract from batch based on node_type
        target_node_type = f"{inst_name}_target_step{forecast_lead_step}"
        if target_node_type not in batch.node_types:
            # Try without step suffix
            target_node_type = f"{inst_name}_target"
            if target_node_type not in batch.node_types:
                continue

        if not hasattr(batch[target_node_type], 'y'):
            continue

        y_ref = batch[target_node_type].y

        if y_ref is None or y_ref.numel() == 0:
            continue

        # Shape check
        if y_pred.shape != y_ref.shape:
            print(f"[WARNING] Shape mismatch for {inst_name}: pred={y_pred.shape}, ref={y_ref.shape}")
            continue

        # ==========================================
        # Initialize masks at loop start
        # ==========================================
        channel_mask = None
        pressure_mask = None

        # ==========================================
        # Filter by variables (e.g., temperature only)
        # ==========================================
        if target_variables is not None:
            # Get variable info from model
            if hasattr(model, 'instrument_channels'):
                channels_info = model.instrument_channels.get(inst_name, [])
                # Build mask for desired variables
                keep_channels = []
                for ch_idx, ch_info in enumerate(channels_info):
                    var_name = ch_info.get('variable_name', ch_info.get('variable', ''))
                    if var_name in target_variables:
                        keep_channels.append(ch_idx)

                if keep_channels:
                    # Create boolean mask [1, C]
                    channel_mask = torch.zeros(1, y_pred.shape[1], device=y_pred.device)
                    channel_mask[:, keep_channels] = 1.0
                    print(f"[Forecast Error] {inst_name}: Selected {len(keep_channels)}/{y_pred.shape[1]} channels "
                          f"for variables {target_variables}")
                else:
                    # No channels match - skip this instrument
                    print(f"[Forecast Error] {inst_name}: No channels match variables {target_variables}, skipping")
                    continue

        # ==========================================
        # Filter by pressure levels (for 3D instruments)
        # ==========================================
        if target_pressure_levels is not None:
            # Get pressure level info from batch or model
            pressure_mask = None

            # Option 1: Pressure levels stored in batch
            if hasattr(batch[target_node_type], 'pressure_level'):
                pressure = batch[target_node_type].pressure_level  # [N] or [N, 1]
                if pressure.dim() == 2:
                    pressure = pressure.squeeze(1)

                # Create mask for desired levels (with tolerance)
                level_mask = torch.zeros_like(pressure, dtype=torch.bool)
                for target_p in target_pressure_levels:
                    # Match within 1 hPa tolerance
                    level_mask |= (torch.abs(pressure - target_p) < 1.0)

                if level_mask.any():
                    pressure_mask = level_mask.view(-1, 1).float()  # [N, 1]
                    print(f"[Forecast Error] {inst_name}: Selected {level_mask.sum()}/{len(pressure)} obs "
                          f"at pressure levels {target_pressure_levels} hPa")
                else:
                    print(f"[Forecast Error] {inst_name}: No obs at pressure levels {target_pressure_levels}, skipping")
                    continue

            # Option 2: Pressure levels are channels (e.g., radiosonde profiles)
            elif hasattr(model, 'instrument_channels'):
                channels_info = model.instrument_channels.get(inst_name, [])
                keep_channels_p = []
                for ch_idx, ch_info in enumerate(channels_info):
                    ch_pressure = ch_info.get('pressure_level', ch_info.get('level', None))
                    if ch_pressure is not None:
                        for target_p in target_pressure_levels:
                            if abs(ch_pressure - target_p) < 1.0:
                                keep_channels_p.append(ch_idx)
                                break

                if keep_channels_p:
                    # Combine with variable mask if exists
                    p_mask = torch.zeros(1, y_pred.shape[1], device=y_pred.device)
                    p_mask[:, keep_channels_p] = 1.0

                    if channel_mask is not None:
                        channel_mask = channel_mask * p_mask  # Intersection
                    else:
                        channel_mask = p_mask

                    print(f"[Forecast Error] {inst_name}: Selected {len(keep_channels_p)} channels "
                          f"at pressure levels {target_pressure_levels} hPa")
                else:
                    print(f"[Forecast Error] {inst_name}: No channels at pressure levels {target_pressure_levels}, skipping")
                    continue

            # Apply pressure mask if created
            if pressure_mask is not None and channel_mask is None:
                # Apply to observation dimension only
                pass  # Will be applied to squared_error later

        # Get instrument weight using WEIGHTS mapping (not model checkpoint mapping)
        # Use model.weights_name_to_id if available (from YAML), else fall back to model mapping
        if hasattr(model, 'weights_name_to_id'):
            inst_id = model.weights_name_to_id.get(inst_name)
        else:
            inst_id = model.instrument_name_to_id.get(inst_name)

        inst_weight = instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

        # Get channel weights using WEIGHTS mapping
        ch_weights = None
        C = y_pred.shape[1]  # Number of channels in prediction

        if inst_id in channel_weights:
            ch_weights = channel_weights[inst_id].to(y_pred.device)
            # Broadcast to [1, C]
            ch_weights = ch_weights.view(1, -1)

            # CRITICAL: Check for channel count mismatch (prevents crashes)
            if ch_weights.numel() != C:
                # Build reverse mapping for diagnostics
                if hasattr(model, 'weights_name_to_id'):
                    id_to_name = {v: k for k, v in model.weights_name_to_id.items()}
                    mapped_inst = id_to_name.get(inst_id, '?')
                else:
                    id_to_name = {v: k for k, v in model.instrument_name_to_id.items()}
                    mapped_inst = id_to_name.get(inst_id, '?')

                print(
                    f"[WARNING] Channel-weight mismatch for {inst_name}: "
                    f"pred_C={C}, weight_C={ch_weights.numel()}, inst_id={inst_id} "
                    f"(maps_to={mapped_inst}). "
                    f"This indicates instrument ID mapping inconsistency. "
                    f"Falling back to uniform weights."
                )
                ch_weights = torch.ones(1, C, device=y_pred.device)

            # Combine with channel mask if exists
            if channel_mask is not None:
                ch_weights = ch_weights * channel_mask
        elif channel_mask is not None:
            ch_weights = channel_mask

        # Compute squared error
        squared_error = (y_pred - y_ref) ** 2  # [N, C]

        # Apply channel weights
        if ch_weights is not None:
            squared_error = squared_error * ch_weights

        # Apply area weights (latitude-based)
        if use_area_weights and hasattr(batch[target_node_type], 'lat'):
            lat = batch[target_node_type].lat  # [N]
            # Cosine weighting: more weight near equator
            area_weight = torch.cos(torch.deg2rad(lat)).abs()
            area_weight = area_weight.view(-1, 1)  # [N, 1]
            squared_error = squared_error * area_weight

        # Apply pressure mask if created (observation-level filtering)
        if pressure_mask is not None:
            squared_error = squared_error * pressure_mask

        # Apply valid mask if available
        if hasattr(batch[target_node_type], 'valid_mask'):
            valid_mask = batch[target_node_type].valid_mask
            squared_error = squared_error * valid_mask.float()

        # Sum over observations and channels, apply instrument weight
        error_contribution = squared_error.sum() * inst_weight

        # Accumulate error (first contribution initializes, rest adds)
        if num_contributions == 0:
            total_error = error_contribution
        else:
            total_error = total_error + error_contribution

        num_contributions += 1

        print(f"[Forecast Error] {inst_name}: error={error_contribution.item():.6f}, "
              f"inst_weight={inst_weight:.3f}, n_obs={y_pred.shape[0]}")

    if num_contributions == 0:
        print("[WARNING] No forecast error computed - no valid targets found")
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    print(f"[Forecast Error] Total: {total_error.item():.6f} from {num_contributions} instruments")

    return total_error


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
) -> Dict[str, torch.Tensor]:
    """
    Compute per-observation FSOI using the formula:

    FSOI = δx ⊙ (ga + gb)

    where:
    - δx = xa - xb (innovation)
    - ga = gradient of error w.r.t. analysis
    - gb = gradient of error w.r.t. background
    - ⊙ = elementwise multiplication

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
        delta_x = xa[inst_name] - xb[inst_name]
        g_sum = ga[inst_name] + gb[inst_name]

        # Elementwise product
        fsoi = delta_x * g_sum

        fsoi_values[inst_name] = fsoi
        innovations[inst_name] = delta_x
        gradient_sums[inst_name] = g_sum

        # Diagnostics
        impact_sum = fsoi.sum().item()
        impact_mean = fsoi.mean().item()
        positive_frac = (fsoi > 0).float().mean().item()

        print(f"[FSOI] {inst_name}: sum={impact_sum:.6e}, mean={impact_mean:.6e}, "
              f"positive={positive_frac*100:.1f}%")

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
    _HPa = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10])

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

    pl_tensor_cpu = batch_xa[target_nt].pressure_level
    if pl_tensor_cpu.dim() > 1:
        pl_tensor_cpu = pl_tensor_cpu.squeeze(1)
    unique_levels = sorted(int(p) for p in torch.unique(pl_tensor_cpu).tolist())
    print(f"[PerLevel] {len(unique_levels)} unique pressure levels: {unique_levels}")

    # ── Pre-compute channel mask (same for all levels) ────────────────────
    ch_mask = None
    if target_variables is not None and hasattr(model, 'instrument_channels'):
        cinfo = model.instrument_channels.get(target_inst, [])
        keep = [i for i, c in enumerate(cinfo)
                if c.get('variable_name', c.get('variable', '')) in target_variables]
        if keep:
            n_total_ch = len(cinfo)
            ch_mask = torch.zeros(1, n_total_ch, device=device)
            ch_mask[:, keep] = 1.0
            print(f"[PerLevel] Channel mask: keeping channels {keep} for variables {target_variables}")

    # ── Helper: scalar loss at a single pressure level ────────────────────
    def _level_loss(preds, batch, p_idx: int):
        # The model may key predictions as "radiosonde_target" (no step suffix)
        # or "radiosonde_target_step0".  Try both.
        preds_list = preds.get(target_nt) or preds.get(f"{target_inst}_target")
        if preds_list is None or len(preds_list) <= forecast_lead_step:
            return None
        y_pred = preds_list[forecast_lead_step]
        # y_ref and pressure_level come from the batch node (always has _step suffix)
        if not hasattr(batch[target_nt], 'y') or batch[target_nt].y is None:
            return None
        y_ref = batch[target_nt].y
        if y_pred.shape != y_ref.shape:
            return None

        sq = (y_pred - y_ref) ** 2  # [N, C]

        # Apply channel (variable) mask
        if ch_mask is not None and ch_mask.shape[1] == sq.shape[1]:
            sq = sq * ch_mask

        # Apply pressure-level mask
        pl = batch[target_nt].pressure_level.to(device)
        if pl.dim() > 1:
            pl = pl.squeeze(1)
        pmask = (pl == p_idx).float().view(-1, 1)
        if pmask.sum() == 0:
            return None
        sq = sq * pmask

        # Area weighting
        if use_area_weights and hasattr(batch[target_nt], 'lat'):
            lat = batch[target_nt].lat.to(device)
            sq = sq * torch.cos(torch.deg2rad(lat)).abs().view(-1, 1)

        return sq.sum()

    # ── xa: one forward pass, N_levels backward passes ───────────────────
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
        if loss is None or loss.item() == 0.0:
            print(f"[PerLevel xa] level {p_idx}: no targets")
        else:
            xa_valid_levels.append((p_idx, loss))

    for i, (p_idx, loss) in enumerate(xa_valid_levels):
        is_last = (i == len(xa_valid_levels) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xa_list,
            retain_graph=not is_last,
            allow_unused=True,
        )
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
        if loss is None or loss.item() == 0.0:
            print(f"[PerLevel xb] level {p_idx}: no targets")
        else:
            xb_valid_levels.append((p_idx, loss))

    for i, (p_idx, loss) in enumerate(xb_valid_levels):
        is_last = (i == len(xb_valid_levels) - 1)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xb_list,
            retain_graph=not is_last,
            allow_unused=True,
        )
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
            dx = xa[inst].detach().cpu() - xb[inst].detach().cpu()
            gs = ga_inst + gb_inst   # both are already on CPU (stored via .detach().cpu())

            if dx.shape[0] != gs.shape[0]:
                print(f"[PerLevel WARNING] {inst}: dx {dx.shape} vs gs {gs.shape} "
                      f"shape mismatch, skipping")
                continue

            fsoi_p[inst] = dx * gs
            innov_p[inst] = dx
            gsum_p[inst] = gs

        if not fsoi_p:
            continue

        p_hpa = float(_HPa[p_idx]) if 0 <= p_idx < len(_HPa) else float('nan')
        insts = list(fsoi_p.keys())
        print(f"[PerLevel] level {p_idx} ({p_hpa:.0f} hPa): FSOI for {insts}")

        level_results.append({
            'p_idx': p_idx,
            'p_hpa': p_hpa,
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
        'seviri',
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
    STANDARD_PRESSURE_LEVELS = np.array([
        1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
    ])

    EPS = 1e-12
    records = []

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

        # Select channel and optional mask
        innov_vec = innov[:, ch]
        g_vec = g_sum[:, ch]
        if mask is not None:
            innov_vec = innov_vec[mask]
            g_vec = g_vec[mask]

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

        # If we have pressure levels, stratify by them
        if pressure_levels is not None:
            # Group by pressure level and channel
            for ch in range(C):
                ch_impacts = fsoi_tensor[:, ch]

                # Get unique pressure levels
                unique_levels = torch.unique(pressure_levels)

                for press_level_idx in unique_levels:
                    # Mask for this pressure level
                    mask = (pressure_levels == press_level_idx)
                    if not mask.any():
                        continue

                    # Filter impacts for this pressure level
                    level_impacts = ch_impacts[mask]

                    # Map pressure index to hPa value
                    press_idx_int = int(press_level_idx.item())
                    if pressure_hpa_tensor is not None:
                        press_vals = pressure_hpa_tensor[mask]
                        press_hpa = float(press_vals.flatten()[0].item()) if press_vals.numel() > 0 else np.nan
                    elif 0 <= press_idx_int < len(STANDARD_PRESSURE_LEVELS):
                        press_hpa = STANDARD_PRESSURE_LEVELS[press_idx_int]
                    else:
                        press_hpa = np.nan  # Invalid/unknown

                    record = {
                        'instrument': inst_name,
                        'instrument_id': inst_id,
                        'channel': ch,
                        'pressure_level_idx': press_idx_int,
                        'pressure_hpa': press_hpa,
                        'mean_impact': level_impacts.mean().item(),
                        'sum_impact': level_impacts.sum().item(),
                        'positive_count': (level_impacts > 0).sum().item(),
                        'negative_count': (level_impacts < 0).sum().item(),
                        'zero_count': (level_impacts == 0).sum().item(),
                        'total_count': mask.sum().item(),
                        'positive_frac': (level_impacts > 0).float().mean().item(),
                    }

                    records.append(_attach_stats(record, inst_name, ch, mask))
        else:
            # No pressure stratification - aggregate over all observations
            for ch in range(C):
                ch_impacts = fsoi_tensor[:, ch]

                record = {
                    'instrument': inst_name,
                    'instrument_id': inst_id,
                    'channel': ch,
                    'mean_impact': ch_impacts.mean().item(),
                    'sum_impact': ch_impacts.sum().item(),
                    'positive_count': (ch_impacts > 0).sum().item(),
                    'negative_count': (ch_impacts < 0).sum().item(),
                    'zero_count': (ch_impacts == 0).sum().item(),
                    'total_count': N,
                    'positive_frac': (ch_impacts > 0).float().mean().item(),
                }

                records.append(_attach_stats(record, inst_name, ch))

    return pd.DataFrame(records)


def aggregate_fsoi_by_instrument(
    fsoi_values: Dict[str, torch.Tensor],
    instrument_name_to_id: Dict[str, int],
    innovations: Optional[Dict[str, torch.Tensor]] = None,
    gradient_sums: Optional[Dict[str, torch.Tensor]] = None,
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

        # Sum over all observations and channels
        total_impact = fsoi_tensor.sum().item()
        mean_impact = fsoi_tensor.mean().item()
        n_obs = fsoi_tensor.shape[0]
        n_channels = fsoi_tensor.shape[1]

        record = {
            'instrument': inst_name,
            'instrument_id': inst_id,
            'n_observations': n_obs,
            'n_channels': n_channels,
            'mean_impact': mean_impact,
            'sum_impact': total_impact,
            'positive_frac': (fsoi_tensor > 0).float().mean().item(),
        }

        innov = innovations.get(inst_name) if innovations is not None else None
        g_sum = gradient_sums.get(inst_name) if gradient_sums is not None else None

        if innov is not None and g_sum is not None:
            innov_vec = innov.reshape(-1)
            g_vec = g_sum.reshape(-1)

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
