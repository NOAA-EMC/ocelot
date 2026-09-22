"""
FSOI Model Extensions - Add background prediction capability to GNNLightning.

This module extends the GNN model with methods needed for FSOI:
1. predict_at_targets: Predict observations at specified target locations
2. freeze_for_inference: Prepare model for gradient computation w.r.t. inputs

Author: Azadeh Gholoubi
"""

import torch
import numpy as np
from typing import Dict
from torch_geometric.data import HeteroData, Batch


def _stratified_spatial_subsample(
    lat: torch.Tensor,
    lon: torch.Tensor,
    n_raw: int,
    max_n: int,
    seed: int = 42,
    grid_deg: float = 10.0,
) -> torch.Tensor:
    """Return a sorted index tensor of length max_n drawn by stratified spatial sampling.

    The globe is divided into grid_deg × grid_deg cells.  Slots are allocated
    proportionally to the number of observations per non-empty cell, then filled
    by random sampling within each cell.  This ensures the subsample is
    geographically representative, preventing swath-concentration bias when
    scaling totals via sum_impact_scaled.

    Falls back to uniform random sampling when lat/lon are unavailable or
    when fewer than 4 non-empty cells exist (too sparse to stratify).
    """
    rng = np.random.default_rng(seed)

    if lat is None or lon is None or lat.numel() != n_raw or lon.numel() != n_raw:
        idx = torch.from_numpy(
            np.sort(rng.choice(n_raw, size=max_n, replace=False))
        ).long()
        return idx

    lat_np = lat.detach().cpu().float().numpy()
    lon_np = lon.detach().cpu().float().numpy()

    # Assign each observation to a lat/lon grid cell
    lat_edges = np.arange(-90, 90 + grid_deg, grid_deg)
    lon_edges = np.arange(-180, 180 + grid_deg, grid_deg)
    lat_idx = np.clip(np.digitize(lat_np, lat_edges) - 1, 0, len(lat_edges) - 2)
    lon_idx = np.clip(np.digitize(lon_np, lon_edges) - 1, 0, len(lon_edges) - 2)
    cell_id = lat_idx * len(lon_edges) + lon_idx

    unique_cells, cell_counts = np.unique(cell_id, return_counts=True)
    n_cells = len(unique_cells)

    if n_cells < 4:
        # Too sparse to stratify: fall back to uniform sampling
        idx = torch.from_numpy(
            np.sort(rng.choice(n_raw, size=max_n, replace=False))
        ).long()
        return idx

    # Proportional allocation: at least 1 slot per non-empty cell
    alloc = np.maximum(1, np.round(max_n * cell_counts / n_raw).astype(int))
    # Trim or extend so total == max_n
    diff = max_n - int(alloc.sum())
    if diff > 0:
        # Add extra slots to the largest cells
        order = np.argsort(-cell_counts)
        for i in range(diff):
            alloc[order[i % n_cells]] += 1
    elif diff < 0:
        # Remove slots from the smallest cells (but keep >= 1)
        order = np.argsort(cell_counts)
        for i in range(-diff):
            ci = order[i % n_cells]
            if alloc[ci] > 1:
                alloc[ci] -= 1

    # Sample within each cell
    selected = []
    for c, take in zip(unique_cells, alloc):
        obs_in_cell = np.where(cell_id == c)[0]
        take = min(take, len(obs_in_cell))
        if take > 0:
            chosen = rng.choice(obs_in_cell, size=take, replace=False)
            selected.append(chosen)

    if not selected:
        idx = torch.from_numpy(
            np.sort(rng.choice(n_raw, size=max_n, replace=False))
        ).long()
        return idx

    all_selected = np.concatenate(selected)
    # Deduplicate and trim to max_n (rounding may have given slightly more)
    all_selected = np.unique(all_selected)[:max_n]
    # If somehow we have fewer than max_n, top up uniformly from remaining obs
    if len(all_selected) < max_n:
        remaining = np.setdiff1d(np.arange(n_raw), all_selected)
        extra = rng.choice(remaining, size=min(max_n - len(all_selected), len(remaining)),
                           replace=False)
        all_selected = np.sort(np.concatenate([all_selected, extra]))
    else:
        all_selected = np.sort(all_selected)

    return torch.from_numpy(all_selected.astype(np.int64)).long()


def predict_at_targets(
    model,
    prev_batch: HeteroData,
    curr_batch_metadata: HeteroData,
    observation_config: dict,
    forecast_step: int = 0,
    keep_instruments: list = None,  # NEW: filter which instruments to predict
    max_decoder_nodes: dict = None,  # NEW: cap decoder nodes per instrument {inst: int}
) -> tuple:
    """
    Use observations from prev_batch to predict what the observations
    in curr_batch INPUT locations should be (background forecast xb).

    KEY CHANGE for GraphDOP-style FSOI:
    - xb must be background estimate at CURRENT INPUT observation locations
    - NOT at target verification locations
    - This ensures δx = xa - xb compares observations at same locations

    Strategy:
    1. Encoder: Use prev_batch INPUT nodes (prev obs → mesh)
    2. Processor: Run latent steps forward
    3. Decoder: Build pseudo-target nodes at curr INPUT locations (mesh → curr obs fit)

    This creates xb predictions at the same locations as xa (current inputs).

    NOTE: This runs under torch.no_grad() because xb is detach()-ed immediately
    after. Gradients are never needed w.r.t. the forward pass parameters here —
    only the *resulting tensor* values are needed. This halves peak CUDA memory.

    Args:
        model: GNNLightning model
        prev_batch: Input batch from previous time window (k-1)
        curr_batch_metadata: Current batch (k) with INPUT nodes to predict at
        observation_config: Configuration dict for instruments
        forecast_step: Which latent step to use for prediction
        keep_instruments: List of instruments to create predictions for (None = all)
                         Used to reduce memory by not creating heavy pseudo-targets
        max_decoder_nodes: Dict mapping instrument name → max decoder nodes to
                           allocate (e.g. {"avhrr": 20000}).  None or missing key
                           means no limit.  Returns the subsample index as a
                           separate dict so the caller can subsample xa to match.

    Returns:
        Tuple of:
          background_predictions: Dict mapping instrument names to predicted tensors
          subsample_indices: Dict mapping instrument names to index tensors used for
                             subsampling (or None if no subsampling was done).
                             Caller must apply same subsampling to xa.
    """
    from torch_geometric.data import HeteroData
    from create_mesh_graph_global import obs_mesh_conn

    device = next(model.parameters()).device
    if max_decoder_nodes is None:
        max_decoder_nodes = {}

    # Track subsample indices so the caller can align xa accordingly
    subsample_indices: Dict[str, torch.Tensor] = {}

    # Create forecast batch
    forecast_batch = HeteroData()

    # ===========================================================================
    # STEP 1: Copy INPUT nodes from prev_batch (encoder - previous observations)
    # ===========================================================================
    # NOTE: Skip node types with 0 observations.  PyG's Batch.from_data_list
    # calls value.max() on edge_index/index attrs and crashes on 0-row tensors,
    # and a 0-row node store contributes nothing to the forward pass anyway.
    for node_type in prev_batch.node_types:
        if "_input" in node_type:
            store = prev_batch[node_type]
            n_rows = 0
            if hasattr(store, 'x') and store.x is not None:
                n_rows = store.x.shape[0]
            elif hasattr(store, 'lat') and store.lat is not None:
                n_rows = store.lat.shape[0]
            if n_rows == 0:
                print(f"[Background] Skipping empty input node store {node_type} (0 obs in prev_batch)")
                continue

            # Copy ALL attributes from prev_batch input nodes
            for attr_name in store.keys():
                if attr_name not in ['edge_index', 'y']:  # Skip edges and targets
                    forecast_batch[node_type][attr_name] = store[attr_name]

            inst_name = node_type.replace("_input", "")
            print(f"[Background] Copied {node_type} from prev_batch ({n_rows} obs)")

    # ===========================================================================
    # STEP 2: Build ENCODER edges from prev_batch INPUT locations → mesh
    # ===========================================================================
    for node_type in prev_batch.node_types:
        if "_input" in node_type:
            # Skip if we did not copy this input node store in STEP 1
            if node_type not in forecast_batch.node_types:
                continue
            edge_type = (node_type, "to", "mesh")

            # Rebuild encoder edges using prev obs locations
            if hasattr(prev_batch[node_type], 'lat') and hasattr(prev_batch[node_type], 'lon'):
                prev_lat = prev_batch[node_type].lat.cpu().numpy()
                prev_lon = prev_batch[node_type].lon.cpu().numpy()

                edge_index_enc, edge_attr_enc = obs_mesh_conn(
                    prev_lat,
                    prev_lon,
                    model.mesh_structure["m2m_graphs"],
                    model.mesh_structure["mesh_lat_lon_list"],
                    model.mesh_structure["mesh_list"],
                    o2m=True,  # obs to mesh
                )

                if edge_index_enc.numel() == 0 or edge_index_enc.shape[1] == 0:
                    print(f"[Background] Skipping encoder edges for {node_type}: 0 edges")
                    continue

                forecast_batch[edge_type].edge_index = edge_index_enc
                forecast_batch[edge_type].edge_attr = edge_attr_enc

                print(f"[Background] Built encoder edges for {node_type}: {edge_index_enc.shape[1]} edges")

    # ===========================================================================
    # STEP 3: Create pseudo-TARGET nodes at curr INPUT locations for decoding
    # ===========================================================================
    # We predict at current INPUT observation locations (not verification targets)
    # This gives us background estimate xb at the same locations as analysis xa
    # Large instruments (e.g. avhrr with 1.3M nodes) are subsampled to
    # max_decoder_nodes[inst] to cap decoder edge memory and avoid OOM.

    for obs_type, instruments in observation_config.items():
        for inst_name, inst_cfg in instruments.items():
            # MEMORY OPTIMIZATION: Skip instruments not in keep_instruments
            if keep_instruments is not None and inst_name not in keep_instruments:
                continue

            node_type_input = f"{inst_name}_input"

            if node_type_input not in curr_batch_metadata.node_types:
                continue

            pseudo_target_type = f"{inst_name}_target_step{forecast_step}"
            curr_input = curr_batch_metadata[node_type_input]

            # Skip instruments with 0 obs in curr_batch — building a 0-row
            # pseudo-target would later crash Batch.from_data_list.
            n_curr = 0
            if hasattr(curr_input, 'x') and curr_input.x is not None:
                n_curr = curr_input.x.shape[0]
            elif hasattr(curr_input, 'lat') and curr_input.lat is not None:
                n_curr = curr_input.lat.shape[0]
            if n_curr == 0:
                print(f"[Background] Skipping pseudo-target {pseudo_target_type}: 0 curr obs")
                subsample_indices[inst_name] = None
                continue

            # ---- optional subsample (geographically stratified) -----------
            N_raw = curr_input.x.shape[0] if hasattr(curr_input, 'x') else 0
            max_n = max_decoder_nodes.get(inst_name, None)
            if max_n is not None and N_raw > max_n:
                lat_for_strat = getattr(curr_input, 'lat', None)
                lon_for_strat = getattr(curr_input, 'lon', None)
                idx = _stratified_spatial_subsample(
                    lat_for_strat, lon_for_strat,
                    n_raw=N_raw, max_n=max_n, seed=42,
                )
                idx = idx.to(curr_input.x.device)
                subsample_indices[inst_name] = idx
                print(f"[Background] {inst_name}: stratified spatial subsample "
                      f"{N_raw} → {max_n} nodes (geographically representative)")
            else:
                idx = None  # no subsampling
                subsample_indices[inst_name] = None
            # ----------------------------------------------------------------

            # Create pseudo-target node at INPUT locations
            if hasattr(curr_input, 'lat'):
                lat_src = curr_input.lat if idx is None else curr_input.lat[idx]
                forecast_batch[pseudo_target_type].lat = lat_src.clone()
            if hasattr(curr_input, 'lon'):
                lon_src = curr_input.lon if idx is None else curr_input.lon[idx]
                forecast_batch[pseudo_target_type].lon = lon_src.clone()

            # For decoder .x: extract metadata from INPUT .x
            # Decoder needs scan angles (for satellites) or can use minimal dummy
            if hasattr(curr_input, 'x'):
                x_input = curr_input.x if idx is None else curr_input.x[idx]
                n_channels = len(inst_cfg.get('features', []))

                if n_channels == 0:
                    print(f"[Background] WARNING: {inst_name} has no channels in config, skipping")
                    continue

                # Get scan_angle_channels from config (satellites only)
                scan_angle_channels = inst_cfg.get('scan_angle_channels', 0)

                # x_input layout: [7 geo/time | n_meta inst-metadata | n_channels obs | trailing]
                # Instrument metadata (cols 7..7+n_meta) holds scan/solar angles for satellites.
                n_meta = len(inst_cfg.get('metadata', []))
                bt_start = 7 + n_meta

                if n_meta > 0:
                    metadata = x_input[:, 7:bt_start]  # actual instrument metadata (scan angles etc.)

                    # For satellites: decoder expects scan angles
                    # Use first scan_angle_channels from instrument metadata
                    if scan_angle_channels > 0 and metadata.shape[1] >= scan_angle_channels:
                        decoder_x = metadata[:, :scan_angle_channels].clone()
                    else:
                        decoder_x = metadata.clone() if metadata.shape[1] > 0 else torch.zeros(
                            (x_input.shape[0], 1), dtype=torch.float32, device=device
                        )

                    forecast_batch[pseudo_target_type].x = decoder_x

                    print(f"[Background] {inst_name}: decoder .x shape={decoder_x.shape} "
                          f"(scan_angle_channels={scan_angle_channels})")
                else:
                    # No metadata - create minimal dummy
                    forecast_batch[pseudo_target_type].x = torch.zeros(
                        (x_input.shape[0], max(1, scan_angle_channels)),
                        dtype=torch.float32,
                        device=device,
                    )
                    print(f"[Background] {inst_name}: no metadata, created dummy decoder .x")
            else:
                # No .x - skip this instrument
                print(f"[Background] WARNING: {inst_name} has no .x, skipping")
                continue

            print(f"[Background] Created pseudo-target {pseudo_target_type}: "
                  f"n_obs={forecast_batch[pseudo_target_type].x.shape[0]}")

    # ===========================================================================
    # STEP 4: Build DECODER edges from mesh → curr INPUT locations
    # ===========================================================================
    for obs_type, instruments in observation_config.items():
        for inst_name, inst_cfg in instruments.items():
            # MEMORY OPTIMIZATION: Skip instruments not in keep_instruments
            if keep_instruments is not None and inst_name not in keep_instruments:
                continue

            node_type_input = f"{inst_name}_input"

            if node_type_input not in curr_batch_metadata.node_types:
                continue

            pseudo_target_type = f"{inst_name}_target_step{forecast_step}"
            edge_type = ("mesh", "to", pseudo_target_type)

            # Check if pseudo-target was created in STEP 3
            if pseudo_target_type not in forecast_batch.node_types:
                continue

            curr_input = curr_batch_metadata[node_type_input]

            # Use potentially-subsampled lat/lon (already stored on pseudo-target)
            if hasattr(forecast_batch[pseudo_target_type], 'lat') and \
               hasattr(forecast_batch[pseudo_target_type], 'lon'):
                curr_lat = forecast_batch[pseudo_target_type].lat.cpu().numpy()
                curr_lon = forecast_batch[pseudo_target_type].lon.cpu().numpy()

                # ALIGNMENT VERIFICATION: Print checksums for first instrument
                if obs_type == list(observation_config.keys())[0] and \
                   inst_name == list(observation_config[obs_type].keys())[0]:
                    lat_mean = forecast_batch[pseudo_target_type].lat.float().mean().item()
                    lon_mean = forecast_batch[pseudo_target_type].lon.float().mean().item()
                    lat_first5 = forecast_batch[pseudo_target_type].lat[:min(5, len(curr_lat))].cpu().numpy()
                    lon_first5 = forecast_batch[pseudo_target_type].lon[:min(5, len(curr_lon))].cpu().numpy()

                    print(f"\n[PREDICT_AT_TARGETS SPATIAL CHECK] {inst_name}:")
                    print(f"  DECODER lat: mean={lat_mean:.4f}, first_5={lat_first5}")
                    print(f"  DECODER lon: mean={lon_mean:.4f}, first_5={lon_first5}")
                    print(f"  These should MATCH the INPUT checksums from verify_alignment()")

                edge_index_dec, edge_attr_dec = obs_mesh_conn(
                    curr_lat,
                    curr_lon,
                    model.mesh_structure["m2m_graphs"],
                    model.mesh_structure["mesh_lat_lon_list"],
                    model.mesh_structure["mesh_list"],
                    o2m=False,  # mesh to obs (decoder)
                )

                if edge_index_dec.numel() == 0 or edge_index_dec.shape[1] == 0:
                    print(f"[Background] Skipping decoder edges to {pseudo_target_type}: 0 edges")
                    continue

                forecast_batch[edge_type].edge_index = edge_index_dec
                forecast_batch[edge_type].edge_attr = edge_attr_dec

                print(f"[Background] Built decoder edges to {pseudo_target_type}: {edge_index_dec.shape[1]} edges")

    # ===========================================================================
    # STEP 5: Forward pass to get predictions
    # ===========================================================================
    # Add dummy mesh nodes (will be overwritten by model.forward())
    num_mesh_nodes = model.mesh_x.shape[0]
    forecast_batch["mesh"].x = torch.zeros(
        (num_mesh_nodes, model.mesh_x.shape[1]),
        dtype=torch.float32,
        device=device,
    )

    # ---- Defensive scrub: drop any 0-element node/edge stores ------------
    # PyG's collate calls value.max() on edge_index / index attrs and crashes
    # if a tensor has numel()==0.  Remove any node/edge type that ended up
    # empty before batching.
    for nt in list(forecast_batch.node_types):
        store = forecast_batch[nt]
        empty = False
        for k in list(store.keys()):
            v = store[k]
            if torch.is_tensor(v) and v.numel() == 0:
                empty = True
                break
        if empty:
            print(f"[Background] Scrubbing empty node store {nt} before batching")
            del forecast_batch[nt]

    for et in list(forecast_batch.edge_types):
        store = forecast_batch[et]
        ei = getattr(store, 'edge_index', None)
        if ei is None or (torch.is_tensor(ei) and ei.numel() == 0):
            print(f"[Background] Scrubbing empty edge store {et} before batching")
            del forecast_batch[et]

    # Batch and move to device
    forecast_batch = Batch.from_data_list([forecast_batch])
    forecast_batch = forecast_batch.to(device)

    # *** Run forward pass under no_grad.  xb is detached immediately after, so
    #     we never need gradients w.r.t. the model parameters here.  Using
    #     no_grad() prevents autograd from allocating the activation buffers
    #     required for a backward pass, cutting peak CUDA memory ~40-50%.
    with torch.no_grad():
        forward_output = model(forecast_batch)

    # GNNLightning can return either:
    #   - predictions: Dict[str, List[Tensor]]
    #   - (predictions, mesh_features_per_step)
    if isinstance(forward_output, tuple):
        predictions = forward_output[0]
    else:
        predictions = forward_output

    # ===========================================================================
    # STEP 6: Extract predictions at INPUT locations
    # ===========================================================================
    background_predictions = {}

    for node_type, preds_list in predictions.items():
        if "_target_step" in node_type:
            inst_name = node_type.split("_target_step")[0]
        else:
            inst_name = node_type.replace("_target", "")

        # Get prediction for the requested step
        if len(preds_list) > forecast_step:
            background_predictions[inst_name] = preds_list[forecast_step]
            print(f"[Background] Extracted xb for {inst_name}: shape={preds_list[forecast_step].shape}")
        else:
            print(f"[WARNING] {node_type} has only {len(preds_list)} steps, requested step {forecast_step}")

    return background_predictions, subsample_indices


def freeze_model_for_fsoi(model):
    """
    Prepare model for FSOI inference:
    1. Set to eval mode (disable dropout, batchnorm)
    2. Freeze all parameters (requires_grad=False)
    3. But keep computation graph for input gradients

    DO NOT use torch.no_grad() context for FSOI!
    We need gradients w.r.t. inputs, not weights.

    Args:
        model: GNNLightning model
    """
    model.eval()

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad_(False)

    print("[FSOI] Model frozen for inference")
    print("[FSOI] All parameters have requires_grad=False")
    print("[FSOI] But computation graph is retained for input gradients")


def unfreeze_model(model):
    """
    Restore model to trainable state.

    Args:
        model: GNNLightning model
    """
    model.train()

    for param in model.parameters():
        param.requires_grad_(True)

    print("[FSOI] Model unfrozen for training")
