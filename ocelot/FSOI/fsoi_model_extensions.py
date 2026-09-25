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
    return_sampling_info: bool = False,
):
    """Sample rows and optionally expose exact first-order inclusion probabilities."""
    from fsoi_sampling import sample_rows

    lat_np = lat.detach().cpu().float().numpy() if lat is not None else None
    lon_np = lon.detach().cpu().float().numpy() if lon is not None else None
    idx, info = sample_rows(lat_np, lon_np, n_raw, max_n, seed, grid_deg)
    indices = torch.from_numpy(idx).long()
    return (indices, info) if return_sampling_info else indices


def pseudo_target_conditioning(curr_input, inst_name: str, idx: torch.Tensor = None) -> dict:
    """Decoder conditioning carried by real target nodes: pressure level and valid time.

    Without it the radiosonde/aircraft decoder starts from zeros (no level) and the
    target-time bias is skipped for every instrument, a decoding path the model never
    saw in training. target_metadata is [lat_rad, lon_rad, time features]; the model
    reads only the trailing time-feature columns.
    """
    from process_timeseries import _encode_target_time_features

    def rows(value):
        return value if idx is None else value[idx.to(value.device)]

    n_rows = curr_input.lat.numel()
    lat, lon = rows(curr_input.lat).reshape(-1), rows(curr_input.lon).reshape(-1)
    conditioning = {}
    if inst_name in ('radiosonde', 'aircraft'):
        level = getattr(curr_input, 'pressure_level', None)
        if level is None or level.numel() != n_rows:
            raise ValueError(f"{inst_name}_input needs row-aligned pressure_level for a level-conditioned xb")
        conditioning['pressure_level'] = rows(level).reshape(-1).long().clone()
    times = getattr(curr_input, 'input_times', None)
    if times is None or times.numel() != n_rows:
        raise ValueError(f"{inst_name}_input needs row-aligned input_times for a time-conditioned xb")
    time_features = _encode_target_time_features(
        rows(times).reshape(-1).detach().cpu().numpy().astype(np.int64),
        lon.detach().cpu().double().numpy(),
    )
    conditioning['target_metadata'] = torch.cat([
        torch.deg2rad(lat.detach().float()).view(-1, 1).cpu(),
        torch.deg2rad(lon.detach().float()).view(-1, 1).cpu(),
        torch.from_numpy(time_features).float(),
    ], dim=1)
    return conditioning


def predict_at_targets(
    model,
    prev_batch: HeteroData,
    curr_batch_metadata: HeteroData,
    instrument_catalog,
    pipeline_config,
    forecast_step: int = 0,
    keep_instruments: list = None,  # NEW: filter which instruments to predict
    max_decoder_nodes: dict = None,  # NEW: cap decoder nodes per instrument {inst: int}
    return_sampling_info: bool = False,
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
        instrument_catalog: Typed instrument catalog
        pipeline_config: Typed pipeline configuration selecting instruments
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
        If return_sampling_info is True, a third dictionary contains row indices,
        strata, and inclusion probabilities for design-weighted population totals.
    """
    from torch_geometric.data import HeteroData
    from create_mesh_graph_global import obs_mesh_conn

    device = next(model.parameters()).device
    if max_decoder_nodes is None:
        max_decoder_nodes = {}

    # Track subsample indices so the caller can align xa accordingly
    subsample_indices: Dict[str, torch.Tensor] = {}
    sampling_designs = {}

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

    enabled_instruments = list(pipeline_config.enabled(instrument_catalog))
    for inst_name, instrument in enabled_instruments:
        if True:
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
                idx, sampling_designs[inst_name] = _stratified_spatial_subsample(
                    lat_for_strat, lon_for_strat,
                    n_raw=N_raw, max_n=max_n, seed=42, return_sampling_info=True,
                )
                idx = idx.to(curr_input.x.device)
                subsample_indices[inst_name] = idx
                design = sampling_designs[inst_name]
                pi = design['inclusion_probability']
                print(f"[Background] {inst_name}: {design['sampling_design']} "
                      f"{N_raw} -> {len(idx)} rows; grid={design['sampling_grid_deg']:g} deg; "
                      f"inclusion probability=[{pi.min():.6g}, {pi.max():.6g}]")
            else:
                idx = None  # no subsampling
                subsample_indices[inst_name] = None
                from fsoi_sampling import sample_rows
                _, sampling_designs[inst_name] = sample_rows(None, None, N_raw, N_raw)
            # ----------------------------------------------------------------

            # Create pseudo-target node at INPUT locations
            if hasattr(curr_input, 'lat'):
                lat_src = curr_input.lat if idx is None else curr_input.lat[idx]
                forecast_batch[pseudo_target_type].lat = lat_src.clone()
            if hasattr(curr_input, 'lon'):
                lon_src = curr_input.lon if idx is None else curr_input.lon[idx]
                forecast_batch[pseudo_target_type].lon = lon_src.clone()

            # Match the conditioning of real target nodes (pressure level, valid time).
            for attr, value in pseudo_target_conditioning(curr_input, inst_name, idx).items():
                forecast_batch[pseudo_target_type][attr] = value

            # For decoder .x: extract metadata from INPUT .x
            # Decoder needs scan angles (for satellites) or can use minimal dummy
            if hasattr(curr_input, 'x'):
                x_input = curr_input.x if idx is None else curr_input.x[idx]
                n_channels = instrument.target_dim

                if n_channels == 0:
                    print(f"[Background] WARNING: {inst_name} has no channels in config, skipping")
                    continue

                # Get scan_angle_channels from config (satellites only)
                scan_angle_channels = instrument.scan_angle_channels

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
    for enabled_index, (inst_name, instrument) in enumerate(enabled_instruments):
        if True:
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
                if enabled_index == 0:
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
    num_mesh_nodes = model.mesh.x.shape[0]
    forecast_batch["mesh"].x = torch.zeros(
        (num_mesh_nodes, model.mesh.x.shape[1]),
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

    if return_sampling_info:
        return background_predictions, subsample_indices, sampling_designs
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
