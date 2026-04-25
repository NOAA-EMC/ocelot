"""
FSOI Model Extensions - Add background prediction capability to GNNLightning.

This module extends the GNN model with methods needed for FSOI:
1. predict_at_targets: Predict observations at specified target locations
2. freeze_for_inference: Prepare model for gradient computation w.r.t. inputs

Author: Azadeh Gholoubi
"""

import torch
from typing import Dict
from torch_geometric.data import HeteroData, Batch


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
    for node_type in prev_batch.node_types:
        if "_input" in node_type:
            # Copy ALL attributes from prev_batch input nodes
            for attr_name in prev_batch[node_type].keys():
                if attr_name not in ['edge_index', 'y']:  # Skip edges and targets
                    forecast_batch[node_type][attr_name] = prev_batch[node_type][attr_name]

            inst_name = node_type.replace("_input", "")
            print(f"[Background] Copied {node_type} from prev_batch")

    # ===========================================================================
    # STEP 2: Build ENCODER edges from prev_batch INPUT locations → mesh
    # ===========================================================================
    for node_type in prev_batch.node_types:
        if "_input" in node_type:
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

            # ---- optional subsample ----------------------------------------
            N_raw = curr_input.x.shape[0] if hasattr(curr_input, 'x') else 0
            max_n = max_decoder_nodes.get(inst_name, None)
            if max_n is not None and N_raw > max_n:
                torch.manual_seed(42)
                idx = torch.randperm(N_raw, device=curr_input.x.device)[:max_n]
                idx = idx.sort()[0]
                subsample_indices[inst_name] = idx
                print(f"[Background] {inst_name}: subsampling decoder {N_raw} → {max_n} nodes")
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

                # Extract metadata portion (everything after channels)
                if x_input.shape[1] > n_channels:
                    metadata = x_input[:, n_channels:]

                    # For satellites: decoder expects scan angles
                    # Use first scan_angle_channels from metadata
                    if scan_angle_channels > 0 and metadata.shape[1] >= scan_angle_channels:
                        decoder_x = metadata[:, :scan_angle_channels].clone()
                    else:
                        # Use all metadata or create minimal dummy
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
