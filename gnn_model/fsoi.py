"""
Forecast Sensitivity to Observation Impact (FSOI) for GNN-based observation prediction.

This implementation follows operational FSOI methodology from:
- Baker & Daley (2000): Observation and background adjoint sensitivity
- Langland & Baker (2004): Estimation of observation impact using adjoint
- Cardinali (2009): Monitoring observation impact on short-range forecast

FSOI measures the impact of individual observations on forecast accuracy by computing:
1. Sensitivity: gradient of forecast loss with respect to input observations (adjoint)
2. Impact: sensitivity weighted by observation increment

Mathematical formulation (Two-State Adjoint):
    FSOI_i = (o_i - o_background_i)^T · [M_b^T·∇L + M_a^T·∇L]

Where:
    o_i = input observation value (analysis)
    o_background_i = background state (model forecast)
    L = forecast verification loss (prediction error at target time)
    M_b^T = adjoint at background state
    M_a^T = adjoint at analysis state
    ∇L = gradient of loss with respect to predictions

Note: Observations are already normalized by their climatological standard deviation
during preprocessing, so no additional R_i normalization is needed here.

Sign Convention (following Langland & Baker 2004):
    Negative FSOI → observation reduces forecast error (beneficial)
    Positive FSOI → observation increases forecast error (detrimental)

Background State Options:
    - Sequential mode: Use forecast from previous time window
    - Same-window mode: Use model prediction at current time
    - Climatological mode: Use zero background

Target Filtering:
    - Conventional-only mode: Measure impact on prognostic variables (u, v, T, q, p)
    - All-targets mode: Measure impact on all observation types

Author: Azadeh Gholoubi (NOAA/EMC)
Date: November 2025
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def compute_observation_sensitivity(
    model,
    batch: HeteroData,
    predictions: Dict[str, List[torch.Tensor]],
    ground_truths: Dict[str, List[torch.Tensor]],
    instrument_weights: Optional[Dict[int, float]] = None,
    channel_weights: Optional[Dict[int, torch.Tensor]] = None,
    target_instrument: Optional[str] = None,
    conventional_only: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute sensitivity of forecast loss to input observations via backpropagation.

    This implements the adjoint calculation ∇_{o_input} L using automatic differentiation,
    analogous to the adjoint model in traditional 4D-Var systems. We use PyTorch's 
    reverse-mode automatic differentiation to compute the Vector-Jacobian Product (VJP), 
    which gives us the sensitivity of the forecast error to each input observation.
    
    The gradient is computed at the analysis state (with all observations included).

    Args:
        model: The trained GNN model
        batch: Input batch containing observations
        predictions: Model predictions for each instrument
        ground_truths: Ground truth targets for verification
        instrument_weights: Weights for different instruments
        channel_weights: Weights for different channels
        target_instrument: If specified, only compute loss for this target instrument
                          (e.g., "surface_obs" to see impact on surface predictions only)
        conventional_only: If True, only compute loss for conventional obs (surface_obs + radiosonde)
                          representing prognostic variables (u, v, T, q, p).

    Returns:
        Dictionary mapping node_type -> sensitivity tensor [N, C]
        where N is number of observations, C is number of channels
        
    References:
        - Baker & Daley (2000): Observation and background adjoint sensitivity
        - Langland & Baker (2004): Estimation of observation impact using adjoint
    """
    from loss import weighted_huber_loss

    model.eval()  # Set to eval mode but enable gradients
    sensitivities = {}

    # Zero existing gradients
    model.zero_grad()

    # Compute total forecast loss (same as validation loss)
    total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    num_predictions = 0

    for node_type, preds_list in predictions.items():
        if node_type not in ground_truths:
            continue

        gts_list = ground_truths[node_type]

        # Get instrument info
        if "_target_step" in node_type:
            inst_name = node_type.split("_target_step")[0]
        else:
            inst_name = node_type.replace("_target", "")

        # Skip if conventional_only and this is not a conventional instrument
        if conventional_only:
            conventional_instruments = ['surface_obs', 'radiosonde']
            if inst_name not in conventional_instruments:
                continue
        
        # Skip if target_instrument is specified and this is not it
        if target_instrument is not None and inst_name != target_instrument:
            continue

        inst_id = model.instrument_name_to_id.get(inst_name, None)
        instrument_weight = instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

        # Compute loss for each prediction step
        for step, (y_pred, y_true) in enumerate(zip(preds_list, gts_list)):
            if y_pred is None or y_true is None:
                continue

            # Get valid mask if available
            valid_mask = None
            step_info = model._get_latent_step_info(batch)
            step_mapping = step_info["step_mapping"]
            if node_type in step_mapping and step in step_mapping[node_type]:
                step_node_type = step_mapping[node_type][step]
                if hasattr(batch[step_node_type], "target_channel_mask"):
                    valid_mask = batch[step_node_type].target_channel_mask

            # Compute channel-weighted loss
            channel_loss = weighted_huber_loss(
                y_pred,
                y_true,
                instrument_ids=None,
                channel_weights=channel_weights,
                delta=0.1,
                rebalancing=True,
                valid_mask=valid_mask,
            )

            if torch.isfinite(channel_loss):
                weighted_loss = channel_loss * instrument_weight
                total_loss = total_loss + weighted_loss
                num_predictions += 1

    # Average loss
    avg_loss = total_loss / max(num_predictions, 1)

    # Backward pass to compute gradients
    avg_loss.backward(retain_graph=True)

    # Extract gradients for input observations
    for node_type in batch.node_types:
        if node_type.endswith("_input"):
            if batch[node_type].x.grad is not None:
                # Gradient shape: [N, feature_dim]
                # Store as sensitivity
                sensitivities[node_type] = batch[node_type].x.grad.clone().detach()
            else:
                # No gradient computed - observation didn't contribute
                sensitivities[node_type] = torch.zeros_like(batch[node_type].x)

    return sensitivities


def compute_observation_sensitivity_two_state(
    model,
    batch: HeteroData,
    predictions: Dict[str, List[torch.Tensor]],
    ground_truths: Dict[str, List[torch.Tensor]],
    instrument_weights: Optional[Dict[int, float]] = None,
    channel_weights: Optional[Dict[int, torch.Tensor]] = None,
    target_instrument: Optional[str] = None,
    conventional_only: bool = False,
    use_model_background: bool = True,
    previous_batch: Optional[HeteroData] = None,
    previous_predictions: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute FSOI using two-state adjoint formulation.
    
    Mathematical formulation (Laloyaux et al. 2025, Equation 3):
        Δe = e(x_a) - e(x_b) = (x_a - x_b)^T · g
    
    Where:
        g = M_b^T·C·(M(x_b) - x_ref) + M_a^T·C·(M(x_a) - x_ref)
    
    Background State Options:
        
        Sequential mode (with temporal continuity):
              - Time Window N-1 (e.g., 00Z-12Z): Process obs → produce 12h forecast
              - Time Window N (e.g., 12Z-24Z): 
                  x_b = the 12h forecast FROM Window N-1
                  x_a = actual observations in Window N
                  Innovation = x_a - x_b = "actual obs in Window N" - "what Window N-1 predicted"
        
        Same-window approximation (current limitation):
              - Each batch is INDEPENDENT (no temporal connection between batches)
              - We don't have access to "previous batch's forecast"
              - APPROXIMATION: We use model's prediction AT CURRENT TIME as x_b
                (what model predicts observations should be, given current state)
        
        Sequential mode requires:
              - Sequential processing of time windows (not independent batches)
              - Save each batch's 12h forecast → use as next batch's x_b
              - Temporal continuity in training/inference pipeline
        
        x_a = analysis state = ACTUAL observations received
        
        Innovation (x_a - x_b):
            - Difference between "what was observed" and "what model predicted"
            - Measures how much observations CORRECT the model
            - NOT deviation from climatology
        
        M^T·C·(M(x) - x_ref):
            - M^T = adjoint via PyTorch autograd
            - C = weighting matrix (target variable selection)
            - (M(x) - x_ref) = forecast error residual
            - Computed via backpropagation of residual through model
    
    Key differences from climatological FSOI:
        - Background: model forecast (not climatology)
        - Innovation: obs - forecast (not obs - climate)
        - Residual-based: uses L1 residual, not L2 loss
    
    This two-state formulation (M_b^T + M_a^T) reduces contamination compared to 
    single-state adjoint via 3rd-order Taylor expansion approximation.
    
    Args:
        model: The trained GNN model
        batch: Input batch with ACTUAL observations (analysis state x_a)
        predictions: Model predictions at analysis state (already computed)
        ground_truths: Ground truth targets for verification (x_ref)
        instrument_weights: Weights for different instruments
        channel_weights: Weights for different channels
        target_instrument: If specified, only compute sensitivity for this target
        conventional_only: If True, only compute for conventional obs (C matrix filter)
        use_model_background: If True (default), compute x_b as model's forecast.
                             If False, fall back to zeros.
        previous_batch: Optional batch from previous sequential window
        previous_predictions: Optional model predictions from previous_batch
                             If provided, uses sequential background (12h forecast from previous window)
                             If None, falls back to climatological/same-window approximation
    
    Returns:
        Dictionary mapping node_type -> FSOI values [N, C]
        
    References:
        Laloyaux et al. (2025): Using data assimilation tools to dissect AI models
        arXiv:2510.27388, Equation 3, Section 2
        
    Note:
        Sign convention: Negative FSOI = beneficial observation
    """
    from loss import weighted_huber_loss
    
    model.eval()
    
    # === STEP 1: Compute M_a^T (adjoint at analysis state) ===
    # This is gradient at current state with all observations
    # Enable gradients for entire computation (model is in eval mode)
    with torch.enable_grad():
        sensitivities_analysis = {}
        model.zero_grad()
        
        # CRITICAL: Recompute predictions inside gradient context
        # The predictions passed in were computed outside, so we need fresh ones with gradients
        predictions_with_grad = model(batch)
        
        # Compute forecast loss at analysis
        loss_analysis = None  # Will be initialized when first loss is computed
        num_predictions = 0
        
        for node_type, preds_list in predictions_with_grad.items():
            if node_type not in ground_truths:
                continue
            
            gts_list = ground_truths[node_type]
            
            # Get instrument info
            if "_target_step" in node_type:
                inst_name = node_type.split("_target_step")[0]
            else:
                inst_name = node_type.replace("_target", "")
            
            # Apply filters
            if conventional_only:
                conventional_instruments = ['surface_obs', 'radiosonde']
                if inst_name not in conventional_instruments:
                    continue
            
            if target_instrument is not None and inst_name != target_instrument:
                continue
            
            inst_id = model.instrument_name_to_id.get(inst_name, None)
            instrument_weight = instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0
            
            # Compute loss
            for step, (y_pred, y_true) in enumerate(zip(preds_list, gts_list)):
                if y_pred is None or y_true is None:
                    continue
                
                valid_mask = None
                step_info = model._get_latent_step_info(batch)
                step_mapping = step_info["step_mapping"]
                if node_type in step_mapping and step in step_mapping[node_type]:
                    step_node_type = step_mapping[node_type][step]
                    if hasattr(batch[step_node_type], "target_channel_mask"):
                        valid_mask = batch[step_node_type].target_channel_mask
                
                channel_loss = weighted_huber_loss(
                    y_pred, y_true,
                    instrument_ids=None,
                    channel_weights=channel_weights,
                    delta=0.1,
                    rebalancing=True,
                    valid_mask=valid_mask,
                )
                
                if torch.isfinite(channel_loss):
                    weighted_loss = channel_loss * instrument_weight
                    # Initialize or accumulate loss
                    if loss_analysis is None:
                        loss_analysis = weighted_loss
                    else:
                        loss_analysis = loss_analysis + weighted_loss
                    num_predictions += 1
        
        # Check if we have any valid predictions
        if loss_analysis is None or num_predictions == 0:
            print("[FSOI] Warning: No valid predictions found for FSOI computation")
            return {}
        
        avg_loss_analysis = loss_analysis / num_predictions
        avg_loss_analysis.backward(retain_graph=True)
    
    # Extract M_a^T (gradients at analysis)
    for node_type in batch.node_types:
        if node_type.endswith("_input"):
            if batch[node_type].x.grad is not None:
                sensitivities_analysis[node_type] = batch[node_type].x.grad.clone().detach()
            else:
                sensitivities_analysis[node_type] = torch.zeros_like(batch[node_type].x)
    
    # === STEP 2: Compute x_b (background state) and M_b^T (adjoint at background) ===
    # x_b = forecast from previous sequential window
    #
    # For radiosonde observations:
    #   - Filter to only compute FSOI for radiosonde (target_instrument or conventional_only)
    #   - Use sequential background when available (previous window's forecast)
    #   - Fall back to zero background for first batch or non-sequential transitions
    
    background_observations = {}
    
    # Use forecast from previous sequential window if available
    if previous_predictions is not None:
        print("[FSOI] Using forecast from previous sequential batch as x_b")
        with torch.no_grad():
            for node_type in batch.node_types:
                if not node_type.endswith("_input"):
                    continue
                
                inst_name = node_type.replace("_input", "")
                
                # Filter: Only process radiosonde (or target_instrument if specified)
                if conventional_only and inst_name not in ['radiosonde']:
                    continue
                if target_instrument is not None and inst_name != target_instrument:
                    continue
                
                obs_values = batch[node_type].x
                
                # Check if we have predictions from previous batch for this instrument
                target_key = f"{inst_name}_target"
                if target_key in previous_predictions and len(previous_predictions[target_key]) > 0:
                    # Use last prediction step from previous window as background for current window
                    # This is the 12h forecast from previous window
                    prev_prediction = previous_predictions[target_key][-1]  # [N, C] - last step
                    
                    # Match dimensions - previous batch may have different number of observations
                    if prev_prediction.shape[1] == obs_values.shape[1]:  # Same channels
                        # Use previous forecast as background
                        # If different number of observations, pad or truncate
                        if prev_prediction.shape[0] == obs_values.shape[0]:
                            background_observations[node_type] = prev_prediction.detach()
                            print(f"[FSOI] {inst_name}: Using previous forecast (same obs count: {prev_prediction.shape[0]})")
                        elif prev_prediction.shape[0] < obs_values.shape[0]:
                            # Previous had fewer obs - pad with zeros
                            bg = torch.zeros_like(obs_values)
                            bg[:prev_prediction.shape[0]] = prev_prediction.detach()
                            background_observations[node_type] = bg
                            print(f"[FSOI] {inst_name}: Padded previous forecast ({prev_prediction.shape[0]} → {obs_values.shape[0]})")
                        else:
                            # Previous had more obs - use first N
                            background_observations[node_type] = prev_prediction[:obs_values.shape[0]].detach()
                            print(f"[FSOI] {inst_name}: Truncated previous forecast ({prev_prediction.shape[0]} → {obs_values.shape[0]})")
                    else:
                        # Channel mismatch, use zero background
                        background_observations[node_type] = torch.zeros_like(obs_values)
                        print(f"[FSOI] {inst_name}: Channel mismatch, using zero background")
                else:
                    # No previous prediction available, use zero background
                    background_observations[node_type] = torch.zeros_like(obs_values)
                    print(f"[FSOI] {inst_name}: No previous prediction, using zero background")
    
    elif use_model_background:
        # FALLBACK: Use model's prediction from same window
        print("[FSOI] Climatological approximation: Using same-window predictions as x_b")
        
        with torch.no_grad():
            for node_type in batch.node_types:
                if not node_type.endswith("_input"):
                    continue
                
                inst_name = node_type.replace("_input", "")
                
                # Filter: Only process radiosonde (or target_instrument if specified)
                if conventional_only and inst_name not in ['radiosonde']:
                    continue
                if target_instrument is not None and inst_name != target_instrument:
                    continue
                
                obs_values = batch[node_type].x
                
                # Check if we have predictions for this instrument
                target_key = f"{inst_name}_target"
                if target_key in predictions and len(predictions[target_key]) > 0:
                    # Use model's prediction as background (what model expects)
                    # Take first prediction step as background
                    model_prediction = predictions[target_key][0]  # [N, C]
                    
                    # In observation space, the background is what model predicts
                    # For instruments that predict themselves, this is direct
                    if model_prediction.shape == obs_values.shape:
                        background_observations[node_type] = model_prediction.detach()
                    else:
                        # If shapes don't match, use zero (no background available)
                        background_observations[node_type] = torch.zeros_like(obs_values)
                else:
                    # No prediction available, use zero background
                    background_observations[node_type] = torch.zeros_like(obs_values)
    else:
        # Fall back to zero background (climatology)
        print("[FSOI] Using zero background (climatological)")
        for node_type in batch.node_types:
            if node_type.endswith("_input"):
                inst_name = node_type.replace("_input", "")
                
                # Filter: Only process radiosonde (or target_instrument if specified)
                if conventional_only and inst_name not in ['radiosonde']:
                    continue
                if target_instrument is not None and inst_name != target_instrument:
                    continue
                    
                background_observations[node_type] = torch.zeros_like(batch[node_type].x)
    
    # Create background batch with x_b
    batch_background = batch.clone()
    model.zero_grad()
    
    for node_type in batch_background.node_types:
        if node_type.endswith("_input"):
            if node_type in background_observations:
                background = background_observations[node_type]
                # Set background state and enable gradients
                batch_background[node_type].x = background.clone().requires_grad_(True)
    
    # Clear any cached gradients before forward pass
    torch.cuda.empty_cache()
    
    # Forward pass at background state with gradients enabled
    with torch.enable_grad():
        preds_background = model(batch_background)
        
        # Compute loss at background
        loss_background = None  # Will be initialized when first loss is computed
        num_predictions_bg = 0
        
        for node_type, preds_list in preds_background.items():
            if node_type not in ground_truths:
                continue
            
            gts_list = ground_truths[node_type]
            
            if "_target_step" in node_type:
                inst_name = node_type.split("_target_step")[0]
            else:
                inst_name = node_type.replace("_target", "")
            
            # Filter: Only process radiosonde (or target_instrument if specified)
            if conventional_only and inst_name not in ['radiosonde']:
                continue
            if target_instrument is not None and inst_name != target_instrument:
                continue
            
            inst_id = model.instrument_name_to_id.get(inst_name, None)
            instrument_weight = instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0
            
            for step, (y_pred, y_true) in enumerate(zip(preds_list, gts_list)):
                if y_pred is None or y_true is None:
                    continue
                
                valid_mask = None
                step_info = model._get_latent_step_info(batch_background)
                step_mapping = step_info["step_mapping"]
                if node_type in step_mapping and step in step_mapping[node_type]:
                    step_node_type = step_mapping[node_type][step]
                    if hasattr(batch_background[step_node_type], "target_channel_mask"):
                        valid_mask = batch_background[step_node_type].target_channel_mask
                
                channel_loss = weighted_huber_loss(
                    y_pred, y_true,
                    instrument_ids=None,
                    channel_weights=channel_weights,
                    delta=0.1,
                    rebalancing=True,
                    valid_mask=valid_mask,
                )
                
                if torch.isfinite(channel_loss):
                    weighted_loss = channel_loss * instrument_weight
                    # Initialize or accumulate loss
                    if loss_background is None:
                        loss_background = weighted_loss
                    else:
                        loss_background = loss_background + weighted_loss
                    num_predictions_bg += 1
        
        # Check if we have any valid predictions at background
        if loss_background is None or num_predictions_bg == 0:
            print("[FSOI] Warning: No valid predictions at background state")
            # Fall back to using only analysis gradient
            sensitivities_background = {k: torch.zeros_like(v) for k, v in sensitivities_analysis.items()}
        else:
            avg_loss_background = loss_background / num_predictions_bg
            avg_loss_background.backward()
        
        # Extract M_b^T (gradients at background)
        sensitivities_background = {}
        for node_type in batch_background.node_types:
            if node_type.endswith("_input"):
                if batch_background[node_type].x.grad is not None:
                    sensitivities_background[node_type] = batch_background[node_type].x.grad.clone().detach()
                else:
                    sensitivities_background[node_type] = torch.zeros_like(batch_background[node_type].x)
    
    # === STEP 3: Combine using two-state adjoint formula ===
    # Equation: Δe = (x_a - x_b)^T · g
    # where g = M_b^T·C·(M(x_b) - x_ref) + M_a^T·C·(M(x_a) - x_ref)
    #
    # Our gradients from backprop already contain M^T·(M(x) - x_ref)
    # So: g = sens_b + sens_a
    
    fsoi_values = {}
    for node_type in batch.node_types:
        if not node_type.endswith("_input"):
            continue
        
        # x_a = actual observations (analysis)
        obs_analysis = batch[node_type].x
        
        # x_b = model's background/forecast (from background_observations computed above)
        obs_background = background_observations.get(node_type, torch.zeros_like(obs_analysis))
        
        # Innovation: x_a - x_b
        # "actual obs" - "what model predicted/expected"
        # This measures how much observations CORRECT the model's forecast
        increment = obs_analysis - obs_background
        
        # g = M_b^T·e_b + M_a^T·e_a (sum of two adjoints weighted by errors)
        # Our backprop gives us these automatically
        sens_b = sensitivities_background.get(node_type, torch.zeros_like(obs_analysis))
        sens_a = sensitivities_analysis.get(node_type, torch.zeros_like(obs_analysis))
        sum_sensitivity = sens_b + sens_a  # M_b^T·e_b + M_a^T·e_a
        
        # FSOI per observation i: FSOI_i = (x_a,i - x_b,i) * g_i
        # Sign convention: Negative = beneficial (reduces error)
        fsoi = -increment * sum_sensitivity
        
        fsoi_values[node_type] = fsoi
    
    # Aggressive memory cleanup
    del batch_background, preds_background, preds_analysis
    del sensitivities_background, sensitivities_analysis
    if 'loss_background' in locals():
        del loss_background
    if 'loss_analysis' in locals():
        del loss_analysis
    torch.cuda.empty_cache()
    
    return fsoi_values


def compute_fsoi(
    sensitivities: Dict[str, torch.Tensor],
    batch: HeteroData,
    background_estimate: Optional[Dict[str, torch.Tensor]] = None,
    feature_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute FSOI (Forecast Sensitivity to Observation Impact).

    Mathematical formulation (following Langland & Baker 2004):
        FSOI_i = -∇_{o_i} L · (o_i - o_background_i)

    Sign Convention:
        Negative FSOI → observation reduces forecast error (beneficial)
        Positive FSOI → observation increases forecast error (detrimental)
    
    Background Options:
        1. Climatological mean (default): background = 0 in normalized space
           - Observations normalized as (obs - 3yr_mean) / 3yr_std
           - Innovation = how much obs deviates from climatology
           
        2. Model background (operational): Use previous forecast as background
           - Requires background_estimate parameter
           - Innovation = obs - first_guess (true innovation)
           - More accurate for operational FSOI (see fsoi_operational.py)

    Args:
        sensitivities: Gradient of loss w.r.t. input observations (from backpropagation)
        batch: Input batch containing observation values (already normalized)
        background_estimate: Background/prior estimate for each observation type.
                           If None, uses climatological mean (normalized space = 0).
        feature_stats: Climatological statistics (mean, std) from observation_config.yaml.
                      Used when background_estimate is None.

    Returns:
        Dictionary mapping node_type -> FSOI values [N, C]
        
    Note:
        This fast approximation computes sensitivity at the analysis state (all obs included).
        For exact FSOI calculations, alternative methodologies may be explored.
    """
    fsoi_values = {}

    for node_type, sensitivity in sensitivities.items():
        if node_type not in batch.node_types:
            continue

        # Get observation values (already normalized)
        obs_values = batch[node_type].x  # [N, C]

        # Determine background
        if background_estimate is not None and node_type in background_estimate:
            background = background_estimate[node_type]
        else:
            # Use climatological mean as background
            # Since observations are normalized as (obs - clim_mean) / clim_std,
            # the climatological mean in normalized space is 0
            # This is the 3-year average used in normalization
            background = torch.zeros_like(obs_values)

        # Observation increment (innovation from climatology)
        # This measures how much the observation deviates from the 3-year mean
        increment = obs_values - background

        # FSOI = -gradient · increment (sign convention: negative = beneficial)
        fsoi = -sensitivity * increment  # Element-wise product [N, C]

        fsoi_values[node_type] = fsoi

    return fsoi_values


def aggregate_fsoi_statistics(
    fsoi_values: Dict[str, torch.Tensor],
    batch: HeteroData,
    observation_config: dict,
) -> pd.DataFrame:
    """
    Aggregate FSOI values into summary statistics.

    Args:
        fsoi_values: FSOI tensors for each observation type
        batch: Batch containing metadata (lat, lon, time, etc.)
        observation_config: Configuration with instrument information

    Returns:
        DataFrame with columns:
            - instrument: instrument name
            - obs_id: observation index
            - lat, lon: location
            - channel: channel number
            - fsoi_value: FSOI value for this obs/channel
            - sensitivity: gradient magnitude
            - increment: observation increment magnitude
    """
    records = []

    for node_type, fsoi_tensor in fsoi_values.items():
        # Parse instrument name
        inst_name = node_type.replace("_input", "")

        # Get metadata if available
        metadata = None
        if hasattr(batch[node_type], "input_metadata"):
            metadata = batch[node_type].input_metadata  # [N, meta_dim]

        # Get observation values
        obs_values = batch[node_type].x  # [N, C]
        N, C = fsoi_tensor.shape

        # Convert to numpy
        fsoi_np = fsoi_tensor.cpu().numpy()
        obs_np = obs_values.cpu().numpy()

        # Extract lat/lon if available (usually first 2 metadata columns)
        if metadata is not None:
            meta_np = metadata.cpu().numpy()
            lats = np.degrees(meta_np[:, 0]) if meta_np.shape[1] > 0 else np.zeros(N)
            lons = np.degrees(meta_np[:, 1]) if meta_np.shape[1] > 1 else np.zeros(N)
        else:
            lats = np.zeros(N)
            lons = np.zeros(N)

        # Get feature names for this instrument
        feature_names = None
        for obs_type, instruments in observation_config.items():
            if inst_name in instruments:
                feature_names = instruments[inst_name].get("features", None)
                break

        # Create records for each observation and channel
        for obs_idx in range(N):
            for ch_idx in range(C):
                # Get the feature name
                if feature_names is not None and ch_idx < len(feature_names):
                    feature_name = feature_names[ch_idx]
                else:
                    feature_name = f"ch{ch_idx+1}"
                
                # For plotting: use channel INDEX (0-based) so we can aggregate properly
                # The feature_name is saved for reference but channel number is used for grouping
                channel_num = ch_idx + 1  # 1-based channel number
                
                records.append(
                    {
                        "instrument": inst_name,
                        "obs_id": obs_idx,
                        "lat": lats[obs_idx],
                        "lon": lons[obs_idx],
                        "channel": channel_num,  # Use integer channel number
                        "channel_name": feature_name,  # Keep full name for reference
                        "fsoi_value": fsoi_np[obs_idx, ch_idx],
                        "obs_value": obs_np[obs_idx, ch_idx],
                    }
                )

    return pd.DataFrame(records)


def compute_batch_fsoi(
    model,
    batch: HeteroData,
    save_dir: str = "fsoi_results",
    epoch: int = 0,
    batch_idx: int = 0,
    compute_sensitivity: bool = True,
    feature_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    conventional_only: bool = False,
    previous_batch: Optional[HeteroData] = None,
    previous_predictions: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Complete FSOI computation pipeline for a single batch using two-state adjoint methodology.

    This function implements the two-state adjoint FSOI approach (Laloyaux et al. 2025,
    arXiv:2510.27388, Equation 3), which computes:
    
        e(x_a) - e(x_b) ≈ (x_a - x_b)^T · [M_b^T·C·e_b + M_a^T·C·e_a]
    
    Where:
        x_b = background state (model first-guess without observations)
        x_a = analysis state (with observations)
        (x_a - x_b) = innovation (observation increment from model background)
        M_b^T = adjoint at background
        M_a^T = adjoint at analysis
        e_b, e_a = forecast errors at background and analysis
    
    Innovation Definition:
        (x_a - x_b) where x_b is the 12h forecast from previous window
        - Measures observation value relative to MODEL'S OWN FORECAST
        - In operational DA: x_b would be the previous 6h analysis + 12h forecast
        - In this GNN: x_b ≈ 0 (model predictions with zero observations = cold start)
        
        This differs from climatological FSOI:
        - Climatological: (obs - 3yr_mean) measures deviation from climate
        - Operational: (obs - first_guess) measures model correction
    
    Adjoint Summation (NOT Averaging):
        FSOI = (x_a - x_b)^T · (M_b^T·e_b + M_a^T·e_a)
        - We SUM the two adjoints
        - No division by 2
        - The two-state sum reduces contamination vs single-state
    
    This two-state formulation reduces contamination compared to single-state adjoint
    via the sum of sensitivities from both states. However, it's still an approximation 
    (3rd-order Taylor expansion).
    
    Implementation Details (Equation 3):
    - Background state: Model first-guess in observation space, approximated by 
                        current-window predictions where available, otherwise 0
    - Analysis state: Current observations (x_a)
    - Innovation: x_a - x_b (observation minus model background)
    - Two forward passes: One at background, one at analysis
    - Two backward passes: Compute M_b^T·e_b and M_a^T·e_a via backprop
    - Adjoint sum: M_b^T·e_b + M_a^T·e_a (sum, not average)
    - FSOI = -(x_a - x_b) · (M_b^T + M_a^T)
    
    Use Cases:
    - Fast FSOI during training for monitoring
    - Exploratory analysis of observation importance
    - Comparison with two-state adjoint results (Laloyaux et al. 2025)
    - Debugging and visualization
    
    Pipeline Steps:
    1. Forward pass at analysis: Generate predictions with all observations (x_a)
    2. Backward pass at analysis: Compute M_a^T·e_a via adjoint (∂L/∂x_a)
    3. Forward pass at background: Generate predictions at background state (x_b, approximated)
    4. Backward pass at background: Compute M_b^T·e_b via adjoint (∂L/∂x_b)
    5. FSOI calculation: FSOI = -(x_a - x_b) · (M_b^T·e_b + M_a^T·e_a)
    6. Aggregation: Collect statistics by instrument/channel
    7. Export: Save to CSV with metadata (lat, lon, channel, etc.)

    Args:
        model: Trained GNN model
        batch: Input batch containing observations
        save_dir: Directory to save FSOI results (default: "fsoi_results")
        epoch: Current training epoch (for filename)
        batch_idx: Batch index (for filename)
        compute_sensitivity: If True, compute gradients (requires more memory)
        feature_stats: Climatological statistics from observation_config.yaml
                      Used if background_estimate not provided (default: zero obs)
        conventional_only: If True, only compute FSOI for conventional obs (surface_obs + radiosonde)
                          representing prognostic variables (u, v, T, q, p).
                          Analogous to C-matrix filtering (Figure 5, Laloyaux et al. 2025).
        previous_batch: Optional batch from previous sequential window
        previous_predictions: Optional model predictions from previous_batch (x_b in Eq. 3, Laloyaux et al. 2025)
                             If provided, uses forecast from previous window as background
                             If None, falls back to climatological background (x_b = 0 or same-window)

    Returns:
        Tuple of:
        - fsoi_values: Dict mapping node_type -> FSOI tensor [N, C]
        - stats_df: DataFrame with columns [instrument, obs_id, lat, lon, channel, 
                    fsoi_value, obs_value, target_variable (if conventional_only)]
                    
    References:
        - Laloyaux et al. (2025): Using DA tools to dissect AI models, arXiv:2510.27388, Equation 3
        - Langland & Baker (2004): Estimation of observation impact using adjoint
    """
    os.makedirs(save_dir, exist_ok=True)

    # Enable gradient computation for input observations
    for node_type in batch.node_types:
        if node_type.endswith("_input"):
            batch[node_type].x.requires_grad_(True)

    # Forward pass with gradients enabled (model is in eval mode during validation)
    with torch.enable_grad():
        predictions = model(batch)

    # Extract ground truths
    ground_truth_data = model._extract_ground_truths_and_metadata(batch, predictions)

    # Convert to simple dict for FSOI computation
    ground_truths = {}
    for node_type, gt_data in ground_truth_data.items():
        ground_truths[node_type] = gt_data["gts_list"]

    if compute_sensitivity:
        # Use two-state adjoint formulation
        # Computes FSOI = (x_a - x_b)^T · (M_b^T·e_b + M_a^T·e_a)
        # where x_b = model's prediction (background approximation), x_a = actual observations
        fsoi_values = compute_observation_sensitivity_two_state(
            model=model,
            batch=batch,
            predictions=predictions,
            ground_truths=ground_truths,
            instrument_weights=model.instrument_weights,
            channel_weights=model.channel_weights,
            conventional_only=conventional_only,
            use_model_background=True,  # Use model's forecast as x_b
            previous_batch=previous_batch,  # For sequential: forecast from previous window
            previous_predictions=previous_predictions,  # x_b in two-state formulation
        )

        # Aggregate statistics
        stats_df = aggregate_fsoi_statistics(
            fsoi_values=fsoi_values, batch=batch, observation_config=model.observation_config
        )
        
        # Add target_variable column if conventional_only
        if conventional_only:
            # Determine which conventional targets exist
            conventional_targets = []
            for node_type in predictions.keys():
                inst_name = node_type.replace("_target", "").split("_target_step")[0]
                if inst_name in ['surface_obs', 'radiosonde'] and inst_name not in conventional_targets:
                    conventional_targets.append(inst_name)
            
            # Mark all stats as targeting conventional obs
            if len(conventional_targets) == 1:
                stats_df['target_variable'] = conventional_targets[0]
            else:
                stats_df['target_variable'] = 'conventional'  # Both surface + radiosonde

        # Save to CSV
        csv_path = os.path.join(save_dir, f"fsoi_epoch{epoch}_batch{batch_idx}.csv")
        stats_df.to_csv(csv_path, index=False)
        print(f"[FSOI] Saved statistics to {csv_path}")

        return fsoi_values, stats_df
    else:
        # Skip sensitivity computation (for faster evaluation)
        return {}, pd.DataFrame()


def summarize_fsoi_by_instrument(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize FSOI statistics by instrument.

    Args:
        stats_df: DataFrame from aggregate_fsoi_statistics

    Returns:
        Summary DataFrame with mean/std FSOI per instrument
    """
    summary = (
        stats_df.groupby("instrument")
        .agg(
            {
                "fsoi_value": ["mean", "std", "min", "max", "sum"],
                "obs_id": "count",  # number of observations
            }
        )
        .reset_index()
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in summary.columns]

    return summary


def summarize_fsoi_by_channel(stats_df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Summarize FSOI statistics by channel for a specific instrument.

    Args:
        stats_df: DataFrame from aggregate_fsoi_statistics
        instrument: Instrument name to filter

    Returns:
        Summary DataFrame with mean/std FSOI per channel
    """
    inst_df = stats_df[stats_df["instrument"] == instrument]

    summary = (
        inst_df.groupby("channel")
        .agg(
            {
                "fsoi_value": ["mean", "std", "min", "max", "sum"],
                "obs_id": "count",
            }
        )
        .reset_index()
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in summary.columns]

    return summary


def identify_high_impact_observations(
    stats_df: pd.DataFrame, top_n: int = 100, impact_type: str = "beneficial"
) -> pd.DataFrame:
    """
    Identify observations with highest (or lowest) FSOI values.

    Args:
        stats_df: DataFrame from aggregate_fsoi_statistics
        top_n: Number of top observations to return
        impact_type: "beneficial" (negative FSOI, reduces error) 
                    or "detrimental" (positive FSOI, increases error)

    Returns:
        DataFrame with top_n observations sorted by FSOI
        
    Note:
        Sign convention: Negative FSOI = beneficial observation
    """
    if impact_type == "beneficial":
        # Most beneficial = most negative FSOI
        sorted_df = stats_df.sort_values("fsoi_value", ascending=True).head(top_n)
    elif impact_type == "detrimental":
        # Most detrimental = most positive FSOI
        sorted_df = stats_df.sort_values("fsoi_value", ascending=False).head(top_n)
    else:
        raise ValueError(f"Unknown impact_type: {impact_type}")

    return sorted_df.reset_index(drop=True)


def compute_fsoi_spatial_statistics(stats_df: pd.DataFrame, lat_bins: int = 18, lon_bins: int = 36) -> pd.DataFrame:
    """
    Compute spatial statistics of FSOI values on a lat/lon grid.

    Args:
        stats_df: DataFrame from aggregate_fsoi_statistics
        lat_bins: Number of latitude bins
        lon_bins: Number of longitude bins

    Returns:
        DataFrame with gridded FSOI statistics
    """
    # Create lat/lon bins
    stats_df["lat_bin"] = pd.cut(stats_df["lat"], bins=lat_bins, labels=False)
    stats_df["lon_bin"] = pd.cut(stats_df["lon"], bins=lon_bins, labels=False)

    # Aggregate by grid cell
    grid_stats = (
        stats_df.groupby(["lat_bin", "lon_bin"])
        .agg(
            {
                "fsoi_value": ["mean", "std", "sum", "count"],
                "lat": "mean",
                "lon": "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    grid_stats.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in grid_stats.columns]

    return grid_stats
