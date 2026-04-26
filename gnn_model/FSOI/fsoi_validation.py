"""
FSOI Validation - Finite-difference sanity checks for FSOI implementation.

This module provides validation tests to ensure gradients are computed correctly
by comparing automatic differentiation (autograd) with finite differences.

Author: Azadeh Gholoubi
"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import Dict


# Allow running this module from gnn_model/FSOI/ while importing from parent.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def finite_difference_check(
    model,
    batch,
    inst_name: str,
    obs_idx: int,
    channel_idx: int,
    epsilon: float = 1e-4,
    forecast_step: int = 0,
) -> Dict[str, float]:
    """
    Verify gradient computation using finite differences.

    Perturb one observation value by ε and check that:
    1. Forecast error changes
    2. Gradient direction matches finite difference
    3. Gradient magnitude is reasonable

    This is the "gold standard" validation that gradients are correct.

    Args:
        model: GNN model (frozen)
        batch: Current batch
        inst_name: Instrument to test
        obs_idx: Observation index to perturb
        channel_idx: Channel index to perturb
        epsilon: Perturbation size
        forecast_step: Forecast lead time

    Returns:
        Dict with validation metrics
    """
    from fsoi_utils import get_fsoi_inputs, compute_forecast_error
    from fsoi_model_extensions import freeze_model_for_fsoi

    freeze_model_for_fsoi(model)
    device = next(model.parameters()).device
    batch = batch.to(device)

    print(f"\n{'='*80}")
    print(f"FINITE DIFFERENCE VALIDATION")
    print(f"  Instrument: {inst_name}")
    print(f"  Observation: {obs_idx}, Channel: {channel_idx}")
    print(f"  Epsilon: {epsilon}")
    print(f"{'='*80}\n")

    # Get original inputs
    xa_original = get_fsoi_inputs(
        batch,
        model.observation_config,
        model.instrument_name_to_id,
    )

    if inst_name not in xa_original:
        print(f"[ERROR] Instrument {inst_name} not found in batch")
        return {}

    x_orig = xa_original[inst_name]

    # Check indices
    if obs_idx >= x_orig.shape[0] or channel_idx >= x_orig.shape[1]:
        print(f"[ERROR] Invalid indices: obs_idx={obs_idx}, channel_idx={channel_idx}, "
              f"shape={x_orig.shape}")
        return {}

    original_value = x_orig[obs_idx, channel_idx].item()
    print(f"Original value: {original_value:.6f}")

    # ============================================================================
    # STEP 1: Compute gradient using autograd
    # ============================================================================
    print("\n[1/3] Computing gradient via autograd...")

    xa = xa_original.copy()
    xa[inst_name] = x_orig.clone().detach()
    xa[inst_name].requires_grad_(True)

    # Replace batch inputs
    from fsoi_utils import replace_batch_inputs
    batch_grad = batch.clone()
    replace_batch_inputs(batch_grad, xa)

    # Compute error
    error_grad = compute_forecast_error(
        model,
        batch_grad,
        forecast_lead_step=forecast_step,
        instrument_weights=model.instrument_weights,
        channel_weights=model.channel_weights,
        use_area_weights=True,
        target_instruments=[inst_name],
    )

    # Compute gradient
    gradient = torch.autograd.grad(error_grad, xa[inst_name])[0]
    grad_value = gradient[obs_idx, channel_idx].item()

    print(f"  Error (original): {error_grad.item():.6e}")
    print(f"  Gradient at [{obs_idx},{channel_idx}]: {grad_value:.6e}")

    # ============================================================================
    # STEP 2: Compute finite difference (forward)
    # ============================================================================
    print("\n[2/3] Computing finite difference (forward)...")

    # Perturb input by +ε
    xa_plus = xa_original.copy()
    x_plus = x_orig.clone().detach()
    x_plus[obs_idx, channel_idx] += epsilon
    xa_plus[inst_name] = x_plus

    batch_plus = batch.clone()
    replace_batch_inputs(batch_plus, xa_plus)

    # Compute error with perturbation
    with torch.no_grad():
        error_plus = compute_forecast_error(
            model,
            batch_plus,
            forecast_lead_step=forecast_step,
            instrument_weights=model.instrument_weights,
            channel_weights=model.channel_weights,
            use_area_weights=True,
            target_instruments=[inst_name],
        ).item()

    print(f"  Error (x + ε): {error_plus:.6e}")

    # ============================================================================
    # STEP 3: Compute finite difference (backward for better estimate)
    # ============================================================================
    print("\n[3/3] Computing finite difference (backward)...")

    # Perturb input by -ε
    xa_minus = xa_original.copy()
    x_minus = x_orig.clone().detach()
    x_minus[obs_idx, channel_idx] -= epsilon
    xa_minus[inst_name] = x_minus

    batch_minus = batch.clone()
    replace_batch_inputs(batch_minus, xa_minus)

    with torch.no_grad():
        error_minus = compute_forecast_error(
            model,
            batch_minus,
            forecast_lead_step=forecast_step,
            instrument_weights=model.instrument_weights,
            channel_weights=model.channel_weights,
            use_area_weights=True,
            target_instruments=[inst_name],
        ).item()

    print(f"  Error (x - ε): {error_minus:.6e}")

    # ============================================================================
    # STEP 4: Compare results
    # ============================================================================
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    # Finite difference approximations
    fd_forward = (error_plus - error_grad.item()) / epsilon
    fd_central = (error_plus - error_minus) / (2 * epsilon)

    print(f"\nGradient (autograd):       {grad_value:.6e}")
    print(f"Finite diff (forward):     {fd_forward:.6e}")
    print(f"Finite diff (central):     {fd_central:.6e}")

    # Compute relative errors
    rel_error_forward = abs(grad_value - fd_forward) / (abs(grad_value) + 1e-10)
    rel_error_central = abs(grad_value - fd_central) / (abs(grad_value) + 1e-10)

    print(f"\nRelative error (forward):  {rel_error_forward:.6e}")
    print(f"Relative error (central):  {rel_error_central:.6e}")

    # Check if errors are within acceptable tolerance
    tolerance = 1e-3  # 0.1% relative error

    if rel_error_central < tolerance:
        print(f"\n✓ PASS: Gradient validation successful (error < {tolerance})")
        status = "PASS"
    elif rel_error_central < 0.01:
        print(f"\n⚠ WARNING: Gradient error {rel_error_central:.2%} exceeds {tolerance*100}% but < 1%")
        print("  This may be acceptable depending on problem scale")
        status = "WARNING"
    else:
        print(f"\n✗ FAIL: Gradient error {rel_error_central:.2%} too large!")
        print("  Check implementation: graph structure, metadata, gradient flow")
        status = "FAIL"

    print("="*80 + "\n")

    return {
        'inst_name': inst_name,
        'obs_idx': obs_idx,
        'channel_idx': channel_idx,
        'original_value': original_value,
        'epsilon': epsilon,
        'error_original': error_grad.item(),
        'error_plus': error_plus,
        'error_minus': error_minus,
        'gradient_autograd': grad_value,
        'gradient_fd_forward': fd_forward,
        'gradient_fd_central': fd_central,
        'rel_error_forward': rel_error_forward,
        'rel_error_central': rel_error_central,
        'status': status,
    }


def validate_fsoi_gradients(
    model,
    prev_batch,
    curr_batch,
    num_samples: int = 3,
    epsilon: float = 1e-4,
) -> bool:
    """
    Run finite-difference validation on multiple random observations.

    Args:
        model: GNN model
        prev_batch: Previous window batch
        curr_batch: Current window batch
        num_samples: Number of random obs to test
        epsilon: Perturbation size

    Returns:
        True if all tests pass, False otherwise
    """
    from fsoi_utils import get_fsoi_inputs

    print("\n" + "="*80)
    print("FSOI GRADIENT VALIDATION - FINITE DIFFERENCE TESTS")
    print("="*80)
    print(f"Testing {num_samples} random observations")
    print("="*80 + "\n")

    # Get inputs from current batch
    xa = get_fsoi_inputs(
        curr_batch,
        model.observation_config,
        model.instrument_name_to_id,
    )

    if not xa:
        print("[ERROR] No inputs found in batch")
        return False

    # Select random observations to test
    results = []

    for sample_idx in range(num_samples):
        # Pick random instrument
        inst_name = np.random.choice(list(xa.keys()))
        x = xa[inst_name]

        # Pick random observation and channel
        obs_idx = np.random.randint(0, x.shape[0])
        channel_idx = np.random.randint(0, x.shape[1])

        print(f"\nSample {sample_idx + 1}/{num_samples}")

        # Run finite difference check
        result = finite_difference_check(
            model,
            curr_batch,
            inst_name,
            obs_idx,
            channel_idx,
            epsilon=epsilon,
        )

        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r.get('status') == 'PASS')
    warned = sum(1 for r in results if r.get('status') == 'WARNING')
    failed = sum(1 for r in results if r.get('status') == 'FAIL')

    print(f"\nTests run:    {len(results)}")
    print(f"✓ Passed:     {passed}")
    print(f"⚠ Warnings:   {warned}")
    print(f"✗ Failed:     {failed}")

    if failed == 0:
        print("\n✓ All gradient validation tests passed!")
        print("  FSOI gradients are computed correctly.")
        return True
    else:
        print(f"\n✗ {failed} gradient validation tests failed!")
        print("  Review implementation:")
        print("    1. Check graph construction (encoder/decoder edges)")
        print("    2. Verify all metadata is copied correctly")
        print("    3. Ensure requires_grad=True for inputs")
        print("    4. Check for detach() calls breaking gradient flow")
        return False


if __name__ == "__main__":
    print("This module provides validation functions for FSOI.")
    print("Import and use in fsoi_inference.py or test_fsoi.py")
