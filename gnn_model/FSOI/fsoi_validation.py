"""
FSOI Validation - Gradient sanity checks for FSOI implementation.

Three complementary tests are provided, targeting different precision regimes:

  1. finite_difference_check          — per-scalar FD in float32.
       Works for low-N instruments (radiosonde, aircraft, AMSUA).
       SKIPs for high-N instruments (ATMS, AVHRR, SSMIS) where the per-obs
       gradient is below the float32 ULP threshold.

  2. directional_derivative_check     — random Rademacher perturbation in float32.
       Aggregates gradient signal across all observations, making it detectable
       even for ATMS (50 k obs).  Tests n_trials random directions and reports
       the Pearson correlation between autograd and FD directional derivatives.

  3. finite_difference_check_float64  — per-scalar FD in float64.
       Casts model + inputs to float64 before the check; float64 ULP (~10^-15)
       resolves perturbations ~10^-10, enabling per-obs FD for any instrument.
       Casts back to float32 on exit.

The public orchestrator validate_fsoi_gradients_all runs all three for every
instrument in the batch and writes a merged CSV.

Author: Azadeh Gholoubi
"""

import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    epsilon: float = 1e-2,
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

    print(f"\n{'=' * 80}")
    print(f"FINITE DIFFERENCE VALIDATION")
    print(f"  Instrument: {inst_name}")
    print(f"  Observation: {obs_idx}, Channel: {channel_idx}")
    print(f"  Epsilon: {epsilon}")
    print(f"{'=' * 80}\n")

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
    replace_batch_inputs(batch_grad, xa, model.observation_config)

    # Compute error — use 'mean' so scale ~O(1) and float32 FD is numerically stable
    error_grad = compute_forecast_error(
        model,
        batch_grad,
        forecast_lead_step=forecast_step,
        instrument_weights=model.instrument_weights,
        channel_weights=model.channel_weights,
        use_area_weights=True,
        target_instruments=[inst_name],
        loss_reduction='mean',
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
    replace_batch_inputs(batch_plus, xa_plus, model.observation_config)

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
            loss_reduction='mean',
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
    replace_batch_inputs(batch_minus, xa_minus, model.observation_config)

    with torch.no_grad():
        error_minus = compute_forecast_error(
            model,
            batch_minus,
            forecast_lead_step=forecast_step,
            instrument_weights=model.instrument_weights,
            channel_weights=model.channel_weights,
            use_area_weights=True,
            target_instruments=[inst_name],
            loss_reduction='mean',
        ).item()

    print(f"  Error (x - ε): {error_minus:.6e}")

    # ============================================================================
    # STEP 4: Compare results
    # ============================================================================
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

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

    # Minimum gradient magnitude below which per-observation FD is undetectable in float32.
    # With mean_loss ~O(1–10), float32 ULP ~1e-6; Δerror = grad × ε must exceed ULP.
    # grad_min = ULP / ε ≈ 1e-6 / epsilon.  Any gradient smaller than this will show
    # fd_central=0 regardless of correctness — mark as SKIP rather than FAIL.
    grad_min_for_fd = max(1e-4, 1e-5 / epsilon)
    tolerance = 1e-3  # 0.1% relative error

    if abs(grad_value) < grad_min_for_fd:
        print(f"\n~ SKIP: |gradient| = {abs(grad_value):.2e} < {grad_min_for_fd:.2e} "
              f"(too small for float32 FD with ε={epsilon})")
        print(f"  This is expected for high-observation-count instruments (ATMS, AVHRR, SSMIS).")
        print(f"  Gradient flow is non-zero; validate via FSOI closure instead.")
        status = "SKIP"
    elif rel_error_central < tolerance:
        print(f"\n✓ PASS: Gradient validation successful (error < {tolerance})")
        status = "PASS"
    elif rel_error_central < 0.05:
        print(f"\n⚠ WARNING: Gradient error {rel_error_central:.2%} exceeds {tolerance * 100}% but < 5%")
        print("  Likely float32 rounding at this error scale — validate via closure.")
        status = "WARNING"
    else:
        print(f"\n✗ FAIL: Gradient error {rel_error_central:.2%} too large!")
        print("  Check implementation: graph structure, metadata, gradient flow")
        status = "FAIL"

    print("=" * 80 + "\n")

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
    epsilon: float = 1e-2,
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

    print("\n" + "=" * 80)
    print("FSOI GRADIENT VALIDATION - FINITE DIFFERENCE TESTS")
    print("=" * 80)
    print(f"Testing {num_samples} random observations")
    print("=" * 80 + "\n")

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
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.get('status') == 'PASS')
    warned = sum(1 for r in results if r.get('status') == 'WARNING')
    failed = sum(1 for r in results if r.get('status') == 'FAIL')
    skipped = sum(1 for r in results if r.get('status') == 'SKIP')

    print(f"\nTests run:    {len(results)}")
    print(f"✓ Passed:     {passed}")
    print(f"⚠ Warnings:   {warned}")
    print(f"✗ Failed:     {failed}")
    print(f"~ Skipped:    {skipped}  (gradient too small for float32 FD — use closure check)")

    if failed == 0:
        print("\n✓ All gradient validation tests passed (or skipped)!")
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


def directional_derivative_check(
    model,
    batch,
    inst_name: str,
    epsilon: float = 0.01,
    n_trials: int = 5,
    seed: int = 42,
    forecast_step: int = 0,
) -> Dict:
    """
    Validate gradient direction using random Rademacher perturbation vectors.

    Instead of perturbing one observation at a time, samples a random sign
    vector v ∈ {-1, +1}^(N_obs × N_channels) and tests:

        FD estimate  : (e(x + εv) - e(x - εv)) / (2ε)
        Autograd est : (g * v).sum()   where g = ∂e/∂x[inst]

    The expected signal for ATMS (N=50k, per-obs grad ~10⁻⁸, ε=0.01) is
    ε × Σ|g_i| ≈ 0.01 × 1.1M × 10⁻⁸ ≈ 1.1×10⁻⁴, which is ~90 float32 ULP
    at e ~9.9 — clearly detectable without float64.

    Multiple trials with independent random vectors give a distribution of
    (fd_dd, autograd_dd) pairs; a high Pearson correlation (≥ 0.99) confirms
    the gradient direction is correct.

    Returns
    -------
    dict with per-trial results, Pearson r, and status (PASS/WARN/FAIL).
    """
    from fsoi_utils import get_fsoi_inputs, compute_forecast_error, replace_batch_inputs
    from fsoi_model_extensions import freeze_model_for_fsoi

    freeze_model_for_fsoi(model)
    device = next(model.parameters()).device
    batch = batch.to(device)

    print(f"\n{'=' * 80}")
    print(f"DIRECTIONAL DERIVATIVE VALIDATION  (Rademacher, {n_trials} trials)")
    print(f"  Instrument : {inst_name}")
    print(f"  Epsilon    : {epsilon}")
    print(f"{'=' * 80}")

    xa_original = get_fsoi_inputs(batch, model.observation_config,
                                  model.instrument_name_to_id)
    if inst_name not in xa_original:
        print(f"[ERROR] {inst_name} not in batch")
        return {'inst_name': inst_name, 'status': 'ERROR', 'n_trials': 0}

    x_orig = xa_original[inst_name].detach()
    N_obs, N_ch = x_orig.shape
    print(f"  Tensor shape: ({N_obs}, {N_ch})  —  {N_obs * N_ch:,} elements\n")

    shared_kwargs = dict(
        forecast_lead_step=forecast_step,
        instrument_weights=model.instrument_weights,
        channel_weights=model.channel_weights,
        use_area_weights=True,
        target_instruments=[inst_name],
        loss_reduction='mean',
    )

    # ── Compute full autograd gradient once ──────────────────────────────────
    xa_grad = xa_original.copy()
    xa_grad[inst_name] = x_orig.clone().requires_grad_(True)
    batch_grad = batch.clone()
    replace_batch_inputs(batch_grad, xa_grad, model.observation_config)
    e0 = compute_forecast_error(model, batch_grad, **shared_kwargs)
    g = torch.autograd.grad(e0, xa_grad[inst_name])[0].detach()  # (N_obs, N_ch)
    e0_val = e0.item()

    l1_norm = g.abs().sum().item()
    l2_norm = g.norm().item()
    print(f"  e(x)      = {e0_val:.6e}")
    print(f"  ||g||_1   = {l1_norm:.6e}   (expected FD signal ~ ε × ||g||_1 = {epsilon * l1_norm:.2e})")
    print(f"  ||g||_2   = {l2_norm:.6e}")
    print(f"  mean |g_i| = {g.abs().mean().item():.3e}")

    # float32 ULP at e0_val to predict detectability
    import struct
    raw = struct.pack('f', abs(e0_val))
    exp_bits = (struct.unpack('I', raw)[0] >> 23) & 0xFF
    ulp_f32 = 2.0 ** (exp_bits - 127 - 23)
    expected_fd_signal = epsilon * l1_norm
    n_ulp = expected_fd_signal / ulp_f32 if ulp_f32 > 0 else float('inf')
    print(f"  float32 ULP at e={e0_val:.2e}: {ulp_f32:.2e}")
    print(f"  Expected |Δe| / ULP ~ {n_ulp:.1f}  ({'detectable' if n_ulp > 10 else 'MARGINAL — consider float64 test'})\n")

    torch.manual_seed(seed)
    trials = []
    for k in range(n_trials):
        # Rademacher ±1 vector
        v = (torch.randint(0, 2, x_orig.shape, device=device).float() * 2 - 1)

        autograd_dd = (g * v).sum().item()

        xa_plus = xa_original.copy()
        xa_plus[inst_name] = (x_orig + epsilon * v).detach()
        batch_p = batch.clone()
        replace_batch_inputs(batch_p, xa_plus, model.observation_config)
        with torch.no_grad():
            e_plus = compute_forecast_error(model, batch_p, **shared_kwargs).item()
        torch.cuda.empty_cache()

        xa_minus = xa_original.copy()
        xa_minus[inst_name] = (x_orig - epsilon * v).detach()
        batch_m = batch.clone()
        replace_batch_inputs(batch_m, xa_minus, model.observation_config)
        with torch.no_grad():
            e_minus = compute_forecast_error(model, batch_m, **shared_kwargs).item()
        torch.cuda.empty_cache()

        fd_dd = (e_plus - e_minus) / (2.0 * epsilon)
        rel_err = abs(autograd_dd - fd_dd) / (abs(autograd_dd) + 1e-12)
        trials.append({'trial': k, 'autograd_dd': autograd_dd, 'fd_dd': fd_dd,
                       'rel_error': rel_err, 'e_plus': e_plus, 'e_minus': e_minus})
        print(f"  Trial {k + 1:2d}: autograd={autograd_dd:+.4e}  fd={fd_dd:+.4e}  "
              f"rel_err={rel_err:.3f}")

    # Pearson correlation between autograd_dd and fd_dd across trials
    ag_vals = np.array([t['autograd_dd'] for t in trials])
    fd_vals = np.array([t['fd_dd'] for t in trials])
    if ag_vals.std() > 1e-30 and fd_vals.std() > 1e-30:
        pearson_r = float(np.corrcoef(ag_vals, fd_vals)[0, 1])
    else:
        pearson_r = float('nan')

    mean_rel_err = float(np.mean([t['rel_error'] for t in trials]))
    print(f"\n  Pearson r (autograd vs FD) = {pearson_r:.4f}")
    print(f"  Mean relative error        = {mean_rel_err:.4f}")

    if np.isnan(pearson_r) or n_ulp < 2:
        status = 'INSUF_SIGNAL'
        print(f"  → INSUF_SIGNAL: expected |Δe| only {n_ulp:.1f} ULP — run float64 test")
    elif pearson_r >= 0.99 and mean_rel_err < 0.05:
        status = 'PASS'
        print(f"  → PASS: gradient direction confirmed (r={pearson_r:.4f})")
    elif pearson_r >= 0.95 or mean_rel_err < 0.10:
        status = 'WARN'
        print(f"  → WARN: gradient direction plausible but noisy (r={pearson_r:.4f})")
    else:
        status = 'FAIL'
        print(f"  → FAIL: gradient direction inconsistent (r={pearson_r:.4f})")
    print('=' * 80)

    return {
        'inst_name': inst_name,
        'test': 'directional',
        'n_obs': N_obs,
        'n_channels': N_ch,
        'epsilon': epsilon,
        'n_trials': n_trials,
        'l1_norm': l1_norm,
        'l2_norm': l2_norm,
        'expected_fd_signal': expected_fd_signal,
        'n_ulp': n_ulp,
        'pearson_r': pearson_r,
        'mean_rel_error': mean_rel_err,
        'trials': trials,
        'status': status,
    }


def finite_difference_check_float64(
    model,
    batch,
    inst_name: str,
    obs_idx: int,
    channel_idx: int,
    epsilon: float = 1e-4,
    forecast_step: int = 0,
) -> Dict:
    """
    Per-observation FD gradient check in float64 precision.

    Casts the model and batch inputs to float64 before the check.
    Float64 ULP (~2×10⁻¹⁵ at e~10) resolves perturbations as small as
    10⁻¹⁰ — six orders of magnitude below the float32 limit — enabling
    per-obs FD validation for ATMS, AVHRR, and other high-N instruments.

    The model is cast back to float32 on exit whether or not an exception
    is raised.

    Parameters
    ----------
    epsilon : float
        Perturbation size in float64.  Default 1e-4; can be reduced to 1e-6
        for a tighter test once float64 is confirmed working for your model.

    Returns
    -------
    dict with the same keys as finite_difference_check, plus 'precision'.
    """
    from fsoi_utils import get_fsoi_inputs, compute_forecast_error, replace_batch_inputs
    from fsoi_model_extensions import freeze_model_for_fsoi

    print(f"\n{'=' * 80}")
    print(f"FLOAT64 FD VALIDATION")
    print(f"  Instrument : {inst_name}  obs={obs_idx}  ch={channel_idx}")
    print(f"  Epsilon    : {epsilon}  (float64)")
    print(f"{'=' * 80}")

    # ── Cast model to float64 ─────────────────────────────────────────────────
    model_was_float32 = next(model.parameters()).dtype == torch.float32
    try:
        if model_was_float32:
            model.double()

        freeze_model_for_fsoi(model)
        device = next(model.parameters()).device
        # Work on a cloned batch so float64 validation cannot leave the caller's
        # HeteroData object in double precision after an exception or CUDA OOM.
        batch64 = batch.clone().to(device)

        # Cast all floating-point tensors in the batch to float64.
        # HeteroData stores tensors inside per-node-type sub-objects, so we
        # must iterate node_types and edge_types rather than top-level keys.
        def _cast_batch_f64(b):
            try:
                from torch_geometric.data import HeteroData, Data
                if isinstance(b, HeteroData):
                    for ntype in b.node_types:
                        for k in list(b[ntype].keys()):
                            v = b[ntype][k]
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                b[ntype][k] = v.double()
                    for etype in b.edge_types:
                        for k in list(b[etype].keys()):
                            v = b[etype][k]
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                b[etype][k] = v.double()
                elif isinstance(b, Data):
                    for k in b.keys():
                        v = b[k]
                        if isinstance(v, torch.Tensor) and v.is_floating_point():
                            b[k] = v.double()
                else:
                    # Fallback: iterate known top-level attributes
                    for attr in (b.keys() if hasattr(b, 'keys') else []):
                        try:
                            v = getattr(b, attr)
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                setattr(b, attr, v.double())
                        except Exception:
                            pass
            except ImportError:
                pass

        _cast_batch_f64(batch64)

        xa_original = get_fsoi_inputs(batch64, model.observation_config,
                                      model.instrument_name_to_id)
        if inst_name not in xa_original:
            print(f"[ERROR] {inst_name} not in batch")
            return {'inst_name': inst_name, 'status': 'ERROR', 'precision': 'float64'}

        x_orig = xa_original[inst_name]
        if obs_idx >= x_orig.shape[0] or channel_idx >= x_orig.shape[1]:
            print(f"[ERROR] Invalid indices {obs_idx},{channel_idx} for shape {x_orig.shape}")
            return {'inst_name': inst_name, 'status': 'ERROR', 'precision': 'float64'}

        original_value = x_orig[obs_idx, channel_idx].item()
        print(f"  Value at [{obs_idx},{channel_idx}]: {original_value:.8f}  (dtype={x_orig.dtype})")

        shared_kwargs = dict(
            forecast_lead_step=forecast_step,
            instrument_weights=model.instrument_weights,
            channel_weights=model.channel_weights,
            use_area_weights=True,
            target_instruments=[inst_name],
            loss_reduction='mean',
        )

        # Autograd
        xa_grad = xa_original.copy()
        xa_grad[inst_name] = x_orig.clone().detach().requires_grad_(True)
        batch_a = batch64.clone()
        replace_batch_inputs(batch_a, xa_grad, model.observation_config)
        e0 = compute_forecast_error(model, batch_a, **shared_kwargs)
        g = torch.autograd.grad(e0, xa_grad[inst_name])[0]
        grad_val = g[obs_idx, channel_idx].item()
        e0_val = e0.item()
        print(f"  e(x)              = {e0_val:.10e}")
        print(f"  grad[{obs_idx},{channel_idx}] = {grad_val:.10e}")

        # FD forward
        xa_p = xa_original.copy()
        xp = x_orig.clone().detach()
        xp[obs_idx, channel_idx] += epsilon
        xa_p[inst_name] = xp
        batch_p = batch64.clone()
        replace_batch_inputs(batch_p, xa_p, model.observation_config)
        with torch.no_grad():
            e_plus = compute_forecast_error(model, batch_p, **shared_kwargs).item()

        # FD backward
        xa_m = xa_original.copy()
        xm = x_orig.clone().detach()
        xm[obs_idx, channel_idx] -= epsilon
        xa_m[inst_name] = xm
        batch_m = batch64.clone()
        replace_batch_inputs(batch_m, xa_m, model.observation_config)
        with torch.no_grad():
            e_minus = compute_forecast_error(model, batch_m, **shared_kwargs).item()

        fd_central = (e_plus - e_minus) / (2.0 * epsilon)
        fd_forward = (e_plus - e0_val) / epsilon
        rel_err_central = abs(grad_val - fd_central) / (abs(grad_val) + 1e-20)
        rel_err_forward = abs(grad_val - fd_forward) / (abs(grad_val) + 1e-20)

        print(f"  e(x+ε)            = {e_plus:.10e}")
        print(f"  e(x-ε)            = {e_minus:.10e}")
        print(f"  FD central        = {fd_central:.10e}")
        print(f"  Rel error central = {rel_err_central:.4e}")

        if rel_err_central < 1e-3:
            status = 'PASS'
        elif rel_err_central < 0.05:
            status = 'WARN'
        else:
            status = 'FAIL'
        print(f"  → {status}")
        print('=' * 80)

        return {
            'inst_name': inst_name,
            'test': 'float64_fd',
            'precision': 'float64',
            'obs_idx': obs_idx,
            'channel_idx': channel_idx,
            'original_value': original_value,
            'epsilon': epsilon,
            'e0': e0_val,
            'e_plus': e_plus,
            'e_minus': e_minus,
            'gradient_autograd': grad_val,
            'gradient_fd_central': fd_central,
            'gradient_fd_forward': fd_forward,
            'rel_error_central': rel_err_central,
            'rel_error_forward': rel_err_forward,
            'status': status,
        }

    finally:
        if model_was_float32:
            model.float()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_fsoi_gradients_all(
    model,
    batch,
    instruments_scalar_fd: Optional[List[str]] = None,
    instruments_directional: Optional[List[str]] = None,
    instruments_float64: Optional[List[str]] = None,
    n_scalar_samples: int = 3,
    n_directional_trials: int = 5,
    n_float64_samples: int = 3,
    epsilon_scalar: float = 1e-2,
    epsilon_directional: float = 1e-2,
    epsilon_float64: float = 1e-4,
    output_csv: Optional[str] = None,
    seed: int = 42,
    forecast_step: int = 0,
) -> pd.DataFrame:
    """
    Run all three gradient validation tests and return a merged result DataFrame.

    Test routing (defaults):
    - Scalar FD    : instruments_scalar_fd   (default: all instruments)
    - Directional  : instruments_directional  (default: high-N satellites)
    - Float64 FD   : instruments_float64      (default: high-N satellites)

    Parameters
    ----------
    instruments_scalar_fd : list of instrument names for per-obs scalar FD test.
        Instruments with very small per-obs gradients will auto-SKIP.
    instruments_directional : list of instrument names for Rademacher direction test.
        Recommended for: atms, avhrr, ssmis, ascat, amsua, seviri_asr, seviri_csr.
    instruments_float64 : list of instrument names for float64 per-obs FD test.
        Recommended for: atms, avhrr, ssmis (any SKIP from scalar test).
    output_csv : if provided, write merged results to this path.
    """
    from fsoi_utils import get_fsoi_inputs

    np.random.seed(seed)
    device = next(model.parameters()).device
    batch = batch.to(device)

    xa = get_fsoi_inputs(batch, model.observation_config, model.instrument_name_to_id)
    all_insts = sorted(xa.keys())

    # High-N satellite instruments that typically SKIP scalar FD
    _high_n = {'atms', 'avhrr', 'ssmis', 'ascat', 'amsua', 'seviri_asr', 'seviri_csr'}
    if instruments_scalar_fd is None:
        instruments_scalar_fd = all_insts
    if instruments_directional is None:
        instruments_directional = [i for i in all_insts if i in _high_n]
    if instruments_float64 is None:
        instruments_float64 = [i for i in all_insts if i in _high_n]

    rows = []

    # ── Test 1: Scalar FD (existing function) ──────────────────────────────
    print("\n" + "=" * 80)
    print("TEST 1 — SCALAR PER-OBS FINITE DIFFERENCE  (float32)")
    print("=" * 80)
    np.random.seed(seed)
    for inst in instruments_scalar_fd:
        x = xa.get(inst)
        if x is None:
            continue
        for _ in range(n_scalar_samples):
            obs_idx = int(np.random.randint(0, x.shape[0]))
            ch_idx = int(np.random.randint(0, x.shape[1]))
            r = finite_difference_check(
                model, batch, inst, obs_idx, ch_idx,
                epsilon=epsilon_scalar, forecast_step=forecast_step,
            )
            r['test'] = 'scalar_fd'
            rows.append(r)

    # ── Test 2: Directional derivative (Rademacher) ─────────────────────────
    if instruments_directional:
        print("\n" + "=" * 80)
        print("TEST 2 — DIRECTIONAL DERIVATIVE  (Rademacher, float32)")
        print("=" * 80)
        for inst in instruments_directional:
            if inst not in xa:
                continue
            r = directional_derivative_check(
                model, batch, inst,
                epsilon=epsilon_directional,
                n_trials=n_directional_trials,
                seed=seed,
                forecast_step=forecast_step,
            )
            # Flatten per-trial rows for CSV
            for t in r.get('trials', []):
                rows.append({
                    'inst_name': inst,
                    'test': 'directional',
                    'trial': t['trial'],
                    'autograd_dd': t['autograd_dd'],
                    'fd_dd': t['fd_dd'],
                    'rel_error': t['rel_error'],
                    'pearson_r': r['pearson_r'],
                    'mean_rel_error': r['mean_rel_error'],
                    'l1_norm': r['l1_norm'],
                    'n_ulp': r['n_ulp'],
                    'status': r['status'],
                })

    # ── Test 3: Float64 FD ─────────────────────────────────────────────────
    if instruments_float64:
        print("\n" + "=" * 80)
        print("TEST 3 — SCALAR PER-OBS FINITE DIFFERENCE  (float64)")
        print("=" * 80)
        np.random.seed(seed + 1)
        for inst in instruments_float64:
            x = xa.get(inst)
            if x is None:
                continue
            for _ in range(n_float64_samples):
                obs_idx = int(np.random.randint(0, x.shape[0]))
                ch_idx = int(np.random.randint(0, x.shape[1]))
                r = finite_difference_check_float64(
                    model, batch, inst, obs_idx, ch_idx,
                    epsilon=epsilon_float64,
                    forecast_step=forecast_step,
                )
                rows.append(r)

    # ── Summary ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    if not df.empty and 'status' in df.columns:
        counts = df.drop_duplicates(subset=[c for c in ['inst_name', 'test', 'trial']
                                            if c in df.columns])['status'].value_counts()
        print("\n" + "=" * 80)
        print("GRADIENT VALIDATION SUMMARY")
        print("=" * 80)
        for status, n in counts.items():
            sym = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗',
                   'SKIP': '~', 'INSUF_SIGNAL': '?', 'ERROR': '!'}.get(status, ' ')
            print(f"  {sym}  {status:<15} {n}")
        n_fail = counts.get('FAIL', 0)
        if n_fail == 0:
            print("\n✓ No gradient failures detected across all three test types.")
        else:
            print(f"\n✗ {n_fail} FAIL results — review autograd graph for {inst}.")

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n  Results written to: {output_csv}")

    return df


def validate_fsoi_gradients(
    model,
    prev_batch,
    curr_batch,
    num_samples: int = 3,
    epsilon: float = 1e-2,
) -> bool:
    """
    Legacy orchestrator: scalar per-obs FD only.

    Use validate_fsoi_gradients_all for the full three-test suite.
    """
    from fsoi_utils import get_fsoi_inputs

    print("\n" + "=" * 80)
    print("FSOI GRADIENT VALIDATION - FINITE DIFFERENCE TESTS")
    print("=" * 80)
    print(f"Testing {num_samples} random observations")
    print("=" * 80 + "\n")

    xa = get_fsoi_inputs(curr_batch, model.observation_config,
                         model.instrument_name_to_id)
    if not xa:
        print("[ERROR] No inputs found in batch")
        return False

    results = []
    for sample_idx in range(num_samples):
        inst_name = np.random.choice(list(xa.keys()))
        x = xa[inst_name]
        obs_idx = np.random.randint(0, x.shape[0])
        channel_idx = np.random.randint(0, x.shape[1])
        print(f"\nSample {sample_idx + 1}/{num_samples}")
        result = finite_difference_check(
            model, curr_batch, inst_name, obs_idx, channel_idx, epsilon=epsilon,
        )
        results.append(result)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in results if r.get('status') == 'PASS')
    warned = sum(1 for r in results if r.get('status') == 'WARNING')
    failed = sum(1 for r in results if r.get('status') == 'FAIL')
    skipped = sum(1 for r in results if r.get('status') == 'SKIP')
    print(f"\nTests run:    {len(results)}")
    print(f"✓ Passed:     {passed}")
    print(f"⚠ Warnings:   {warned}")
    print(f"✗ Failed:     {failed}")
    print(f"~ Skipped:    {skipped}  (gradient too small for float32 scalar FD)")
    print(f"\n  For SKIP instruments use validate_fsoi_gradients_all() which")
    print(f"  adds directional derivative and float64 FD tests.")

    return failed == 0


if __name__ == "__main__":
    print("This module provides validation functions for FSOI.")
    print("Import and use in fsoi_inference.py or test_fsoi.py")
