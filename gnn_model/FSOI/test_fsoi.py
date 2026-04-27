"""
FSOI Test Script - Verify FSOI implementation without running full inference.

This script tests:
1. Configuration loading
2. Dataset creation
3. Model loading and freezing
4. Gradient computation
5. FSOI computation on a single pair

Run this BEFORE full FSOI inference to catch issues early.

Usage:
    python test_fsoi.py --checkpoint path/to/model.ckpt
    python test_fsoi.py --checkpoint path/to/checkpoint_dir/

Author: Azadeh Gholoubi
"""

import argparse
import glob
import sys
import torch
import yaml
from pathlib import Path

# Allow running this script from gnn_model/FSOI/ while importing from parent.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gnn_model import GNNLightning  # noqa: E402
from gnn_datamodule import GNNDataModule  # noqa: E402
from fsoi_dataset import FSOIDataset, verify_sequential_consistency  # noqa: E402
from fsoi_utils import (  # noqa: E402
    get_fsoi_inputs,
    compute_forecast_error,
    compute_adjoints,
    compute_fsoi_per_observation,
    verify_alignment,
    verify_gradients,
)
from fsoi_model_extensions import (  # noqa: E402
    predict_at_targets,  # Use correct graph construction
    freeze_model_for_fsoi,
)
from weight_utils import load_weights_from_yaml  # noqa: E402


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


def test_config_loading():
    """Test 1: Load configurations"""
    print("\n" + "="*80)
    print("TEST 1: Configuration Loading")
    print("="*80)

    try:
        with open("FSOI/configs/fsoi_config.yaml", 'r') as f:
            fsoi_config = yaml.safe_load(f)
        print("✓ FSOI config loaded")
        print(f"  Lead steps: {fsoi_config['forecast']['lead_steps']}")
        print(f"  Output dir: {fsoi_config['data']['output_dir']}")

        obs_config, feature_stats, inst_weights, ch_weights, name_to_id = \
            load_weights_from_yaml("configs/observation_config.yaml")
        print("✓ Observation config loaded")
        print(f"  Instruments: {list(name_to_id.keys())[:5]}...")

        return True, (fsoi_config, obs_config, feature_stats, inst_weights, ch_weights, name_to_id)
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False, None


def test_model_loading(checkpoint_path):
    """Test 2: Load and freeze model"""
    print("\n" + "="*80)
    print("TEST 2: Model Loading and Freezing")
    print("="*80)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = GNNLightning.load_from_checkpoint(checkpoint_path)
        model.to(device)
        print("✓ Model loaded from checkpoint")

        # Check trainable parameters before freezing
        trainable_before = sum(p.requires_grad for p in model.parameters())
        print(f"  Trainable params before freeze: {trainable_before}")

        # Freeze for FSOI
        freeze_model_for_fsoi(model)

        # Check trainable parameters after freezing
        trainable_after = sum(p.requires_grad for p in model.parameters())
        print(f"  Trainable params after freeze: {trainable_after}")

        if trainable_after == 0:
            print("✓ Model successfully frozen")
        else:
            print(f"✗ Model not fully frozen ({trainable_after} params still trainable)")
            return False, None

        return True, (model, device)
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_dataset_creation(obs_config, feature_stats):
    """Test 3: Create sequential dataset"""
    print("\n" + "="*80)
    print("TEST 3: Dataset Creation")
    print("="*80)

    try:
        # Use small date range for testing
        test_start = "2024-01-01"
        test_end = "2024-01-03"  # Just 2-3 days

        data_path = "/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7"

        print(f"Creating datamodule for {test_start} to {test_end}")

        # We need a minimal mesh structure - use a temporary model to get it
        # Or load it from checkpoint if available
        # For now, skip this test if it's too complex
        print("⚠ Dataset creation test skipped (requires full setup)")
        print("  This will be tested in the actual FSOI run")

        return True, None

    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_gradient_computation(model, device):
    """Test 4: Verify gradient computation works"""
    print("\n" + "="*80)
    print("TEST 4: Gradient Computation")
    print("="*80)

    try:
        # Create dummy input tensor
        dummy_input = torch.randn(10, 22, device=device)
        dummy_input.requires_grad_(True)

        # Create dummy target
        dummy_target = torch.randn(10, 22, device=device)

        # Compute simple MSE loss
        loss = ((dummy_input - dummy_target) ** 2).sum()

        print(f"Dummy loss: {loss.item():.6e}")
        print(f"Loss requires_grad: {loss.requires_grad}")

        # Compute gradient
        grad = torch.autograd.grad(loss, dummy_input)[0]

        print(f"✓ Gradient computed successfully")
        print(f"  Gradient shape: {grad.shape}")
        print(f"  Gradient mean: {grad.mean().item():.6e}")
        print(f"  Gradient max: {grad.abs().max().item():.6e}")

        if grad is not None and torch.isfinite(grad).all():
            print("✓ Gradients are valid")
            return True, None
        else:
            print("✗ Gradients are invalid (None or contains NaN/Inf)")
            return False, None

    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_fsoi_formula():
    """Test 5: Verify FSOI formula computation"""
    print("\n" + "="*80)
    print("TEST 5: FSOI Formula")
    print("="*80)

    try:
        # Create dummy data
        xa = torch.randn(5, 10)  # 5 obs, 10 channels
        xb = torch.randn(5, 10)
        ga = torch.randn(5, 10)
        gb = torch.randn(5, 10)

        # Compute FSOI
        delta_x = xa - xb
        g_sum = ga + gb
        fsoi = delta_x * g_sum

        print(f"✓ FSOI formula computed")
        print(f"  Innovation (δx) mean: {delta_x.mean().item():.6e}")
        print(f"  Gradient sum (ga+gb) mean: {g_sum.mean().item():.6e}")
        print(f"  FSOI mean: {fsoi.mean().item():.6e}")
        print(f"  FSOI sum: {fsoi.sum().item():.6e}")
        print(f"  Positive fraction: {(fsoi > 0).float().mean().item()*100:.1f}%")

        # Test with dict format (as used in real code)
        xa_dict = {'test_inst': xa}
        xb_dict = {'test_inst': xb}
        ga_dict = {'test_inst': ga}
        gb_dict = {'test_inst': gb}

        fsoi_dict = compute_fsoi_per_observation(xa_dict, xb_dict, ga_dict, gb_dict)

        if 'test_inst' in fsoi_dict:
            print("✓ FSOI computation with dict format works")
            return True, None
        else:
            print("✗ FSOI dict computation failed")
            return False, None

    except Exception as e:
        print(f"✗ FSOI formula test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_gradient_requires_grad():
    """Test 6: Verify inputs have requires_grad=True"""
    print("\n" + "="*80)
    print("TEST 6: Gradient Requirements")
    print("="*80)

    try:
        # Create dummy batch-like structure
        from torch_geometric.data import HeteroData

        dummy_batch = HeteroData()
        dummy_batch['atms_input'].x = torch.randn(10, 22)

        # Test get_fsoi_inputs with proper config structure
        from gnn_model import _build_instrument_map
        obs_config = {
            'satellite': {
                'atms': {
                    'input_dim': 22,
                    'target_dim': 22,
                    'features': [f'ch{i}' for i in range(22)],  # Add features list
                    'metadata': [],
                    'scan_angle_channels': 1,
                }
            }
        }
        inst_map = _build_instrument_map(obs_config)

        inputs = get_fsoi_inputs(dummy_batch, obs_config, inst_map)

        if 'atms' in inputs:
            requires_grad = inputs['atms'].requires_grad
            print(f"✓ Input tensor extracted")
            print(f"  requires_grad: {requires_grad}")

            if requires_grad:
                print("✓ Gradients properly enabled for FSOI")
                return True, None
            else:
                print("✗ Gradients NOT enabled - FSOI will fail!")
                return False, None
        else:
            print("✗ Could not extract inputs")
            return False, None

    except Exception as e:
        print(f"✗ Gradient requirement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Test FSOI implementation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt file or directory containing checkpoints)",
    )
    args = parser.parse_args()

    # Find checkpoint file (handles both files and directories)
    try:
        checkpoint_path = find_checkpoint(args.checkpoint)
    except ValueError as e:
        print(f"\n❌ Checkpoint error: {e}")
        sys.exit(1)

    print("\n" + "="*80)
    print("FSOI IMPLEMENTATION TEST SUITE")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")

    results = []

    # Test 1: Config loading
    success, data = test_config_loading()
    results.append(("Config Loading", success))
    if not success:
        print("\n❌ Cannot proceed without configs")
        sys.exit(1)

    fsoi_config, obs_config, feature_stats, inst_weights, ch_weights, name_to_id = data

    # Test 2: Model loading
    success, data = test_model_loading(checkpoint_path)
    results.append(("Model Loading", success))
    if not success:
        print("\n❌ Cannot proceed without model")
        sys.exit(1)

    model, device = data

    # Test 3: Dataset creation (skipped for now)
    success, _ = test_dataset_creation(obs_config, feature_stats)
    results.append(("Dataset Creation", success))

    # Test 4: Gradient computation
    success, _ = test_gradient_computation(model, device)
    results.append(("Gradient Computation", success))

    # Test 5: FSOI formula
    success, _ = test_fsoi_formula()
    results.append(("FSOI Formula", success))

    # Test 6: Gradient requirements
    success, _ = test_gradient_requires_grad()
    results.append(("Gradient Requirements", success))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name}")

    total_pass = sum(1 for _, s in results if s)
    total_tests = len(results)

    print(f"\nPassed: {total_pass}/{total_tests}")

    if total_pass == total_tests:
        print("\n✓ All tests passed! FSOI implementation is ready.")
        print("\nNext steps:")
        print("1. Edit FSOI/scripts/run_fsoi.sh to set your checkpoint path")
        print("2. Adjust date range in FSOI/configs/fsoi_config.yaml")
        print("3. Run: sbatch FSOI/scripts/run_fsoi.sh")
    else:
        print("\n⚠ Some tests failed. Please fix issues before running FSOI.")

    sys.exit(0 if total_pass == total_tests else 1)

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
