"""
GNN prediction/inference script.

Loads a trained checkpoint and runs predictions on specified date range.
"""

import argparse
import os
import sys
import time
import yaml
import pandas as pd
import socket
import inspect

import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import DDPStrategy

from gnn_datamodule import GNNDataModule
from gnn_model import GNNLightning
from weight_utils import load_weights_from_yaml
from datetime import timedelta

torch.set_float32_matmul_precision("medium")


def main():
    print(f"Hostname: {socket.gethostname()}")
    print(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    parser = argparse.ArgumentParser(description="GNN Prediction Script")

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--start_date", type=str, required=True,
                        help="Start date for prediction (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True,
                        help="End date for prediction (YYYY-MM-DD)")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Output directory (default: predictions)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to data directory (auto-detect if not provided)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of nodes (default: 1)")
    parser.add_argument("--limit_batches", type=int, default=None,
                        help="Limit number of batches (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    # Evaluation mode
    parser.add_argument("--eval-mode", action="store_true",
                        help="Evaluation mode: expects target observations to exist."
                             "Default: False (inference mode - no targets required)")

    args = parser.parse_args()

    has_cuda = torch.cuda.is_available()
    if not has_cuda and (args.devices != 1 or args.num_nodes != 1):
        print(
            "CUDA not available; overriding distributed args: "
            f"--devices {args.devices} -> 1, --num_nodes {args.num_nodes} -> 1"
        )
        args.devices = 1
        args.num_nodes = 1

    # --- HYPERPARAMETERS (loaded from checkpoint) ---
    # These will be loaded from the checkpoint but can be overridden if needed:
    # - data_window_hours: Total window size in hours (from checkpoint)
    # - latent_step_hours: Size of each latent step (from checkpoint)
    # Note: These are automatically extracted from the model checkpoint

    print("\n" + "="*80)
    print("GNN PREDICTION MODE")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output_dir}")
    print(f"Devices: {args.devices}, Nodes: {args.num_nodes}")
    print(f"Mode: {'Evaluation' if args.eval_mode else 'Inference'}")
    if args.eval_mode:
        print("  → Evaluation mode: Expects target observations for comparison")
    else:
        print("  → Inference mode: No targets required (operational forecasting)")
    print("="*80 + "\n")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # Load configuration
    cfg_path = "configs/observation_config.yaml"
    observation_config, feature_stats, instrument_weights, channel_weights, name_to_id = load_weights_from_yaml(cfg_path)

    with open(cfg_path, "r") as f:
        _raw_cfg = yaml.safe_load(f)

    with open('configs/mesh_config.yaml', 'r') as f:
        mesh_config = yaml.safe_load(f)

    pipeline_cfg = _raw_cfg.get("pipeline", {})

    # Data path
    if args.data_path is None:
        region = "global"
        if region == "conus":
            data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
        else:
            data_path = "/scratch3/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v6/global"
    else:
        data_path = args.data_path

    print(f"Data path: {data_path}")

    # Load model from checkpoint
    print(f"\nLoading model from checkpoint: {args.checkpoint}")

    try:
        model = GNNLightning.load_from_checkpoint(
            args.checkpoint,
            observation_config=observation_config,
            mesh_config=mesh_config,
            feature_stats=feature_stats,
            instrument_weights=instrument_weights,
            channel_weights=channel_weights,
            strict=False,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTrying alternative loading method...")

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        hparams = ckpt.get('hyper_parameters', {})

        # Filter hparams to match the current constructor signature (robust to checkpoint drift)
        try:
            sig = inspect.signature(GNNLightning.__init__)
            allowed = set(sig.parameters.keys())
            allowed.discard('self')
            filtered_hparams = {k: v for k, v in (hparams or {}).items() if k in allowed}
        except Exception:
            filtered_hparams = dict(hparams or {})

        model = GNNLightning(
            observation_config=observation_config,
            mesh_config=mesh_config,
            feature_stats=feature_stats,
            instrument_weights=instrument_weights,
            channel_weights=channel_weights,
            **filtered_hparams
        )

        state = ckpt.get('state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            "Model loaded successfully using alternative method (strict=False)! "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )

    model.eval()
    model.prediction_output_dir = args.output_dir

    # Get model hyperparameters
    latent_step_hours = model.hparams.get('latent_step_hours', 3)
    data_window_hours = model.hparams.get('data_window_hours', 12)

    print(f"\nSetting up data module:")
    print(f"  Window size: {data_window_hours}h")
    print(f"  Latent step hours: {latent_step_hours}h")

    # Create data module
    data_module = GNNDataModule(
        data_path=data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=args.batch_size,
        num_neighbors=3,
        feature_stats=feature_stats,
        pipeline=pipeline_cfg,
        window_size=f"{data_window_hours}h",
        latent_step_hours=latent_step_hours,
        train_val_split_ratio=1.0,
        train_start=args.start_date,
        train_end=args.end_date,
        val_start=args.start_date,
        val_end=args.end_date,
        prediction_mode=True,
        require_targets=args.eval_mode,
    )

    setup_end_time = time.time()
    print(f"Setup time: {(setup_end_time - start_time) / 60:.2f} minutes")

    # Configure trainer
    if args.devices == 1 and args.num_nodes == 1:
        strategy = "auto"
        print("Single device mode: Using strategy='auto'")
    else:
        strategy = DDPStrategy(
            process_group_backend="nccl",
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            timeout=timedelta(hours=1),
        )
        print(f"Multi-device mode: Using DDPStrategy with {args.devices} devices")

    trainer_kwargs = {
        "accelerator": "gpu" if has_cuda else "cpu",
        "devices": args.devices,
        "num_nodes": args.num_nodes,
        "strategy": strategy,
        "precision": "16-mixed" if has_cuda else 32,
        "logger": False,
        "enable_progress_bar": True,
        "enable_model_summary": False,
    }

    if args.limit_batches is not None:
        trainer_kwargs["limit_predict_batches"] = args.limit_batches
        print(f"Limiting prediction to {args.limit_batches} batches")

    trainer = pl.Trainer(**trainer_kwargs)

    # Run prediction
    print("\n" + "="*80)
    print("STARTING PREDICTION")
    print("="*80 + "\n")

    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.current_device()} memory allocated:",
              torch.cuda.memory_allocated() / 1024**3, "GB")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    predictions = trainer.predict(model, datamodule=data_module)

    end_time = time.time()
    print("\n" + "="*80)
    print("PREDICTION COMPLETED")
    print("="*80)
    print(f"Prediction time: {(end_time - setup_end_time) / 60:.2f} minutes")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Output saved to: {args.output_dir}")
    print("="*80 + "\n")

    # Generate summary
    print("\nPrediction summary:")

    obs_dir = os.path.join(args.output_dir, 'pred_csv', 'obs-space')
    mesh_dir = os.path.join(args.output_dir, 'pred_csv', 'mesh-grid')

    if os.path.exists(obs_dir):
        csv_files = [f for f in os.listdir(obs_dir) if f.endswith('.csv')]
        print(f"  Observation predictions (obs-space): {len(csv_files)} files")

        instruments = {}
        for f in csv_files:
            parts = f.split('_')
            if len(parts) >= 2:
                inst = parts[1]
                instruments[inst] = instruments.get(inst, 0) + 1

        print("\n  By instrument:")
        for inst, count in sorted(instruments.items()):
            print(f"    {inst}: {count} files")

    if os.path.exists(mesh_dir):
        mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.csv')]
        print(f"\n  Mesh predictions (target): {len(mesh_files)} files")

        mesh_summary = {}
        for f in mesh_files:
            parts = f.replace('.csv', '').split('_')
            if len(parts) >= 2:
                inst = parts[0]
                # Find the 'f' part
                fhr_parts = [p for p in parts if p.startswith('f')]
                fhr = fhr_parts[0] if fhr_parts else 'unknown'
                key = f"{inst} ({fhr})"
                mesh_summary[key] = mesh_summary.get(key, 0) + 1

        print("\n  By instrument and forecast hour:")
        for key, count in sorted(mesh_summary.items()):
            print(f"    {key}: {count} files")

    print("\nPrediction complete!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
