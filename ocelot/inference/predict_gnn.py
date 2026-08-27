"""OCELOT prediction and inference entry point.

Loads a trained checkpoint and runs obs-space evaluation or mesh-grid inference
over a requested date range.
"""

import argparse
import os
import sys
import time
import yaml
import pandas as pd
import socket
import inspect
from datetime import timedelta

sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import DDPStrategy

import ocelot
from ocelot.configs.inference_config import InferenceConfig
from ocelot.configs.model_config import ModelConfig
from ocelot.gnn_datamodule import GNNDataModule
from ocelot.model.ocelot import Ocelot
from ocelot.weight_utils import load_weights_from_yaml


torch.set_float32_matmul_precision("medium")


def main():
    print(f"Hostname: {socket.gethostname()}")
    print(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    parser = argparse.ArgumentParser(description="GNN Prediction Script")

    parser.add_argument(
        "--model_config",
        "--model-config",
        default="configs/model_config.yaml",
        help="Path to the model architecture YAML.",
    )
    parser.add_argument(
        "--inference_config",
        "--inference-config",
        default="configs/inference_config.yaml",
        help="Path to the inference YAML.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging in addition to the training config setting.",
    )

    args = parser.parse_args()

    model_config = ModelConfig(args.model_config)
    inference_config = InferenceConfig(args.inference_config)

    has_cuda = torch.cuda.is_available()
    if not has_cuda and (inference_config.resources.devices != 1 or inference_config.resources.num_nodes != 1):
        print(
            "CUDA not available; overriding distributed args: "
            f"--devices {inference_config.resources.devices} -> 1, --num_nodes {inference_config.resources.num_nodes} -> 1"
        )
        inference_config.resources.devices = 1
        inference_config.resources.num_nodes = 1

    # --- HYPERPARAMETERS (loaded from checkpoint) ---
    # These will be loaded from the checkpoint but can be overridden if needed:
    # - data_window_hours: Total window size in hours (from checkpoint)
    # - latent_step_hours: Size of each latent step (from checkpoint)
    # Note: These are automatically extracted from the model checkpoint

    print("\n" + "="*80)
    print("GNN PREDICTION MODE")
    print("="*80)
    print(f"Checkpoint: {inference_config.checkpoint}")
    print(f"Date range: {inference_config.data.start_date} to {inference_config.data.end_date}")
    print(f"Output: {inference_config.output_dir}")
    print(f"Devices: {inference_config.resources.devices}, Nodes: {inference_config.resources.num_nodes}")
    print(f"Mode: {'Evaluation' if inference_config.eval_mode else 'Inference'}")
    if args.eval_mode:
        print("  → Evaluation mode: Expects target observations for comparison")
    else:
        print("  → Inference mode: No targets required (operational forecasting)")
    print("="*80 + "\n")

    if not os.path.exists(inference_config.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {inference_config.checkpoint}")

    os.makedirs(inference_config.output_dir, exist_ok=True)
    start_time = time.time()

    # Load configuration
    cfg_path = "configs/observation_config.yaml"
    observation_config, feature_stats, instrument_weights, channel_weights, name_to_id = load_weights_from_yaml(cfg_path)

    with open(cfg_path, "r") as f:
        _raw_cfg = yaml.safe_load(f)

    with open('configs/mesh_config.yaml', 'r') as f:
        mesh_config = yaml.safe_load(f)

    # Optional runtime override for mesh pressure level without editing repo config.
    _mesh_idx_env = os.environ.get("MESH_PRESSURE_LEVEL_IDX", "").strip()
    if _mesh_idx_env != "":
        try:
            mesh_config = dict(mesh_config or {})
            mesh_config["mesh_pressure_level_idx"] = int(_mesh_idx_env)
            print(f"[MESH CONFIG] Overriding mesh_pressure_level_idx via env: {mesh_config['mesh_pressure_level_idx']}")
        except Exception as exc:
            raise ValueError(f"Invalid MESH_PRESSURE_LEVEL_IDX={_mesh_idx_env!r}: {exc}")

    pipeline_cfg = _raw_cfg.get("pipeline", {})

    # Data path
    if args.data_path is None:
        region = "global"
        if region == "conus":
            data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
        else:
            # Keep prediction defaults aligned with train/val (multi-year merged Zarrs).
            data_path = "/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7"
    else:
        data_path = args.data_path

    print(f"Data path: {data_path}")

    # Load model from checkpoint
    print(f"\nLoading model from checkpoint: {inference_config.checkpoint}")

    model = ocelot.inference.make_module(
        model_config=model_config,  # Replace with actual model config if available
        inference_config=inference_config,  # Replace with actual inference config if available
        observation_config=observation_config,
        verbose=True
    )

    model.eval()
    model.prediction_output_dir = inference_config.output_dir

    # Get model hyperparameters
    latent_step_hours = model.hparams.get('latent_step_hours', 3)
    data_window_hours = model.hparams.get('data_window_hours', 12)

    print(f"\nSetting up data module:")
    print(f"  Window size: {data_window_hours}h")
    print(f"  Latent step hours: {latent_step_hours}h")

    # Create data module
    data_module = GNNDataModule(
        data_path=data_path,
        start_date=inference_config.data.start_date,
        end_date=inference_config.data.end_date,
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=inference_config.data.batch_size,
        num_neighbors=3,
        feature_stats=feature_stats,
        pipeline=pipeline_cfg,
        window_size=f"{data_window_hours}h",
        latent_step_hours=latent_step_hours,
        train_val_split_ratio=1.0,
        train_start=inference_config.data.start_date,
        train_end=inference_config.data.end_date,
        val_start=inference_config.data.start_date,
        val_end=inference_config.data.end_date,
        prediction_mode=True,
        require_targets=inference_config.eval_mode,
    )

    setup_end_time = time.time()
    print(f"Setup time: {(setup_end_time - start_time) / 60:.2f} minutes")

    # Configure trainer
    if inference_config.resources.devices == 1 and inference_config.resources.num_nodes == 1:
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
        print(f"Multi-device mode: Using DDPStrategy with {inference_config.resources.devices} devices")

    trainer_kwargs = {
        "accelerator": "gpu" if has_cuda else "cpu",
        "devices": inference_config.resources.devices,
        "num_nodes": inference_config.resources.num_nodes,
        "strategy": strategy,
        "precision": "16-mixed" if has_cuda else 32,
        "logger": False,
        "enable_progress_bar": True,
        "enable_model_summary": False,
    }

    if inference_config.resources.limit_batches is not None:
        trainer_kwargs["limit_predict_batches"] = inference_config.resources.limit_batches
        print(f"Limiting prediction to {inference_config.resources.limit_batches} batches")

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
    print(f"Output saved to: {inference_config.output_dir}")
    print("="*80 + "\n")

    # Generate summary
    print("\nPrediction summary:")

    obs_dir = os.path.join(inference_config.output_dir, 'pred_csv', 'obs-space')
    mesh_dir = os.path.join(inference_config.output_dir, 'pred_csv', 'mesh-grid')

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
