"""OCELOT training entry point."""

import argparse
import faulthandler
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.append(
    os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import yaml

from ocelot.configs.model_config import ModelConfig
from ocelot.configs.observation_config import ObservationConfig
from ocelot.configs.training_config import TrainingConfig


@dataclass(frozen=True)
class WindowPlan:
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    train_window_days: int
    val_window_days: int
    initial_train_end: datetime
    initial_val_end: datetime


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an OCELOT model from YAML configs.")
    parser.add_argument(
        "--model_config",
        "--model-config",
        default="configs/model_config.yaml",
        help="Path to the model architecture YAML.",
    )
    parser.add_argument(
        "--train_config",
        "--train-config",
        default="configs/training_config.yaml",
        help="Path to the training YAML.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging in addition to the training config setting.",
    )
    return parser.parse_args(argv)


def _load_yaml(path: str) -> dict:
    with open(path) as config_file:
        return yaml.safe_load(config_file) or {}


def _build_window_plan(config: TrainingConfig) -> WindowPlan:
    sampler = config.data.sampler
    date_range = sampler.date_range
    train_start = date_range.train_start_date
    train_end = date_range.train_end_date
    val_start = date_range.val_start_date
    val_end = date_range.val_end_date

    if train_start >= train_end:
        raise ValueError("Training date range must have a positive duration")
    if val_start >= val_end:
        raise ValueError("Validation date range must have a positive duration")
    if val_start < train_end:
        raise ValueError("Training and validation date ranges must not overlap")

    train_window_days = (
        sampler.train_window_days
        if sampler.type == "random"
        else sampler.window_days
    )
    val_window_days = config.validation.mode.window_days
    initial_train_end = min(train_start + timedelta(days=train_window_days), train_end)
    initial_val_end = min(val_start + timedelta(days=val_window_days), val_end)

    return WindowPlan(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        initial_train_end=initial_train_end,
        initial_val_end=initial_val_end,
    )


def _resolve_checkpoint(config: TrainingConfig):
    from ocelot.ckpt_utils import find_latest_checkpoint

    if config.resume_from_latest:
        checkpoint = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint:
            print(f"[INFO] Auto-resuming from: {checkpoint}")
        else:
            print("[INFO] No checkpoint found, starting fresh")
        return checkpoint
    if config.resume_from_checkpoint:
        print(f"[INFO] Resuming from: {config.resume_from_checkpoint}")
        return config.resume_from_checkpoint

    print("[INFO] No checkpoint, starting fresh training")
    return None


def _build_callbacks(config: TrainingConfig, windows: WindowPlan):
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from ocelot.callbacks import (
        ResampleDataCallback,
        SequentialDataCallback,
        ValWindowCallback,
    )

    class DelayedEarlyStopping(EarlyStopping):
        def __init__(self, *args, start_epoch=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_epoch = int(start_epoch)

        def _delay_active(self, trainer):
            return int(trainer.current_epoch) < self.start_epoch

        def on_validation_end(self, trainer, pl_module):
            if not self._delay_active(trainer):
                return super().on_validation_end(trainer, pl_module)

        def on_train_epoch_end(self, trainer, pl_module):
            if not self._delay_active(trainer):
                return super().on_train_epoch_end(trainer, pl_module)

    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="gnn-epoch-{epoch:02d}-val_loss-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
    ]

    early_stopping = config.trainer.early_stopping
    if early_stopping.enabled:
        callbacks.append(
            DelayedEarlyStopping(
                monitor="val_loss",
                patience=early_stopping.patience,
                mode="min",
                min_delta=early_stopping.min_delta,
                start_epoch=early_stopping.start_epoch,
                verbose=True,
                check_finite=True,
                check_on_train_epoch_end=False,
                strict=False,
            )
        )

    sampler = config.data.sampler
    if sampler.type == "random":
        callbacks.append(
            ResampleDataCallback(
                train_start_date=windows.train_start,
                train_end_date=windows.train_end,
                val_start_date=windows.val_start,
                val_end_date=windows.val_end,
                train_window_days=windows.train_window_days,
                val_window_days=windows.val_window_days,
                mode="random",
                resample_val=False,
                seq_stride_days=sampler.sequential_stride_days,
                switch_to_sequential_after_epochs=sampler.switch_to_sequential_after_epochs,
                auto_switch_to_sequential=sampler.auto_switch_to_sequential,
                auto_switch_metric=sampler.auto_switch_metric,
                auto_switch_patience_epochs=sampler.auto_switch_patience_epochs,
                auto_switch_min_delta=sampler.auto_switch_min_delta,
            )
        )
    else:
        callbacks.append(
            SequentialDataCallback(
                full_start_date=windows.train_start,
                full_end_date=windows.train_end,
                window_days=windows.train_window_days,
                stride_days=sampler.stride_days,
            )
        )

    validation_mode = config.validation.mode
    if validation_mode.type != "fixed":
        callbacks.append(
            ValWindowCallback(
                val_start_date=windows.val_start,
                val_end_date=windows.val_end,
                val_window_days=windows.val_window_days,
                mode=validation_mode.type,
                stride_days=getattr(validation_mode, "stride_days", windows.val_window_days),
                update_every_n_epochs=validation_mode.update_every_n_epochs,
            )
        )

    return callbacks


def _build_trainer(config: TrainingConfig, callbacks):
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DDPStrategy

    trainer_config = config.trainer
    trainer_kwargs = {
        "max_epochs": trainer_config.max_epochs,
        "accelerator": trainer_config.accelerator,
        "devices": trainer_config.devices,
        "num_nodes": trainer_config.num_nodes,
        "precision": trainer_config.precision,
        "log_every_n_steps": trainer_config.log_every_n_steps,
        "num_sanity_val_steps": trainer_config.num_sanity_val_steps,
        "gradient_clip_val": trainer_config.gradient_clip_val,
        "enable_progress_bar": False,
        "reload_dataloaders_every_n_epochs": 1,
        "check_val_every_n_epoch": 1,
        "logger": CSVLogger(save_dir="logs", name=config.experiment_name),
        "callbacks": callbacks,
    }
    if trainer_config.devices > 1 or trainer_config.num_nodes > 1:
        use_gpu = trainer_config.accelerator != "cpu" and torch.cuda.is_available()
        trainer_kwargs["strategy"] = DDPStrategy(
            process_group_backend="nccl" if use_gpu else "gloo",
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            timeout=timedelta(hours=1),
        )
    if trainer_config.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = trainer_config.limit_train_batches
    if trainer_config.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = trainer_config.limit_val_batches
    return pl.Trainer(**trainer_kwargs)


def run_training(model_config_path: str, training_config_path: str, verbose=False):
    import lightning.pytorch as pl
    import torch

    from ocelot.gnn_datamodule import GNNDataModule
    from ocelot.logger import LogLevel, log
    from ocelot.model.ocelot_factory import OcelotFactory

    model_config = ModelConfig(model_config_path)
    training_config = TrainingConfig(training_config_path)
    observation_config = ObservationConfig(training_config.observation_config_path)

    verbose = bool(verbose or training_config.verbose)
    if verbose:
        log.set_log_level(LogLevel.Debug)

    seed = 42 if training_config.debug_mode else (training_config.seed or 12345)
    pl.seed_everything(seed, workers=True)

    windows = _build_window_plan(training_config)

    print(f"Training period:   {windows.train_start} -> {windows.train_end}")
    print(f"Validation period: {windows.val_start} -> {windows.val_end}")

    setup_start = time.time()
    module = OcelotFactory.create_training_module(
        model_config=model_config,
        training_config=training_config,
        observation_config=observation_config,
        verbose=verbose,
    )

    resume_path = _resolve_checkpoint(training_config)
    if resume_path and training_config.load_weights_only:
        print(f"[INFO] Loading weights only (strict=False) from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = module.load_state_dict(state, strict=False)
        print(
            "[INFO] Weights-only load complete. "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )

    date_range = training_config.data.sampler.date_range
    split_ratio = getattr(date_range, "train_val_split_ratio", 0.9)
    data_module = GNNDataModule(
        data_path=training_config.data.path,
        start_date=windows.train_start,
        end_date=windows.initial_train_end,
        observation_config=observation_config.observation_config,
        mesh_structure=module.model.mesh.mesh_structure,
        batch_size=training_config.data.batch_size,
        num_neighbors=training_config.data.num_neighbors,
        feature_stats=observation_config.feature_stats,
        verbose=verbose,
        pipeline=observation_config.pipeline,
        window_size=f"{training_config.data.window_hours}h",
        latent_step_hours=training_config.data.latent_step_hours,
        train_val_split_ratio=split_ratio,
        cache_val_windows=training_config.validation.cache_windows,
        val_cache_max_entries=training_config.validation.cache_max_entries,
        prediction_mode=False,
        train_start=windows.train_start,
        train_end=windows.initial_train_end,
        val_start=windows.val_start,
        val_end=windows.initial_val_end,
    )

    trainer = _build_trainer(training_config, _build_callbacks(training_config, windows))
    setup_end = time.time()
    print(f"Initial setup time: {(setup_end - setup_start) / 60:.2f} minutes")

    checkpoint_for_fit = None if training_config.load_weights_only else resume_path
    trainer.fit(module, data_module, ckpt_path=checkpoint_for_fit)

    end_time = time.time()
    print(f"Training time: {(end_time - setup_end) / 60:.2f} minutes")
    print(f"Total time: {(end_time - setup_start) / 60:.2f} minutes")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main(argv=None):
    args = parse_args(argv)
    faulthandler.enable()
    sys.stderr.write("===> ENTERED MAIN\n")
    print(f"Hostname: {socket.gethostname()}")
    print(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID')}")
    print(f"  SLURM_LOCALID: {os.environ.get('SLURM_LOCALID')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    run_training(args.model_config, args.train_config, verbose=args.verbose)


if __name__ == "__main__":
    main()