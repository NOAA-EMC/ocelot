import os, sys
from typing import Dict, Tuple, List, Optional
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

import numpy as np
from matplotlib import pyplot as plt

from ocelot.configs.training_config import TrainingConfig
from ocelot.configs.observation_config import ObservationConfig
from ocelot.logger import log
from ocelot.model.ocelot import Ocelot
from ocelot.training.loss import weighted_mse_loss, weighted_huber_loss


class OcelotTrainingModule(pl.LightningModule):
    def __init__(self, model: Ocelot, training_config: TrainingConfig):
        super().__init__()
        self.model = model
        self.training_config = training_config
        self.obs_config = ObservationConfig(self.training_config.observation_config_path)
        self._printed_first_train_batch = False
        self.save_hyperparameters()


    def forward(self, data: HeteroData):
        return self.model(data)

    
    def on_fit_start(self):
        # Reset one-time debug cache each run.
        self._edge_attr_debug_seen = set()
        if self.training_config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            if self.trainer.is_global_zero:
                print("[ANOMALY] torch.autograd anomaly mode enabled once at fit start.")

    def _edge_key(self, edge_type: Tuple[str, str, str]) -> str:
        """Converts an edge_type tuple to a string key for ModuleDict."""
        return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"


    def training_step(self, batch, batch_idx):
        print("[DIAG] Entered training_step()")
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Step {batch_idx} - Memory allocated: {allocated:.2f} GB")

        # Print first-batch info for window validation
        if self._printed_first_train_batch is False:
            bt = getattr(batch, "input_time", None) or getattr(batch, "time", None)
            print(f"[FirstTrainBatch] batch_idx=0 time={bt}")
            self._printed_first_train_batch = True

        print(f"[training_step] batch: {getattr(batch, 'bin_name', 'N/A')}")

        # ---- Forward pass and loss calculation ----
        all_predictions = self(batch)

        # Extract ground truths based on rollout mode
        ground_truth_data = self.model._extract_ground_truths_and_metadata(batch, all_predictions)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_predictions = 0

        # Calculate loss for each observation type and add it to the total
        for node_type, preds_list in all_predictions.items():
            if node_type not in ground_truth_data:
                continue

            # Get the base instrument name (e.g., "atms" from "atms_target")
            # Add handling for target_step in latent mode
            if "_target_step" in node_type:
                inst_name = node_type.split("_target_step")[0]
            else:
                inst_name = node_type.replace("_target", "")
            inst_id = self.model.instrument_name_to_id.get(inst_name, None)
            instrument_weight = self.model.instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

            gt_data = ground_truth_data[node_type]
            gts_list = gt_data["gts_list"]
            instrument_ids_list = gt_data["instrument_ids_list"]
            valid_mask_list = gt_data["valid_mask_list"]

            for step, (y_pred, y_true, instrument_ids, valid_mask) in enumerate(
                zip(preds_list, gts_list, instrument_ids_list, valid_mask_list)
            ):
                # Skip if either prediction or ground truth is None or empty
                if y_pred is None or y_true is None or y_pred.numel() == 0 or y_true.numel() == 0:
                    continue

                # Ensure finite tensors
                if not torch.isfinite(y_pred).all():
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(y_true).all():
                    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)

                # Skip if mask exists but nothing valid
                if valid_mask is not None and valid_mask.sum() == 0:
                    continue

                # Shape validation before loss calculation
                if y_pred.shape[0] != y_true.shape[0]:
                    print(f"[ERROR] Shape mismatch  {node_type} step {step}:")
                    print(f"  y_pred: {y_pred.shape} ({y_pred.shape[0]} obs)")
                    print(f"  y_true: {y_true.shape} ({y_true.shape[0]} obs)")
                    print(f"  Skipping this prediction to avoid crash")
                    continue

                channel_loss = self._compute_channel_loss(y_pred, y_true, instrument_ids, valid_mask)

                if not torch.isfinite(channel_loss):
                    if self.trainer.is_global_zero:
                        print(f"[WARN] Non-finite channel_loss for {node_type} at step {step}; skipping this term.")
                    continue

                # Apply the overall instrument weight
                weighted_loss = channel_loss * instrument_weight

                # Add the loss for this instrument to the total
                total_loss = total_loss + weighted_loss
                num_predictions += 1

        dummy_loss = 0.0
        for param in self.parameters():
            dummy_loss += param.sum() * 0.0
        # Average the loss over all observation types that had predictions
        avg_loss = total_loss / num_predictions if num_predictions > 0 else torch.tensor(0.0, device=self.device)
        avg_loss = avg_loss + dummy_loss

        # Log rollout steps appropriately
        step_info = self.model._get_latent_step_info(batch)
        latent_rollout_steps = step_info["num_steps"]
        if self.training_config.verbose:
            print(f"[DEBUG] latent rollout steps: {latent_rollout_steps}")

        self.log(
            "train_loss",
            avg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        self.log("rollout_steps", float(latent_rollout_steps), on_step=True, sync_dist=False)
        if self.trainer.is_global_zero and batch_idx == 0:
            print(f"[TRAIN] Epoch {self.current_epoch} - train_loss: {avg_loss.cpu().item():.6f}")

        return avg_loss


    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        rank = int(os.environ.get("RANK", "0"))

        # One concise banner (only once on global zero)
        if getattr(self.trainer, "is_global_zero", True):
            print(f"=== Starting Epoch {self.current_epoch} ===")

        print(f"[Rank {rank}] === TRAIN EPOCH {self.current_epoch} START ===")

        dm = self.trainer.datamodule
        train_start = getattr(dm.hparams, "train_start", None)
        train_end = getattr(dm.hparams, "train_end", None)
        sum_id = id(getattr(dm, "train_data_summary", None))
        print(f"[TrainWindow] {train_start} .. {train_end} (sum_id={sum_id})")

        # reset first-batch flag for this epoch
        self._printed_first_train_batch = False

        # learning rate tracking
        opts = self.optimizers()
        opt = opts[0] if isinstance(opts, (list, tuple)) else opts
        current_lr = opt.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=False, on_epoch=True, on_step=False)


    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        rank = int(os.environ.get("RANK", "0"))
        print(f"\n[Rank {rank}] === VAL EPOCH {self.current_epoch} START ===")
        dm = self.trainer.datamodule
        print(f"[ValWindow]   {getattr(dm.hparams, 'val_start', None)} .. {getattr(dm.hparams, 'val_end', None)} "
              f"(sum_id={id(getattr(dm, 'val_data_summary', None))})")
        self._printed_first_val_batch = False


    def validation_step(self, batch, batch_idx):
        log.info(f"VALIDATION STEP batch: {batch.bin_name}")

        # Build decoder names from config (all possible node_types with targets)
        decoder_names = [f"{inst_name}_target" for obs_type, instruments in self.obs_config.observation_config.items() for inst_name in instruments]

        # Prepare metrics storage
        all_step_rmse = {name: [] for name in decoder_names}
        all_step_mae = {name: [] for name in decoder_names}
        all_step_bias = {name: [] for name in decoder_names}
        all_losses = []

        # Determine rollout steps based on mode
        step_info = self.model._get_latent_step_info(batch)
        latent_rollout_steps = step_info["num_steps"]
        log.info(f"[validation_step] latent rollout steps: {latent_rollout_steps}")

        # Forward pass: Dict[node_type, List[Tensor]] per step
        # all_predictions = self(batch)
        # if isinstance(all_predictions, tuple):
        #     all_predictions, _ = all_predictions

        # Forward pass: Dict[node_type, List[Tensor]] per step
        result = self(batch)

        # Check if the result is a tuple (only happens in validation mode now)
        if isinstance(result, tuple):
            all_predictions, mesh_features_per_step = result
        else:
            # This path shouldn't be hit in validation_step, but is a good safeguard.
            all_predictions = result
            mesh_features_per_step = []  # Initialize empty list if somehow missed

        # Extract ground truths based on rollout mode
        ground_truth_data = self.model._extract_ground_truths_and_metadata(batch, all_predictions)

        total_loss = torch.tensor(0.0, device=self.device)
        num_predictions = 0

        # --- Loop over all node_types/decoders ---
        for node_type, preds_list in all_predictions.items():
            log.info(f"[validation_step] Processing node_type: {node_type}")
            if node_type not in ground_truth_data:
                continue

            feats = None
            # Latent mode: target_step0, target_step1, etc
            if "_target_step" in node_type:
                inst_name = node_type.split("_target_step")[0]
            else:
                inst_name = node_type.replace("_target", "")
            inst_id = self.model.instrument_name_to_id.get(inst_name, None)
            instrument_weight = self.model.instrument_weights.get(inst_id, 1.0) if inst_id is not None else 1.0

            gt_data = ground_truth_data[node_type]
            gts_list = gt_data["gts_list"]
            instrument_ids_list = gt_data["instrument_ids_list"]
            valid_mask_list = gt_data["valid_mask_list"]

            n_steps = min(len(preds_list), len(gts_list))

            for step, (y_pred, y_true, instrument_ids, valid_mask) in enumerate(
                zip(preds_list, gts_list, instrument_ids_list, valid_mask_list)
            ):
                log.info(f"[validation_step] {node_type} - step {step+1}/{n_steps}")
                # Skip if either prediction or ground truth is None or empty
                if y_pred is None or y_true is None or y_pred.numel() == 0 or y_true.numel() == 0:
                    continue
                if y_pred.shape != y_true.shape:
                    continue

                if not torch.isfinite(y_pred).all():
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(y_true).all():
                    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)

                if valid_mask is not None:
                    valid_mask = valid_mask.to(dtype=torch.bool, device=y_pred.device)
                    if valid_mask.numel() == 0 or valid_mask.sum() == 0:
                        continue  # nothing valid for this node_type/step

                # Get the channel-weighted loss
                channel_loss = self._compute_channel_loss(
                    y_pred,
                    y_true,
                    instrument_ids,
                    valid_mask,
                )

                if not torch.isfinite(channel_loss):
                    if self.trainer.is_global_zero:
                        log.info(f"[WARN] Non-finite channel_loss for {node_type} at step {step}; skipping this term.")
                    continue

                # Apply the overall instrument weight
                weighted_loss = channel_loss * instrument_weight

                total_loss = total_loss + weighted_loss
                num_predictions += 1
                self.log(
                    f"val_loss_{node_type}",
                    weighted_loss.detach(),
                    sync_dist=False,
                    on_epoch=True,
                    batch_size=1,
                    prog_bar=False,
                    logger=True,
                    rank_zero_only=True,
                )

                # --- Metrics Calculation ---
                y_pred_unnorm = self.model.unnormalize_standardscaler(y_pred, node_type)
                y_true_unnorm = self.model.unnormalize_standardscaler(y_true, node_type)

                if valid_mask is not None:
                    # reduce only over valid elements
                    vm = valid_mask
                    # RMSE
                    mse_elems = (y_pred_unnorm - y_true_unnorm).pow(2)
                    rmse = torch.sqrt((mse_elems[vm]).mean() + 1e-12)
                    # MAE
                    mae = (y_pred_unnorm - y_true_unnorm).abs()
                    mae = (mae[vm]).mean()
                    # Bias
                    bias = y_pred_unnorm - y_true_unnorm
                    bias = (bias[vm]).mean()

                    # Keep per-channel vectors to match the logging format
                    # (compute channelwise means with masking)
                    # shape handling:
                    vm_f = vm.float()
                    denom_ch = vm_f.sum(dim=0).clamp_min(1.0)
                    rmse_ch = torch.sqrt((mse_elems * vm_f).sum(dim=0) / denom_ch + 1e-12)
                    mae_ch = (mae := ((y_pred_unnorm - y_true_unnorm).abs() * vm_f).sum(dim=0) / denom_ch)
                    bias_ch = ((y_pred_unnorm - y_true_unnorm) * vm_f).sum(dim=0) / denom_ch

                    step_rmse = rmse_ch
                    step_mae = mae_ch
                    step_bias = bias_ch
                else:
                    step_rmse = torch.sqrt(F.mse_loss(y_pred_unnorm, y_true_unnorm, reduction="none")).mean(dim=0)
                    step_mae = F.l1_loss(y_pred_unnorm, y_true_unnorm, reduction="none").mean(dim=0)
                    step_bias = (y_pred_unnorm - y_true_unnorm).mean(dim=0)

                all_step_rmse[node_type].append(step_rmse)
                all_step_mae[node_type].append(step_mae)
                all_step_bias[node_type].append(step_bias)

                if (
                    self.trainer.is_global_zero  # only main process
                    and step == 0  # only concatenate latent rollout once
                    and self.training_config.validation.csv.enabled
                    and batch_idx < max(1, self.training_config.validation.csv.num_batches)
                    and (self.current_epoch % max(1, self.training_config.validation.csv.every_n_epochs) == 0)
                ):
                    # --- CSV save block ---
                    out_dir = self.training_config.validation.csv.out_dir
                    os.makedirs(out_dir, exist_ok=True)

                    # LATENT ROLLOUT: Concatenate all steps into standard format
                    self.model._save_latent_concatenated_csv(
                        batch, node_type, preds_list, gts_list,
                        valid_mask_list, out_dir, batch_idx
                    )

            # Placeholder logging for missing steps (to ensure stable CSV shape for loggers)
            num_channels = all_step_rmse[node_type][0].shape[0] if all_step_rmse[node_type] else 1
            for step in range(n_steps, self.training_config.data.max_rollout_steps):
                placeholder_metric = torch.tensor(float("nan"), device=self.device)

        # --- Average metrics across steps for each node_type ---
        for node_type in decoder_names:
            if all_step_rmse[node_type]:
                avg_rmse = torch.stack(all_step_rmse[node_type]).mean(dim=0)
                avg_mae = torch.stack(all_step_mae[node_type]).mean(dim=0)
                avg_bias = torch.stack(all_step_bias[node_type]).mean(dim=0)

        if self.trainer.is_global_zero and batch_idx == 0:
            for node_type in decoder_names:
                if all_step_rmse[node_type]:
                    print(f"[VAL] {node_type} RMSE (avg): {torch.stack(all_step_rmse[node_type]).mean().item():.4f}")

        if self.training_config.verbose and self.trainer.is_global_zero and batch_idx == 0:
            for node_type in decoder_names:
                if node_type not in all_predictions or not all_predictions[node_type]:
                    continue
                y_pred = all_predictions[node_type][0]
                y_true = ground_truth_data[node_type]["gts_list"][0]
                y_pred_unnorm = self.model.unnormalize_standardscaler(y_pred, node_type)
                y_true_unnorm = self.model.unnormalize_standardscaler(y_true, node_type)

                n_channels = y_pred_unnorm.shape[1]
                for i in range(min(5, n_channels)):
                    try:
                        plt.figure()
                        # Get data and remove any NaN/inf values
                        y_true_data = y_true_unnorm[:, i].cpu().numpy()
                        y_pred_data = y_pred_unnorm[:, i].cpu().numpy()

                        # Filter out non-finite values
                        y_true_finite = y_true_data[np.isfinite(y_true_data)]
                        y_pred_finite = y_pred_data[np.isfinite(y_pred_data)]

                        # Skip if no finite data
                        if len(y_true_finite) == 0 or len(y_pred_finite) == 0:
                            plt.close()
                            continue

                        # Use auto bins or limit to reasonable number
                        n_bins = min(50, max(10, len(y_true_finite) // 20))

                        plt.hist(
                            y_true_finite,
                            bins=n_bins,
                            alpha=0.6,
                            color="blue",
                            label="y_true",
                        )
                        plt.hist(
                            y_pred_finite,
                            bins=n_bins,
                            alpha=0.6,
                            color="orange",
                            label="y_pred",
                        )
                        plt.xlabel(f"{node_type} - Channel {i+1}")
                        plt.ylabel("Frequency")
                        plt.title(f"Histogram - {node_type} Channel {i+1} (Epoch {self.current_epoch})")
                        plt.legend()
                        plt.tight_layout()
                    except Exception as e:
                        print(f"Warning: Could not create histogram for {node_type} channel {i+1}: {e}")
                        plt.close()
                        continue
                    plt.savefig(f"hist_{node_type}_ch_{i+1}_epoch{self.current_epoch}.png")
                    plt.close()

        # --- Final loss calculation for the entire validation step ---
        avg_loss = total_loss / num_predictions if num_predictions > 0 else torch.tensor(0.0, device=self.device)

        self.log(
            "val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        if self.trainer.is_global_zero:
            print(f"--- Epoch {self.current_epoch} Validation ---")
            print(f"val_loss: {avg_loss.item():.6f}")

        # Save mesh features from first batch for epoch-end processing
        if batch_idx == 0 and self.obs_config.mesh_config.enable_mesh_pred:
            self._last_val_mesh_features = mesh_features_per_step
            self._last_val_batch = batch

        return avg_loss


    def on_validation_epoch_end(self):
        """Generate mesh predictions at END of validation epoch."""
        if not self.obs_config.mesh_config.enable_mesh_pred or not self.trainer.is_global_zero:
            return

        # Check if we saved mesh features during validation
        if not hasattr(self, '_last_val_mesh_features') or self._last_val_mesh_features is None:
            print("[MESH PRED] No mesh features from validation, skipping")
            return

        try:
            with torch.no_grad():
                temp_mesh_pred_edges = self.model._get_mesh_pred_edges()
                init_time_unix = self.model._extract_init_time_unix(self._last_val_batch)
                mesh_predictions = self.model._decode_all_steps_to_mesh(
                    self._last_val_mesh_features,
                    temp_mesh_pred_edges,
                    init_time_unix
                )
                if mesh_predictions:
                    self._save_mesh_predictions(
                        mesh_predictions,
                        temp_mesh_pred_edges,
                        batch_idx=0,
                        epoch=self.current_epoch,
                        mode='val',
                        batch=self._last_val_batch,
                        output_dir='val_mesh_csv'
                    )
        except Exception as e:
            print(f"[MESH PRED] Failed (non-critical): {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self._last_val_mesh_features = None
            self._last_val_batch = None


    def configure_optimizers(self):
        opt_config = self.training_config.optimizer

        if opt_config.type != "adamw":
            raise ValueError(f"Unsupported optimizer type: {opt_config.type}")

        optimizer = torch.optim.AdamW(self.parameters(), lr=opt_config.lr, weight_decay=opt_config.weight_decay)

        # TenYearTrain-style schedule: warmup + cosine decay (robust to noisy validation)
        if opt_config.lr_schedule == "cosine_warmup":
            max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 328
            warmup_epochs = max(1, int(opt_config.warmup_pct * max_epochs))
            warmup_epochs = min(warmup_epochs, max(1, max_epochs - 1))

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=opt_config.min_lr,
            )

            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=opt_config.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            combined_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs],
            )

            if self.trainer.is_global_zero:
                print("[LR Schedule] Cosine decay with warmup")
                print(f"  Warmup epochs: {warmup_epochs} ({opt_config.warmup_start_factor}×lr → 1.0×lr)")
                print(f"  Cosine decay: {max_epochs - warmup_epochs} epochs (lr → {opt_config.min_lr})")
                print(f"  Total epochs: {max_epochs}")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": combined_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        # Default: validation-loss plateau schedule (existing behavior)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=opt_config.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_after_backward(self):
        # Check if encoded gradient is available
        if hasattr(self, "_encoded_ref"):
            if self._encoded_ref is not None:
                if self._encoded_ref.grad is not None:
                    log.debug(f"[DEBUG] encoded.grad norm: {self._encoded_ref.grad.norm().item():.6f}")
                else:
                    log.debug("[DEBUG] encoded.grad is still None after backward.")
            else:
                log.debug("[DEBUG] _encoded_ref is None")

        # x_hidden grad
        if hasattr(self, "_x_hidden_ref"):
            if self._x_hidden_ref is not None and self._x_hidden_ref.grad is not None:
                log.debug(f"[DEBUG] x_hidden.grad norm: {self._x_hidden_ref.grad.norm().item():.6f}")
            else:
                log.debug("[DEBUG] x_hidden.grad is still None after backward.")

        # Print all parameter gradients
        if self.trainer.is_global_zero:
            total_grad_norm = 0.0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    norm = param.grad.data.norm(2)
                    log.debug(f"[DEBUG] Grad for {name}: {norm:.6f}")
                    total_grad_norm += norm.item() ** 2
                else:
                    log.debug(f"[DEBUG] Grad for {name}: None")
            total_grad_norm = total_grad_norm**0.5
            log.debug(f"[DEBUG] Total Gradient Norm: {total_grad_norm:.6f}")


    def get_current_rollout_steps(self):
        """
        Determines the current number of rollout steps based on training progress.
        Implements curriculum learning where rollout length increases over time.
        """

        current_epoch = self.current_epoch
        current_step = self.global_step  # This tracks gradient descent updates

        rollout_schedule = self.training_config.data.rollout_schedule

        if rollout_schedule == "graphcast":
            # GraphCast schedule based on gradient descent updates
            # Graphcast: 300,000 gradient descent updates - 1 autoregressive
            #            300,001 to 311,000: add 1 per 1000 updates
            #           (i.e., use 1000 steps for each autoregressive step)
            # testing functionality: train 1 rollout for 5 epochs [0-4], add 1 for every epoch
            threshold = 5  # 300000 # MK: using 5 for testing
            interval = 1  # 1000
            if current_step < threshold:
                return 1
            else:
                additional_steps = 2 + (current_step - threshold) // interval
                return min(additional_steps, self.training_config.data.max_rollout_steps)

        elif rollout_schedule == "linear":
            # Linearly increase from 1 to max_rollout_steps over training
            max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
            progress = min(current_epoch / max_epochs, 1.0)
            current_steps = 1 + int(progress * (self.training_config.data.max_rollout_steps - 1))
            return current_steps

        elif rollout_schedule == "step":
            # Step-wise increase (GraphCast style)
            if current_epoch < 10:
                return 1
            elif current_epoch < 20:
                return 2
            else:
                return min(self.training_config.data.max_rollout_steps, 3 + (current_epoch - 20) // 10)

        else:
            # "fixed"
            return self.training_config.data.max_rollout_steps


    def _compute_channel_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        instrument_ids: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        
        if self.training_config.loss.type == "mse":
            return weighted_mse_loss(
                y_pred,
                y_true,
                instrument_ids=instrument_ids,
                channel_weights=self.obs_config.channel_weights,
                rebalancing=True,
                valid_mask=valid_mask,
            )
        return weighted_huber_loss(
            y_pred,
            y_true,
            instrument_ids=instrument_ids,
            channel_weights=self.obs_config.channel_weights,
            delta=self.training_config.loss.huber_delta,
            rebalancing=True,
            valid_mask=valid_mask,
        )

