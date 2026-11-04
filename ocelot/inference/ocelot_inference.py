import os
import sys

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class OcelotInferenceModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data: HeteroData):
        return self.model(data)

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference mode.

        This method:
        1. Runs forward pass to generate predictions
        2. Saves predictions to CSV files
        3. Does NOT compute loss or gradients

        Args:
            batch: Input batch data
            batch_idx: Batch index

        Returns:
            dict: Predictions for all node types
        """
        print(f"[PREDICT] Processing batch {batch_idx}: {batch.bin_name}")

        target_init_filter = os.environ.get("PREDICT_INIT_TIME_FILTER", "").strip()
        if target_init_filter:
            batch_init_time = self.model._extract_init_time_str(batch)
            if batch_init_time != target_init_filter:
                print(
                    f"[PREDICT] Skipping batch {batch_idx}: init_time={batch_init_time} "
                    f"does not match PREDICT_INIT_TIME_FILTER={target_init_filter}"
                )
                return {}

        # Forward pass
        forward_output = self(batch)
        if isinstance(forward_output, tuple):
            all_predictions, mesh_features_per_step = forward_output
        else:
            all_predictions = forward_output
            mesh_features_per_step = None

        # Extract ground truths and metadata
        ground_truth_data = self.model._extract_ground_truths_and_metadata(batch, all_predictions)

        # Determine rollout steps
        step_info = self.model._get_latent_step_info(batch)
        latent_rollout_steps = step_info["num_steps"]
        print(f"[PREDICT] Latent rollout steps: {latent_rollout_steps}")

        # Save predictions for each instrument
        for node_type, preds_list in all_predictions.items():
            print(f"[PREDICT] Processing node_type: {node_type}")

            if node_type not in ground_truth_data:
                continue

            gt_data = ground_truth_data[node_type]
            gts_list = gt_data["gts_list"]
            valid_mask_list = gt_data["valid_mask_list"]

            # Check if any real ground truth data exists
            has_real_targets = any(
                gt is not None and gt.numel() > 0
                for gt in gts_list
            )

            if not has_real_targets:
                print(f"[PREDICT] Pred step: Skipping {node_type} - no ground truth data (inference mode)")
                continue

            # Save to CSV
            # if batch_idx < 10:
            out_dir = os.path.join(self._prediction_output_dir, 'pred_csv', 'obs-space')
            self.model.save_latent_concatenated_csv(
                current_epoch=self.current_epoch,
                batch=batch,
                node_type=node_type,
                preds_list=preds_list,
                gts_list=gts_list,
                valid_mask_list=valid_mask_list,
                out_dir=out_dir,
                batch_idx=batch_idx,
                mode='predict'
            )

        # Save mesh predictions (target variables on grid)
        if self.model.enable_mesh_pred:
            try:
                with torch.no_grad():
                    # Use mesh features from forward pass
                    if not mesh_features_per_step:
                        print("[PREDICT] No mesh features available for mesh predictions")
                    else:
                        mesh_pred_edges = self.model.get_mesh_pred_edges()
                        init_time_unix = self.model._extract_init_time_unix(batch)
                        mesh_predictions = self.model._decode_all_steps_to_mesh(mesh_features_per_step, mesh_pred_edges, init_time_unix)
                        if mesh_predictions:
                            mesh_dir = os.path.join(self._prediction_output_dir, 'pred_csv', 'mesh-grid')
                            self.model.save_mesh_predictions(
                                mesh_predictions,
                                mesh_pred_edges,
                                batch_idx=batch_idx,
                                epoch=0,
                                mode='predict',
                                batch=batch,
                                output_dir=mesh_dir
                            )
            except Exception as e:
                print(f"[PREDICT] Mesh prediction failed (non-critical): {e}")
                import traceback
                traceback.print_exc()

        return all_predictions

    def on_predict_epoch_start(self):
        """Setup before prediction epoch starts."""
        print("[PREDICT] Starting prediction epoch")
        if not self.model.enable_mesh_pred:
            print("[WARN] Mesh prediction is disabled in the pipeline configuration.")
        self._mesh_predictions_buffer = {}
        self._prediction_output_dir = self.model.pipeline_config.prediction_output_dir
        os.makedirs(self._prediction_output_dir, exist_ok=True)
        # Create pred_csv subdirectories
        os.makedirs(os.path.join(self._prediction_output_dir, 'pred_csv', 'obs-space'), exist_ok=True)
        os.makedirs(os.path.join(self._prediction_output_dir, 'pred_csv', 'mesh-grid'), exist_ok=True)
        print(f"[PREDICT] Output directory: {self._prediction_output_dir}")

    def on_predict_batch_end(self, outputs, batch, batch_idx):
        """Cleanup after each prediction batch."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_predict_epoch_end(self):
        """Cleanup and summary after prediction epoch ends."""
        print("[PREDICT] Prediction epoch completed")
        self._cached_mesh_pred_edges = None

        # Generate summary statistics
        if hasattr(self, '_prediction_output_dir'):
            obs_dir = os.path.join(self._prediction_output_dir, 'pred_csv', 'obs-space')
            if os.path.exists(obs_dir):
                csv_files = [f for f in os.listdir(obs_dir) if f.endswith('.csv')]
                print(f"[PREDICT] Generated {len(csv_files)} observation CSV files (obs-space)")

            mesh_dir = os.path.join(self._prediction_output_dir, 'pred_csv', 'mesh-grid')
            if os.path.exists(mesh_dir):
                mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.csv')]
                print(f"[PREDICT] Generated {len(mesh_files)} mesh CSV files (mesh-grid)")
