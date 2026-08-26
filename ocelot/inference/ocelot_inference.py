import os, sys
import lightning.pytorch as pl
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
            batch_init_time = self._extract_init_time_str(batch)
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
            self.model._save_latent_concatenated_csv(
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
        if self.enable_mesh_pred:
            try:
                with torch.no_grad():
                    # Use mesh features from forward pass
                    if not mesh_features_per_step:
                        print("[PREDICT] No mesh features available for mesh predictions")
                    else:
                        mesh_pred_edges = self.model._get_mesh_pred_edges()
                        init_time_unix = self.model._extract_init_time_unix(batch)
                        mesh_predictions = self.model._decode_all_steps_to_mesh(mesh_features_per_step, mesh_pred_edges, init_time_unix)
                        if mesh_predictions:
                            mesh_dir = os.path.join(self._prediction_output_dir, 'pred_csv', 'mesh-grid')
                            self._save_mesh_predictions(
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
        if not self.enable_mesh_pred:
            print("[WARN] enable_mesh_pred is False — mesh grid outputs will NOT be generated. "
                  "Set 'enable_mesh_pred: true' in mesh_config.yaml to produce gridded outputs.")
        self._mesh_predictions_buffer = {}
        self._prediction_output_dir = getattr(self, 'prediction_output_dir', 'predictions')
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


    def _save_mesh_predictions(self, predictions, mesh_pred_edges, batch_idx, epoch, mode='val', batch=None, output_dir='val_mesh_csv'):
        """
        Save predictions on mesh grid - one file per forecast hour.

        Args:
            predictions: Dict of predictions per instrument
            batch_idx: Batch index
            epoch: Epoch number
            mode: 'val' or 'predict'
            batch: Batch data (for extracting input_time)
            output_dir: Directory to save files
        """

        os.makedirs(output_dir, exist_ok=True)

        # Extract init time for logging
        init_time_str = self._extract_init_time_str(batch)

        # Calculate forecast hours
        num_steps = len(next(iter(predictions.values())))
        latent_step_hours = self.latent_step_hours
        forecast_hours = [(i + 1) * latent_step_hours for i in range(num_steps)]

        print(f"[MESH PRED] Init time: {init_time_str}, Forecast hours: {forecast_hours} (latent_step={latent_step_hours}h, steps={num_steps})")

        for inst_name, pred_list in predictions.items():
            edges = mesh_pred_edges[inst_name]
            mesh_lats = edges['lats']
            mesh_lons = edges['lons']
            base_inst_name = inst_name.replace('_target', '')

            # Get target variables (only the ones we want to predict on mesh)
            mesh_vars = self.mesh_variable_config.get('variables', {}).get(inst_name, [])

            # Get ALL features and find indices of target variables
            obs_type = "satellite" if inst_name in self.observation_config.get("satellite", {}) else "conventional"
            all_features = self.observation_config[obs_type][inst_name]['features']

            # Find indices of target variables
            mesh_indices = [i for i, feat in enumerate(all_features) if feat in mesh_vars]

            for step_idx, (pred_tensor, fhr) in enumerate(zip(pred_list, forecast_hours)):
                # Unnormalize using existing method
                node_type = f"{inst_name}_target"
                pred_unnorm = self.unnormalize_standardscaler(pred_tensor, node_type)
                pred_np = pred_unnorm.detach().cpu().numpy()

                df = pd.DataFrame({
                    'mesh_idx': np.arange(len(mesh_lats), dtype=np.int64),
                    'lat': mesh_lats,
                    'lon': mesh_lons,
                })

                # Only add pressure columns for instruments that use pressure-level conditioning
                if base_inst_name in ['radiosonde', 'aircraft']:
                    STANDARD_PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]
                    pressure_hpa = STANDARD_PRESSURE_LEVELS[self.mesh_pressure_level_idx]
                    log_pressure_height = -8000.0 * np.log(np.clip(pressure_hpa, 1.0, 1100.0) / 1013.25)

                    df['pressure_hPa'] = pressure_hpa
                    df['pressure_level_idx'] = self.mesh_pressure_level_idx
                    df['pressure_level_label'] = f"{pressure_hpa}hPa"
                    df['log_pressure_height_m'] = log_pressure_height
                    df['log_pressure_height_norm'] = log_pressure_height / 20000.0

                # Add only target variable predictions
                for feat_idx in mesh_indices:
                    feat_name = all_features[feat_idx]
                    df[f'pred_{feat_name}'] = pred_np[:, feat_idx]

                # Use init_time if available, otherwise fall back to batch_idx
                if init_time_str != 'unknown':
                    if mode == 'predict':
                        filepath = f'{output_dir}/{base_inst_name}_init_{init_time_str}_f{fhr:03d}.csv'
                    else:
                        filepath = f'{output_dir}/{base_inst_name}_init_{init_time_str}_f{fhr:03d}_epoch{epoch}_batch{batch_idx}.csv'
                else:
                    filepath = f'{output_dir}/{base_inst_name}_f{fhr:03d}_epoch{epoch}_batch{batch_idx}.csv'
                df.to_csv(filepath, index=False)
                print(f"[MESH PRED] Saved {filepath}: {len(df)} points")
