# Standard library

import math
import re

import numpy as np
import torch
import torch.distributed as dist

# Third-party
import matplotlib.pyplot as plt
import pytorch_lightning as pl

# Local
from ocelot import metrics_dop, vis


from torch.optim.lr_scheduler import LambdaLR


def get_world_size_safe():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _instrument_key_for_obs_config(instrument_key: str) -> str:
    """
    Map target node name segments to ``observation_config`` keys.

    Multi-bin targets use ``…_h0``, ``…_h1`` suffixes on the instrument part
    (see ``graph_utils.target_node_type``); config still uses the base name
    (e.g. ``surface_obs``, not ``surface_obs_h0``).
    """
    return re.sub(r'_h\d+$', '', instrument_key)


def scale_learning_rate(base_lr, base_world_size=1, scale_type="sqrt"):
    world_size = get_world_size_safe()

    if scale_type == "linear":
        scaled_lr = base_lr * (world_size / base_world_size)
    elif scale_type == "sqrt":
        scaled_lr = base_lr * (world_size / base_world_size) ** 0.5
    else:
        scaled_lr = base_lr  # no scaling

    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        print(f"✅ Using scaled LR = {scaled_lr:.3e} (world_size={world_size}, mode={scale_type})")
    return scaled_lr


def warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """
    Create a linear warm-up + cosine decay scheduler.
    During warm-up: LR increases from min_lr_ratio*base_lr to base_lr.
    After warm-up: Cosine decay toward min_lr_ratio*base_lr.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warm-up
            return min_lr_ratio + (1 - min_lr_ratio) * (step / warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)



class ARDOPModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # Instantiate loss function
        self.loss = metrics_dop.get_metric(args.loss)


    def _log_channel_stats(self, obs_type, node, pred_tensor, feature_names):
        """Print per-channel bias/RMSE/std table in normalized units for diagnostic use."""
        y_all = node.y.cpu().detach().numpy()           # (N, C)
        p_all = pred_tensor.cpu().detach().numpy()      # (N, C)
        p_all = p_all[:y_all.shape[0]]                  # align to same N

        valid_mask = None
        if hasattr(node, 'valid_mask') and node.valid_mask is not None:
            valid_mask = node.valid_mask.cpu().detach().numpy()  # (N, C) or (N,)

        n_ch = y_all.shape[1]
        sep = '=' * 72
        header = (
            f"\n{sep}\n"
            f"[channel stats] {obs_type}  epoch={self.current_epoch}  step={self.global_step}\n"
            f"{'ch':<5} {'feature':<32} {'n_valid':>7} {'bias':>8} {'rmse':>8} "
            f"{'pred_std':>9} {'truth_std':>9}\n"
            f"{'-' * 72}"
        )
        rows = [header]
        for ci in range(n_ch):
            y_ch = y_all[:, ci]
            p_ch = p_all[:, ci]
            if valid_mask is not None:
                m = valid_mask[:, ci] if valid_mask.ndim == 2 else valid_mask
                y_ch, p_ch = y_ch[m], p_ch[m]
            n_valid = len(y_ch)
            fname = feature_names[ci] if ci < len(feature_names) else f'ch{ci}'
            if n_valid == 0:
                rows.append(f"  {ci:<4} {fname:<32} {'0':>7} {'NO VALID DATA':>8}")
                continue
            bias = float(np.mean(p_ch - y_ch))
            rmse = float(np.sqrt(np.mean((p_ch - y_ch) ** 2)))
            pred_std = float(np.std(p_ch))
            truth_std = float(np.std(y_ch))
            rows.append(
                f"  {ci:<4} {fname:<32} {n_valid:>7} {bias:>8.4f} {rmse:>8.4f} "
                f"{pred_std:>9.4f} {truth_std:>9.4f}"
            )
        rows.append(sep)
        print('\n'.join(rows))

    def plot_examples(self, batch, predictions):
        """
        Plot forecasts for each example in the batch.

        batch: The input batch with data to plot.
        target_dict: The dictionary with target data.
        predicted_dict: The dictionary with predicted data.
        """
        # Only plot on rank 0 to avoid duplicate plots and file conflicts in multi-GPU training
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return
        
        del batch[('mesh', 'to', 'mesh')]
        batch_1 = batch.to_data_list()[0]
        target_list = []
        for node_type in batch_1.node_types:
            if node_type.endswith('_target'):
                first_part = node_type[:-len('_target')]
                target_list.append(first_part)
        for obs_type in target_list:
            # Dynamically find the feature stats instead of using a hardcoded dictionary
            # obs_type is like 'satellite_atms' or 'conventional_surface_obs'
            # We need to find the corresponding config to get the feature name
            obs_category, *inst_parts = obs_type.split('_')
            instrument_key = '_'.join(inst_parts)
            cfg_key = _instrument_key_for_obs_config(instrument_key)

            # Get the name of the first feature being plotted
            try:
                first_feature_name = self.args.observation_config[obs_category][cfg_key]['features'][0]
            except KeyError as e:
                print(f"Skipping plotting for {obs_type}: could not find observation_config entry. Error: {e}")
                continue

            # Get the stats for that feature if available; otherwise fall back to [0, 1]
            stats_dict = self.args.feature_stats.get(cfg_key, {})
            obs_stats = stats_dict.get(first_feature_name, [0.0, 1.0])

            node = batch_1[f'{obs_type}_target']

            # --- Per-channel diagnostic stats ---
            feature_names = self.args.observation_config[obs_category][cfg_key].get('features', [])
            pred_key = f'{obs_type}_target'
            if pred_key in predictions:
                self._log_channel_stats(
                    obs_type, node,
                    predictions[pred_key],
                    feature_names,
                )
            target_lat_deg = node.pos[:, 1].cpu().detach().numpy()
            target_lon_deg = node.pos[:, 0].cpu().detach().numpy()

            # Get the corresponding prediction and target tensors
            target = node.y[:, 0].cpu().detach().numpy()
            predicted = predictions[f'{obs_type}_target'][:, 0].cpu().detach().numpy()
            predicted = predicted[0:target.shape[0]]

            if hasattr(node, 'valid_mask') and node.valid_mask is not None:
                mask = node.valid_mask.cpu().detach().numpy().astype(bool)
                if mask.ndim == 2:
                    mask = mask[:, 0]
                target = target[mask]
                predicted = predicted[mask]
                target_lat_deg = target_lat_deg[mask]
                target_lon_deg = target_lon_deg[mask]

            # For radiosonde, restrict plots to the first (highest) pressure level only
            is_radiosonde = (
                'radiosonde' in obs_type.lower()
                and hasattr(node, 'x') and node.x is not None
            )
            if is_radiosonde:
                # airPressure is stored in metadata (x); find its index from observation_config
                try:
                    metadata_keys = self.args.observation_config[obs_category][cfg_key].get('metadata', [])
                    airpressure_idx = metadata_keys.index('airPressure') if 'airPressure' in metadata_keys else 0
                    pressure = node.x[:, airpressure_idx].cpu().detach().numpy()
                    valid_mask = ~np.isnan(pressure)
                    pressure_valid = pressure[valid_mask]
                    if pressure_valid.size > 0:
                        # Choose the highest pressure level (surface) as the "first level"
                        first_level = float(np.max(pressure_valid))
                        level_mask = (pressure == first_level)
                        if level_mask.any():
                            target = target[level_mask]
                            predicted = predicted[level_mask]
                            target_lat_deg = target_lat_deg[level_mask]
                            target_lon_deg = target_lon_deg[level_mask]
                            # If per-level stats are available, override obs_stats with this level's stats
                            by_level = self.args.feature_stats.get(cfg_key, {}).get('mean_std_by_level', {})
                            level_key = int(round(first_level))
                            if first_feature_name in by_level and level_key in by_level[first_feature_name]:
                                obs_stats = by_level[first_feature_name][level_key]
                except Exception as e:
                    print(f"Warning: could not apply radiosonde level filter in plot_examples: {e}")
            
            # Generate unique plot names for each sample and instrument
            # Use global_step for step-based validation, fallback to epoch for epoch-based
            plot_name = f'{obs_type}_epoch{self.current_epoch}_step{self.global_step}'
        
            vis.plot_hist(
                plot_name,
                [predicted, target],
                obs_stats,
                exp_name=self.args.exp_name,
                version=getattr(self.args, 'version', 'version_0')
            )
            vis.plot_map(
                plot_name,
                target_lon_deg,
                target_lat_deg,
                [predicted, target],
                obs_stats,
                title=plot_name,
                exp_name=self.args.exp_name,
                version=getattr(self.args, 'version', 'version_0')
            )

        plt.close("all")  # Close all figs for this time step, saves memory

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log the current learning rate after each training batch."""
        # Get current learning rate from the optimizer
        current_lr = self.optimizers().param_groups[0]['lr']
        # Log it to the logger (will appear in metrics.csv)
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def configure_optimizers(self):
        base_lr = self.args.lr
        scaled_lr = scale_learning_rate(base_lr, scale_type="sqrt")
        optimizer = torch.optim.AdamW(self.parameters(), lr=scaled_lr, weight_decay=1e-5)

        # Try to get estimated stepping batches, but handle case where trainer isn't fully set up
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except (AttributeError, ValueError) as e:
            # Fallback: estimate based on max_epochs and a reasonable batch count
            # This is a rough estimate that will be corrected when training actually starts
            if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs:
                estimated_batches_per_epoch = 100  # Conservative estimate
                total_steps = self.trainer.max_epochs * estimated_batches_per_epoch
                if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                    print(f"⚠️  Could not get estimated_stepping_batches: {e}")
                    print(f"⚠️  Using fallback estimate: {total_steps} steps")
            else:
                # Last resort fallback
                total_steps = 500
                if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                    print(f"⚠️  Could not estimate total steps, using default: {total_steps}")
        
        warmup_steps = int(0.05 * total_steps)  # e.g. 5% warm-up

        scheduler = warmup_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.01,  # min LR = 1% of scaled LR (allows more fine-tuning at end)
        )

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print(f"✅ Warm-up for {warmup_steps} steps, total {total_steps}, scaled_lr={scaled_lr:.2e}")

        return [optimizer], [scheduler_dict]
