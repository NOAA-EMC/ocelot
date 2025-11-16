"""
Lightning callback for computing and tracking FSOI during training/validation.

This callback computes FSOI metrics at specified intervals and saves:
- CSV files with detailed FSOI values per observation
- Summary plots showing spatial distribution and statistics
- Time series of FSOI metrics across epochs

Uses two-state adjoint method for fast computation.

Author: Azadeh Gholoubi
Date: November 2025
"""

import os
from typing import Optional

import lightning.pytorch as pl
import pandas as pd
import torch

from fsoi import compute_batch_fsoi, summarize_fsoi_by_instrument
from fsoi_plots import create_fsoi_summary_report


class FSOICallback(pl.Callback):
    """
    PyTorch Lightning callback for FSOI computation and tracking.

    This callback:
    - Computes FSOI on validation batches at specified intervals
    - Saves detailed FSOI statistics to CSV
    - Generates visualization plots
    - Tracks FSOI metrics over training
    - Uses two-state adjoint method for fast computation
    - Supports sequential background using previous batches

    Args:
        compute_every_n_epochs: Compute FSOI every N epochs (default: 5)
        save_dir: Directory to save FSOI results
        max_batches: Maximum number of batches to analyze per epoch (to save time)
        generate_plots: Whether to generate visualization plots
        start_epoch: Start computing FSOI from this epoch (default: 0, can set to 5-10 to skip early training)
        feature_stats: Climatological statistics (mean, std) from observation_config.yaml
        conventional_only: Only compute for conventional obs (surface_obs + radiosonde)
        use_sequential_background: Use forecast from previous batch as background
    """

    def __init__(
        self,
        compute_every_n_epochs: int = 5,
        save_dir: str = "fsoi_results",
        max_batches: int = 3,
        generate_plots: bool = True,
        start_epoch: int = 0,
        feature_stats: Optional[dict] = None,
        conventional_only: bool = False,
        use_sequential_background: bool = True,
    ):
        super().__init__()
        self.compute_every_n_epochs = compute_every_n_epochs
        self.save_dir = save_dir
        self.max_batches = max_batches
        self.generate_plots = generate_plots
        self.start_epoch = start_epoch
        self.feature_stats = feature_stats
        self.conventional_only = conventional_only
        self.use_sequential_background = use_sequential_background

        # Storage for time series tracking
        self.fsoi_history = []

        # Cache for sequential background
        self._previous_batch = None
        self._previous_predictions = None
        self._previous_bin_name = None

        os.makedirs(self.save_dir, exist_ok=True)
        
        # Operational mode is now handled directly in fsoi.py
        # No need for separate operational calculator
        self.operational_calculator = None

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """
        Compute FSOI at the end of validation batches.
        
        Per-rank sequential mode: Each rank processes its bins sequentially.
        Sequential background works within each rank (bin2 uses bin1 forecast on same GPU).
        
        If use_sequential_background=True and previous batch exists:
        - Uses forecast from previous batch as background (x_b)
        - Implements per-rank sequential FSOI with two-state adjoint
        Otherwise:
        - Uses zero/climatological background (approximate)
        """
        # Only compute on specific epochs
        current_epoch = trainer.current_epoch
        
        # Skip if before start_epoch
        if current_epoch < self.start_epoch:
            return
            
        if current_epoch % self.compute_every_n_epochs != 0:
            return

        # Only process first few batches to save computation time
        if batch_idx >= self.max_batches:
            return

        # Only run on rank 0 to avoid duplication in distributed setting
        if not trainer.is_global_zero:
            return

        print(f"\n[FSOI Callback] Computing FSOI for epoch {current_epoch}, batch {batch_idx}")

        # Check if we can use sequential background
        can_use_sequential = (
            self.use_sequential_background
            and self._previous_batch is not None
            and self._can_link_batches(self._previous_batch, batch)
        )

        if can_use_sequential:
            print(f"[FSOI Callback] Using sequential background from previous batch (per-rank)")
            print(f"[FSOI Callback]   Previous: {self._previous_bin_name}")
            print(f"[FSOI Callback]   Current:  {batch.bin_name}")
        else:
            if self.use_sequential_background and self._previous_batch is None:
                print(f"[FSOI Callback] First batch - using climatological background (x_b=0)")
            elif self.use_sequential_background:
                print(f"[FSOI Callback] Non-sequential bins - using climatological background")
                print(f"[FSOI Callback]   Previous: {self._previous_bin_name}")
                print(f"[FSOI Callback]   Current:  {batch.bin_name}")
            else:
                print(f"[FSOI Callback] Sequential background disabled - using climatological")

        try:
            # Clear model gradients before FSOI computation
            pl_module.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            
            # CRITICAL: Enable gradients for FSOI computation
            # trainer.validate() puts model in eval mode, but FSOI needs gradients
            pl_module.train()
            torch.set_grad_enabled(True)
            
            # Use unified compute_batch_fsoi from fsoi.py
            print(f"[FSOI Callback] Using two-state adjoint method")
            if self.conventional_only:
                print(f"[FSOI Callback] Target: CONVENTIONAL obs only (surface_obs + radiosonde)")
            
            # Compute FSOI (function will use previous predictions if available)
            fsoi_values, stats_df = compute_batch_fsoi(
                model=pl_module,
                batch=batch,
                save_dir=os.path.join(self.save_dir, "detailed"),
                epoch=current_epoch,
                batch_idx=batch_idx,
                compute_sensitivity=True,
                feature_stats=self.feature_stats,
                conventional_only=self.conventional_only,
                # Pass previous batch info for sequential background
                previous_batch=self._previous_batch if can_use_sequential else None,
                previous_predictions=self._previous_predictions if can_use_sequential else None,
            )

            # Cache current batch and predictions for next iteration
            self._previous_batch = batch
            self._previous_bin_name = batch.bin_name
            
            # Restore eval mode and disable gradients for normal validation
            with torch.no_grad():
                pl_module.eval()
                self._previous_predictions = pl_module(batch)
            
            # Ensure gradients are disabled after FSOI computation
            torch.set_grad_enabled(False)

            if len(stats_df) > 0:
                # Generate summary plots if requested
                if self.generate_plots and batch_idx == 0:  # Only for first batch
                    plot_dir = os.path.join(self.save_dir, "plots", f"epoch{current_epoch}")
                    create_fsoi_summary_report(
                        stats_df=stats_df, output_dir=plot_dir, epoch=current_epoch, batch_idx=batch_idx
                    )

                # Store summary statistics
                summary = summarize_fsoi_by_instrument(stats_df)
                summary["epoch"] = current_epoch
                summary["batch_idx"] = batch_idx
                self.fsoi_history.append(summary)

                # Log key metrics to trainer (avoid batch_size extraction from HeteroData)
                mean_fsoi = float(stats_df["fsoi_value"].mean())
                std_fsoi = float(stats_df["fsoi_value"].std())
                beneficial_pct = float(100 * (stats_df["fsoi_value"] > 0).sum() / len(stats_df))

                # Use prog_bar=False and batch_size=1 to avoid HeteroData iteration error
                pl_module.log("fsoi_mean", mean_fsoi, on_epoch=True, prog_bar=False, batch_size=1)
                pl_module.log("fsoi_std", std_fsoi, on_epoch=True, prog_bar=False, batch_size=1)
                pl_module.log("fsoi_beneficial_pct", beneficial_pct, on_epoch=True, prog_bar=False, batch_size=1)

                print(f"[FSOI Callback] Mean FSOI: {mean_fsoi:.4e}, Beneficial: {beneficial_pct:.1f}%")
            
            # Clear FSOI computation results to free memory
            del fsoi_values, stats_df
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[FSOI Callback] Error computing FSOI: {e}")
            import traceback

            traceback.print_exc()

    def _can_link_batches(self, batch_prev, batch_curr) -> bool:
        """
        Check if two batches are temporally sequential (can be linked for background).
        
        Sequential batches have consecutive bin numbers (e.g., bin123 -> bin124),
        indicating they are 12 hours apart and can be linked for background.
        
        Args:
            batch_prev: Previous batch
            batch_curr: Current batch
            
        Returns:
            True if batches are sequential (bin_curr = bin_prev + 1)
        """
        try:
            # Extract bin numbers from bin names (e.g., "bin123" -> 123)
            bin_prev = int(batch_prev.bin_name.replace("bin", ""))
            bin_curr = int(batch_curr.bin_name.replace("bin", ""))
            
            # Check if consecutive (12h apart with 12h bins)
            is_sequential = (bin_curr - bin_prev) == 1
            
            if is_sequential:
                print(f"[FSOI Callback] Batches are SEQUENTIAL: {bin_prev} -> {bin_curr}")
            else:
                print(f"[FSOI Callback] Batches are NOT sequential: {bin_prev} -> {bin_curr} (gap={bin_curr-bin_prev})")
            
            return is_sequential
            
        except (AttributeError, ValueError) as e:
            print(f"[FSOI Callback] Could not parse bin names: {e}")
            return False

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Reset cache at the start of each validation epoch."""
        if trainer.is_global_zero:
            print(f"\n[FSOI Callback] Starting validation epoch {trainer.current_epoch}")
            if self.use_sequential_background:
                print(f"[FSOI Callback] Sequential background enabled - will use previous batch forecasts when available")
            # Reset cache for new epoch (batches from different epochs shouldn't be linked)
            self._previous_batch = None
            self._previous_predictions = None
            self._previous_bin_name = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Save FSOI history at the end of each validation epoch.
        """
        if not trainer.is_global_zero:
            return

        current_epoch = trainer.current_epoch
        if current_epoch % self.compute_every_n_epochs != 0:
            return

        if len(self.fsoi_history) > 0:
            # Combine all history into a single DataFrame
            history_df = pd.concat(self.fsoi_history, ignore_index=True)

            # Save to CSV
            history_path = os.path.join(self.save_dir, "fsoi_history.csv")
            history_df.to_csv(history_path, index=False)
            print(f"[FSOI Callback] Saved FSOI history to {history_path}")

            # Create time series plot
            self._plot_fsoi_time_series(history_df)

    def _plot_fsoi_time_series(self, history_df: pd.DataFrame):
        """
        Create time series plot of FSOI evolution over training.
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot 1: Mean FSOI by instrument over epochs
            instruments = history_df["instrument"].unique()
            for inst in instruments:
                inst_data = history_df[history_df["instrument"] == inst]
                epochs = inst_data["epoch"].values
                mean_fsoi = inst_data["fsoi_value_mean"].values

                axes[0].plot(epochs, mean_fsoi, marker="o", label=inst, linewidth=2)

            axes[0].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            axes[0].set_xlabel("Epoch", fontsize=12)
            axes[0].set_ylabel("Mean FSOI", fontsize=12)
            axes[0].set_title("FSOI Evolution by Instrument", fontsize=13)
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Plot 2: Total FSOI by instrument
            for inst in instruments:
                inst_data = history_df[history_df["instrument"] == inst]
                epochs = inst_data["epoch"].values
                total_fsoi = inst_data["fsoi_value_sum"].values

                axes[1].plot(epochs, total_fsoi, marker="o", label=inst, linewidth=2)

            axes[1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            axes[1].set_xlabel("Epoch", fontsize=12)
            axes[1].set_ylabel("Total FSOI", fontsize=12)
            axes[1].set_title("Cumulative FSOI by Instrument", fontsize=13)
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(self.save_dir, "fsoi_time_series.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[FSOI Callback] Saved time series plot to {plot_path}")

        except Exception as e:
            print(f"[FSOI Callback] Error creating time series plot: {e}")


class FSOIAnalysisCallback(pl.Callback):
    """
    Simplified FSOI callback that only computes FSOI once (e.g., at the end of training).

    This is useful for final model evaluation without the overhead of computing FSOI
    at every epoch.

    Args:
        compute_on_epoch: Specific epoch to compute FSOI (default: last epoch)
        save_dir: Directory to save results
        num_batches: Number of validation batches to analyze
    """

    def __init__(self, compute_on_epoch: Optional[int] = None, save_dir: str = "fsoi_final", num_batches: int = 10):
        super().__init__()
        self.compute_on_epoch = compute_on_epoch
        self.save_dir = save_dir
        self.num_batches = num_batches
        self.batch_stats = []

        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Collect FSOI from validation batches."""
        current_epoch = trainer.current_epoch

        # Check if we should compute
        if self.compute_on_epoch is not None and current_epoch != self.compute_on_epoch:
            return

        # Check if this is the last epoch (if compute_on_epoch not specified)
        if self.compute_on_epoch is None and current_epoch < trainer.max_epochs - 1:
            return

        if batch_idx >= self.num_batches:
            return

        if not trainer.is_global_zero:
            return

        print(f"[FSOI Analysis] Computing FSOI for batch {batch_idx}/{self.num_batches}")

        try:
            _, stats_df = compute_batch_fsoi(
                model=pl_module,
                batch=batch,
                save_dir=os.path.join(self.save_dir, "batches"),
                epoch=current_epoch,
                batch_idx=batch_idx,
                compute_sensitivity=True,
            )

            if len(stats_df) > 0:
                self.batch_stats.append(stats_df)

        except Exception as e:
            print(f"[FSOI Analysis] Error: {e}")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate comprehensive FSOI report."""
        if not trainer.is_global_zero:
            return

        if len(self.batch_stats) == 0:
            return

        print(f"\n[FSOI Analysis] Generating final FSOI report from {len(self.batch_stats)} batches")

        # Combine all batches
        combined_df = pd.concat(self.batch_stats, ignore_index=True)

        # Save combined statistics
        combined_path = os.path.join(self.save_dir, "fsoi_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"[FSOI Analysis] Saved combined FSOI data to {combined_path}")

        # Generate comprehensive plots
        create_fsoi_summary_report(
            stats_df=combined_df, output_dir=os.path.join(self.save_dir, "plots"), epoch=trainer.current_epoch, batch_idx=0
        )

        # Print summary statistics
        print("\n" + "=" * 60)
        print("FSOI ANALYSIS SUMMARY")
        print("=" * 60)

        summary = summarize_fsoi_by_instrument(combined_df)
        print(summary.to_string())

        print("\nOverall Statistics:")
        print(f"  Total observations: {len(combined_df)}")
        print(f"  Mean FSOI: {combined_df['fsoi_value'].mean():.4e}")
        print(f"  Std FSOI: {combined_df['fsoi_value'].std():.4e}")
        print(f"  Median FSOI: {combined_df['fsoi_value'].median():.4e}")
        print(f"  Beneficial obs: {(combined_df['fsoi_value'] > 0).sum()} ({100*(combined_df['fsoi_value'] > 0).sum()/len(combined_df):.1f}%)")
        print(f"  Detrimental obs: {(combined_df['fsoi_value'] < 0).sum()} ({100*(combined_df['fsoi_value'] < 0).sum()/len(combined_df):.1f}%)")
        print("=" * 60 + "\n")

        # Clear batch stats to free memory
        self.batch_stats = []
