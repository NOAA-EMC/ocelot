"""
PyTorch Lightning callbacks for GNN training with DDP synchronization.

Provides callbacks for:
- Window-based data resampling with synchronized date ranges across ranks
- Sequential and random sampling modes for temporal data
- Epoch-based sampler management

Author: Azadeh Gholoubi (NOAA/EMC)
Date: November 2025
"""

import random
import os
from typing import Optional

import pandas as pd
import torch
import lightning.pytorch as pl
import torch.distributed as dist


# -----------------------------
# DDP helpers
# -----------------------------


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_ready() else 0


def _is_rank0() -> bool:
    return _rank() == 0


def _broadcast_window(start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """
    Backend-agnostic broadcast of (start, end) from rank-0 to all ranks.
    Uses broadcast_object_list (CPU objects; no device issues).
    """
    if not _dist_ready():
        return start_dt, end_dt
    payload = [start_dt, end_dt] if _is_rank0() else [None, None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0], payload[1]


# -----------------------------
# Verbose rank/device header for logs
# -----------------------------


def log_rank_header(stage: str, epoch: int):
    # global/local rank + node + GPU id
    global_rank = _rank()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    node_rank = int(os.environ.get("NODE_RANK", -1))
    try:
        gpu_id = torch.cuda.current_device()
        if not torch.cuda.is_available():
            gpu_id = -1
    except Exception:
        gpu_id = -1
    header = (f"[GRank {global_rank} | LRank {local_rank} | "
              f"Node {node_rank} | GPU {gpu_id}] "
              f"=== {stage} EPOCH {epoch} START ===")
    print(header)


class AdvanceSamplerEpoch(pl.Callback):
    """Ensure random samplers reshuffle each epoch."""
    def on_train_epoch_start(self, trainer, pl_module):
        try:
            # Try to grab the actual dataloader object Lightning is iterating
            dl = None
            if (hasattr(trainer.fit_loop, "epoch_loop") and
                    hasattr(trainer.fit_loop.epoch_loop, "_loader")):
                dl = getattr(
                    trainer.fit_loop.epoch_loop._loader, "_dataloader", None
                )
            if dl is None and hasattr(trainer.fit_loop, "_combined_loader"):
                dl = getattr(
                    trainer.fit_loop._combined_loader, "_loader", None
                )

            if dl is None:
                # Can't find a dataloader – nothing to do this epoch
                return

            dls = dl if isinstance(dl, (list, tuple)) else [dl]
            for d in dls:
                s = getattr(d, "sampler", None)
                if hasattr(s, "set_epoch"):
                    s.set_epoch(pl_module.current_epoch)

        except Exception as e:
            # Be robust to Lightning internal changes. Only warn on rank 0.
            if _is_rank0():
                print(
                    f"[AdvanceSamplerEpoch] WARNING: "
                    f"Could not advance sampler epoch due to: {e}"
                )
            # Fail soft: training continues, but sampler epoch may stay unchanged.
            return

# -----------------------------
# Train resampling per epoch (SYNCED)
# -----------------------------


class ResampleDataCallback(pl.Callback):
    """
    Resample TRAIN (and optionally VAL) windows from their respective
    date ranges, with synchronized windows across all ranks.

    - Updates happen at **epoch END** so that Lightning's
      `reload_dataloaders_every_n_epochs=1` picks them up for the next
      epoch.
    - By default, VALIDATION IS FIXED. Set `resample_val=True` to roll
      validation too.
    """

    def __init__(
        self,
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        train_window_days: int = 14,
        val_window_days: int = 3,
        mode: str = "random",          # "random" or "sequential"
        resample_val: bool = False,
        seq_stride_days: int = 1,
    ):
        # Training pool
        self.train_start_date = pd.to_datetime(train_start_date)
        self.train_end_date = pd.to_datetime(train_end_date)
        self.train_window = pd.Timedelta(days=train_window_days)

        # Validation pool
        self.val_start_date = pd.to_datetime(val_start_date)
        self.val_end_date = pd.to_datetime(val_end_date)
        self.val_window = pd.Timedelta(days=val_window_days)

        self.mode = mode.lower()
        assert self.mode in {"random", "sequential"}
        self.resample_val = bool(resample_val)
        self.seq_stride = pd.Timedelta(days=seq_stride_days)
        self._state_loaded_from_checkpoint: bool = False  # Explicit flag for resume detection

        self._check_date_ranges()

    # ---- checkpoint persistence -----------------------
    def state_dict(self):
        return {
            # Also save the current train window so we can restore it on resume
            "_saved_train_start": getattr(self, "_saved_train_start", None),
            "_saved_train_end": getattr(self, "_saved_train_end", None),
        }

    def load_state_dict(self, state):
        # Restore the saved train window
        train_start = state.get("_saved_train_start", None)
        if train_start is not None:
            self._saved_train_start = pd.to_datetime(train_start)

        train_end = state.get("_saved_train_end", None)
        if train_end is not None:
            self._saved_train_end = pd.to_datetime(train_end)

        # Mark that state was explicitly restored from checkpoint
        self._state_loaded_from_checkpoint = True
        if _is_rank0():
            print(
                f"[ResampleDataCallback] Restored from checkpoint: "
                f"train_window={self._saved_train_start.date() if self._saved_train_start else None} .. "
                f"{self._saved_train_end.date() if self._saved_train_end else None}"
            )

    # --- helpers ----------------------------------------------------------
    def _check_date_ranges(self):
        if self.train_end_date < self.train_start_date:
            raise ValueError(
                f"[ResampleDataCallback] Training range invalid: "
                f"{self.train_start_date.date()} .. "
                f"{self.train_end_date.date()}"
            )
        if self.val_end_date < self.val_start_date:
            raise ValueError(
                f"[ResampleDataCallback] Validation range invalid: "
                f"{self.val_start_date.date()} .. "
                f"{self.val_end_date.date()}"
            )
        if self.train_window <= pd.Timedelta(0):
            raise ValueError("train_window_days must be > 0")
        if self.val_window <= pd.Timedelta(0):
            raise ValueError("val_window_days must be > 0")

    @staticmethod
    def _clip_end(
        new_end: pd.Timestamp, end_limit: pd.Timestamp
    ) -> pd.Timestamp:
        return min(new_end, end_limit)

    @staticmethod
    def _rand_offset(total_days: int, window_days: int) -> int:
        max_off = max(0, total_days - window_days)
        return random.randint(0, max_off) if max_off > 0 else 0

    def _update_datamodule(
        self, trainer, start_dt, end_dt, is_train: bool
    ):
        dm = trainer.datamodule
        if is_train:
            if _is_rank0():
                print(
                    f"[DM.set_train_window] -> {start_dt} .. {end_dt}"
                )
            dm.set_train_window(start_dt, end_dt)
        else:
            if _is_rank0():
                print(
                    f"[DM.set_val_window]   -> {start_dt} .. {end_dt}"
                )
            dm.set_val_window(start_dt, end_dt)

    # --- epoch hooks ------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        """
        If resuming from checkpoint, restore the train window to where it was.
        """
        dm = trainer.datamodule

        if self._state_loaded_from_checkpoint and hasattr(self, "_saved_train_start"):
            # Restore to the saved window from checkpoint
            start = self._saved_train_start
            end = self._saved_train_end
            self._update_datamodule(trainer, start, end, is_train=True)
            if _is_rank0():
                print(
                    f"[ResampleDataCallback] on_fit_start: Restored train window "
                    f"from checkpoint to {start.date()} .. {end.date()}"
                )
            return

        # Fresh training run: save the initial window
        self._saved_train_start = pd.to_datetime(dm.hparams.train_start)
        self._saved_train_end = pd.to_datetime(dm.hparams.train_end)
        if _is_rank0():
            print(
                f"[ResampleDataCallback] on_fit_start: Fresh training run, "
                f"initialized to {self._saved_train_start.date()} .. {self._saved_train_end.date()}"
            )

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            log_rank_header("TRAIN", pl_module.current_epoch)
            print(
                f"[Train pool] {self.train_start_date.date()} .. "
                f"{self.train_end_date.date()} "
                f"win={self.train_window} mode={self.mode}"
            )
            dm = trainer.datamodule
            if self.mode == "sequential":
                num_bins = len(dm.train_bin_names)
                print(
                    f"[Train] Sequential mode: Bins processed in "
                    f"chronological order (bin1→bin{num_bins})"
                )
            else:
                print(
                    "[Train] Random mode: Bins shuffled randomly "
                    "each epoch"
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            log_rank_header("VAL", pl_module.current_epoch)
            print(
                f"[Val pool]   {self.val_start_date.date()} .. "
                f"{self.val_end_date.date()} "
                f"win={self.val_window} resample_val={self.resample_val}"
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Choose NEXT epoch's TRAIN window on rank-0 **based on the window
        actually used this epoch** (prevents epoch 0/1 duplication),
        broadcast, then set on all ranks.

        IMPORTANT: Save the current window before updating to next, so that
        when we resume from this checkpoint, we can restore the correct window.
        """
        dm = trainer.datamodule

        # Save the current train window (needed for checkpoint resumption)
        self._saved_train_start = pd.to_datetime(dm.hparams.train_start)
        self._saved_train_end = pd.to_datetime(dm.hparams.train_end)

        used_start = pd.to_datetime(dm.hparams.train_start)

        if _is_rank0():
            if self.mode == "sequential":
                start = used_start + self.seq_stride
                if start >= self.train_end_date:
                    print(
                        "[Sequential Train] Reached end; "
                        "looping back to start."
                    )
                    start = self.train_start_date
                end = self._clip_end(
                    start + self.train_window, self.train_end_date
                )
            else:
                total_days = (
                    self.train_end_date - self.train_start_date
                ).days
                offset = self._rand_offset(
                    total_days, self.train_window.days
                )
                start = self.train_start_date + pd.Timedelta(days=offset)
                end = self._clip_end(
                    start + self.train_window, self.train_end_date
                )
        else:
            # dummy; overwritten by broadcast
            start = self.train_start_date
            end = self._clip_end(
                start + self.train_window, self.train_end_date
            )

        start, end = _broadcast_window(start, end)
        print(
            f"[Rank {_rank()}] [Resample {self.mode.upper()}] "
            f"NEXT (sync) -> {start.date()} .. {end.date()}"
        )
        self._update_datamodule(trainer, start, end, is_train=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        If enabled, resample VAL on rank-0, broadcast, then set
        everywhere.
        """
        if not self.resample_val:
            if trainer.is_global_zero:
                print(
                    "[Sampler - VAL] Fixed validation slice "
                    "(no resampling)."
                )
            return

        total_days = (self.val_end_date - self.val_start_date).days
        if _is_rank0():
            offset = self._rand_offset(total_days, self.val_window.days)
            start = self.val_start_date + pd.Timedelta(days=offset)
            end = self._clip_end(start + self.val_window, self.val_end_date)
        else:
            start = self.val_start_date
            end = self._clip_end(
                start + self.val_window, self.val_end_date
            )

        start, end = _broadcast_window(start, end)
        print(
            f"[Rank {_rank()}] [Resample VAL] "
            f"NEXT (sync) -> {start.date()} .. {end.date()}"
        )
        self._update_datamodule(trainer, start, end, is_train=False)


# -----------------------------------
# Sequential (sliding) train windows with synchronized sampling
# -----------------------------------


class SequentialDataCallback(pl.Callback):
    """
    Sequential or random training with synchronized windows across all
    ranks.

    At epoch end:
      1) rank-0 chooses the next window (based on the window actually
         used)
      2) window is broadcast to all ranks
      3) all ranks call set_train_window() with the same window
      4) on next epoch start, Lightning reloads DataLoaders and all ranks
         use the same dataset view
    """

    def __init__(
        self,
        full_start_date,
        full_end_date,
        window_days: int = 7,
        stride_days: int = 1,
        mode: str = "sequential",
        wrap_sequential: bool = True,
    ):
        self.full_start_date = pd.to_datetime(full_start_date)
        self.full_end_date = pd.to_datetime(full_end_date)
        self.window = pd.Timedelta(days=window_days)
        self.stride = pd.Timedelta(days=stride_days)
        self.mode = mode.lower()
        self.wrap_sequential = wrap_sequential

        if self.full_end_date <= self.full_start_date:
            raise ValueError(
                f"[Sequential Sampler] range invalid: "
                f"{self.full_start_date.date()} .. {self.full_end_date.date()}"
            )
        if self.mode not in {"sequential", "random"}:
            raise ValueError("mode must be 'sequential' or 'random'")

        self.total_days = (self.full_end_date - self.full_start_date).days
        self.max_offset_days = max(0, self.total_days - window_days)

        # track actual window used (for correctness on restarts)
        self.current_start: pd.Timestamp = self.full_start_date
        self._last_used_start: Optional[pd.Timestamp] = None
        self._state_loaded_from_checkpoint: bool = False  # Explicit flag for resume detection

    # ---- checkpoint persistence ------------------------------------------
    def state_dict(self):
        return {
            "current_start": self.current_start,
            "_last_used_start": self._last_used_start
        }

    def load_state_dict(self, state):
        cur = state.get("current_start", None)
        if cur is not None:
            self.current_start = pd.to_datetime(cur)
        used = state.get("_last_used_start", None)
        if used is not None:
            self._last_used_start = pd.to_datetime(used)
        # Mark that state was explicitly restored from checkpoint
        self._state_loaded_from_checkpoint = True
        if _is_rank0():
            print(
                f"[Sequential CALLBACK] Restored from checkpoint: "
                f"current_start={self.current_start.date()}, "
                f"last_used={self._last_used_start.date() if self._last_used_start else None}"
            )

    # ---- helpers ---------------------------------------------------------
    def _choose_next_window_rank0(self, used_start: pd.Timestamp):
        if self.mode == "sequential":
            start = used_start + self.stride
            if start >= self.full_end_date:
                if self.wrap_sequential:
                    print(
                        "[Rank 0] [Sequential] Reached end; "
                        "wrapping to start."
                    )
                    start = self.full_start_date
                else:
                    print(
                        "[Rank 0] [Sequential] Reached end; "
                        "clamping at tail."
                    )
                    start = max(
                        self.full_start_date,
                        self.full_end_date - self.window
                    )
            end = min(start + self.window, self.full_end_date)
            self.current_start = start
        else:
            offset_days = (
                random.randint(0, self.max_offset_days)
                if self.max_offset_days > 0 else 0
            )
            start = self.full_start_date + pd.Timedelta(days=offset_days)
            end = min(start + self.window, self.full_end_date)
        return start, end

    # ---- hooks -----------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        dm = trainer.datamodule

        # If restoring from checkpoint, use the restored state.
        # The _state_loaded_from_checkpoint flag is set explicitly in load_state_dict()
        if self._state_loaded_from_checkpoint:
            # Use the callback's current_start (restored from checkpoint) as the window start
            start = self.current_start
            end = min(start + self.window, self.full_end_date)

            # Align DM to the restored window
            dm.set_train_window(start, end)

            if trainer.is_global_zero:
                print(
                    f"[Sequential INIT] Resumed from checkpoint: aligning DM to "
                    f"{start.date()} .. {end.date()}"
                )
            return

        # Fresh training run (no checkpoint loaded)
        # Initialize from the DataModule's default window
        self._last_used_start = pd.to_datetime(dm.hparams.train_start)
        self.current_start = self._last_used_start
        if trainer.is_global_zero:
            print(
                f"[Sequential INIT] Fresh training run: initialized to "
                f"{self._last_used_start.date()}"
            )

    def on_train_epoch_start(self, trainer, pl_module):
        # Record the actual window used THIS epoch from the DM
        dm = trainer.datamodule
        self._last_used_start = pd.to_datetime(dm.hparams.train_start)
        if trainer.is_global_zero:
            train_start = pd.to_datetime(dm.hparams.train_start).date()
            train_end = pd.to_datetime(dm.hparams.train_end).date()
            num_bins = len(dm.train_bin_names)
            print(
                f"\n[Sequential {self.mode.upper()}] "
                f"CURRENT EPOCH {pl_module.current_epoch} -> "
                f"{train_start} .. {train_end}"
            )
            print(
                f"[Sequential] Bins will be processed in "
                f"chronological order (bin1→bin{num_bins})"
            )

    def on_train_epoch_end(self, trainer, pl_module):
        # Rank-0 picks next window based on the window actually used,
        # broadcast, then set on all ranks
        if _is_rank0():
            used_start = (
                self._last_used_start or
                pd.to_datetime(trainer.datamodule.hparams.train_start)
            )
            start, end = self._choose_next_window_rank0(used_start)
            print(
                f"[Rank 0] [Sequential {self.mode.upper()}] "
                f"NEXT -> {start.date()} .. {end.date()}"
            )
        else:
            start = self.full_start_date
            end = min(start + self.window, self.full_end_date)

        start, end = _broadcast_window(start, end)
        print(
            f"[Rank {_rank()}] [Sequential {self.mode.upper()}] "
            f"NEXT (sync) -> {start.date()} .. {end.date()}"
        )
        trainer.datamodule.set_train_window(start, end)
