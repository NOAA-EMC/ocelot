import os
import random
import pandas as pd
import torch
import lightning.pytorch as pl
from typing import Optional


# -----------------------------
# Train resampling per epoch
# -----------------------------
class ResampleDataCallback(pl.Callback):
    """
    Resample TRAIN (and optionally VAL) windows from their respective date ranges.

    - Updates happen at **epoch END** so that PL's `reload_dataloaders_every_n_epochs=1`
      will pick them up at the start of the next epoch.
    - By default, VALIDATION IS FIXED (no resampling). Set `resample_val=True` to enable
      rolling,validation windows.

    Args
    ----
    train_start_date, train_end_date : str | datetime-like
        Inclusive bounds of the training date range.
    val_start_date, val_end_date : str | datetime-like
        Inclusive bounds of the validation date range (used to build the *fixed* val set
        when `resample_val=False`, or the sampling pool when `resample_val=True`).
    train_window_days : int
        Length of each training window in days.
    val_window_days : int
        Length of each validation window in days (only used if `resample_val=True`).
    mode : {"random","sequential"}
        How to choose the next TRAIN window. "sequential" advances by `seq_stride_days`
        (default 1 day). Validation (if enabled) always samples randomly.
    resample_val : bool
        If False (default), validation stays fixed (best practice for checkpointing/ES).
        If True, validation is re-sampled at epoch end (higher variance metric).
    seq_stride_days : int
        Stride (days) for sequential train windows when `mode="sequential"`.
    """

    def __init__(
        self,
        train_start_date,
        train_end_date,
        val_start_date,
        val_end_date,
        train_window_days: int = 14,
        val_window_days: int = 3,
        mode: str = "random",
        resample_val: bool = False,
        seq_stride_days: int = 1,
    ):
        # Training date range
        self.train_start_date = pd.to_datetime(train_start_date)
        self.train_end_date = pd.to_datetime(train_end_date)
        self.train_window = pd.Timedelta(days=train_window_days)

        # Validation date range (pool or fixed slice)
        self.val_start_date = pd.to_datetime(val_start_date)
        self.val_end_date = pd.to_datetime(val_end_date)
        self.val_window = pd.Timedelta(days=val_window_days)

        self.mode = mode.lower()
        assert self.mode in {"random", "sequential"}, "mode must be 'random' or 'sequential'"
        self.resample_val = bool(resample_val)
        self.seq_stride = pd.Timedelta(days=seq_stride_days)
        self._seq_cursor: Optional[pd.Timestamp] = None  # for sequential train mode

        self._check_date_ranges()

    # --- helpers -------------------------------------------------------------

    def _check_date_ranges(self):
        if self.train_end_date < self.train_start_date:
            raise ValueError(
                f"[ResampleDataCallback] Training range invalid: "
                f"{self.train_start_date.date()} .. {self.train_end_date.date()}"
            )
        if self.val_end_date < self.val_start_date:
            raise ValueError(
                f"[ResampleDataCallback] Validation range invalid: "
                f"{self.val_start_date.date()} .. {self.val_end_date.date()}"
            )
        if self.train_window <= pd.Timedelta(0):
            raise ValueError("train_window_days must be > 0")
        if self.val_window <= pd.Timedelta(0):
            raise ValueError("val_window_days must be > 0")

    @staticmethod
    def _clip_end(new_end: pd.Timestamp, end_limit: pd.Timestamp) -> pd.Timestamp:
        return min(new_end, end_limit)

    @staticmethod
    def _rand_offset(total_days: int, window_days: int) -> int:
        max_off = max(0, total_days - window_days)
        return random.randint(0, max_off) if max_off > 0 else 0

    def _update_datamodule(self, trainer, start_dt, end_dt, is_train: bool):
        dm = trainer.datamodule
        if is_train:
            print(f"[DM.set_train_window] -> {start_dt} .. {end_dt}")
            dm.set_train_window(start_dt, end_dt)
        else:
            print(f"[DM.set_val_window]   -> {start_dt} .. {end_dt}")
            dm.set_val_window(start_dt, end_dt)
        # PL will rebuild loaders at next epoch boundary when
        # `reload_dataloaders_every_n_epochs=1` is set.

    # --- epoch hooks ---------------------------------------------------------

    def on_train_epoch_start(self, trainer, pl_module):
        rank = int(os.environ.get("RANK", "0"))
        if getattr(trainer, "is_global_zero", True):
            print(f"\n=== TRAIN EPOCH {pl_module.current_epoch} START ===")
        print(f"[Rank {rank}] train range: {self.train_start_date.date()} .. {self.train_end_date.date()} "
              f"win={self.train_window}")

    def on_validation_epoch_start(self, trainer, pl_module):
        rank = int(os.environ.get("RANK", "0"))
        if getattr(trainer, "is_global_zero", True):
            print(f"\n=== VAL   EPOCH {pl_module.current_epoch} START ===")
        print(f"[Rank {rank}] val range:   {self.val_start_date.date()} .. {self.val_end_date.date()} "
              f"win={self.val_window} resample_val={self.resample_val}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Prepare NEXT epoch's TRAIN window."""
        total_days = (self.train_end_date - self.train_start_date).days
        if self.mode == "sequential":
            if self._seq_cursor is None:
                self._seq_cursor = self.train_start_date
            start = self._seq_cursor
            end = self._clip_end(start + self.train_window, self.train_end_date)
            # advance cursor for next epoch (wrap if needed)
            next_start = start + self.seq_stride
            if next_start >= self.train_end_date:
                print("[Sequential Train] Reached end; looping back to start.")
                next_start = self.train_start_date
            self._seq_cursor = next_start
        else:
            # random
            offset = self._rand_offset(total_days, self.train_window.days)
            start = self.train_start_date + pd.Timedelta(days=offset)
            end = self._clip_end(start + self.train_window, self.train_end_date)

        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] [Sampler - TRAIN] Next -> {start.date()} .. {end.date()}")
        self._update_datamodule(trainer, start, end, is_train=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Prepare NEXT epoch's VAL window (only if resample_val=True)."""
        if not self.resample_val:
            if getattr(trainer, "is_global_zero", True):
                print("[Sampler - VAL] Fixed validation slice (no resampling).")
            return

        total_days = (self.val_end_date - self.val_start_date).days
        offset = self._rand_offset(total_days, self.val_window.days)
        start = self.val_start_date + pd.Timedelta(days=offset)
        end = self._clip_end(start + self.val_window, self.val_end_date)

        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] [Sampler - VAL]   Next -> {start.date()} .. {end.date()}")
        self._update_datamodule(trainer, start, end, is_train=False)


# -----------------------------------
# Sequential (sliding) train windows
# -----------------------------------
class SequentialDataCallback(pl.Callback):
    def __init__(self, full_start_date, full_end_date, window_days: int = 7, stride_days: int = 1):
        self.full_start_date = pd.to_datetime(full_start_date)
        self.full_end_date = pd.to_datetime(full_end_date)
        self.window = pd.Timedelta(days=window_days)
        self.stride = pd.Timedelta(days=stride_days)
        self.current_start = self.full_start_date

        if self.full_end_date <= self.full_start_date:
            raise ValueError(
                f"[Sequential Sampler] range invalid: {self.full_start_date.date()} .. {self.full_end_date.date()}"
            )

    def _window_for_current(self):
        start = self.current_start
        end = min(start + self.window, self.full_end_date)
        return start, end

    def on_train_epoch_start(self, trainer, pl_module):
        # Purely informational logging; do NOT set here.
        start, end = self._window_for_current()
        rank = int(os.environ.get("RANK", "0"))
        print(f"\n[Rank {rank}] [Sequential Train] CURRENT -> {start.date()} .. {end.date()}")

    def on_train_epoch_end(self, trainer, pl_module):
        # Prepare NEXT epoch’s window so PL reload uses it
        start, end = self._window_for_current()
        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] [Sequential Train] NEXT -> {start.date()} .. {end.date()}")
        trainer.datamodule.set_train_window(start, end)

        # advance cursor
        self.current_start += self.stride
        if self.current_start >= self.full_end_date:
            print("[Sequential Train] Reached end of range; looping back.")
            self.current_start = self.full_start_date
