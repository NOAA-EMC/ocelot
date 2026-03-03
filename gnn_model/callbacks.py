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
        switch_to_sequential_after_epochs: Optional[int] = None,
        auto_switch_to_sequential: bool = False,
        auto_switch_metric: str = "val_loss",
        auto_switch_patience_epochs: int = 10,
        auto_switch_min_delta: float = 0.0,
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
        self._seq_start_epoch: Optional[int] = None      # epoch index when sequential started

        # Optional schedule: start random, then switch to sequential after N epochs.
        # Epochs are 0-based in Lightning; if N=50, epochs [0..49] random and epoch 50 starts sequential.
        if switch_to_sequential_after_epochs is not None:
            n = int(switch_to_sequential_after_epochs)
            self.switch_to_sequential_after_epochs = n if n > 0 else None
        else:
            self.switch_to_sequential_after_epochs = None

        # Optional auto-switch: switch to sequential when a metric plateaus.
        self.auto_switch_to_sequential = bool(auto_switch_to_sequential)
        self.auto_switch_metric = str(auto_switch_metric)
        self.auto_switch_patience_epochs = max(1, int(auto_switch_patience_epochs))
        self.auto_switch_min_delta = float(auto_switch_min_delta)

        # Auto-switch state (EarlyStopping-like)
        self._auto_best = float("inf")
        self._auto_wait = 0
        self._auto_switched = False

        self._check_date_ranges()

        # resume helpers
        self._restored_from_ckpt = False
        self._loaded_epoch: Optional[int] = None
        self._saved_seq_stride_days: Optional[int] = None

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Lightning may call on_fit_start before trainer.current_epoch is restored.
        # Keep a copy of the checkpoint epoch so we can align windows correctly.
        try:
            e = checkpoint.get("epoch", None)
            self._loaded_epoch = None if e is None else int(e)
        except Exception:
            self._loaded_epoch = None

    def _resume_epoch(self, trainer) -> int:
        e = int(getattr(trainer, "current_epoch", 0) or 0)
        if e == 0 and (self._loaded_epoch is not None) and self._loaded_epoch > 0:
            return int(self._loaded_epoch)
        return e

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

    @staticmethod
    def _to_float(x) -> Optional[float]:
        try:
            if x is None:
                return None
            if torch.is_tensor(x):
                return float(x.detach().float().cpu().item())
            return float(x)
        except Exception:
            return None

    def _infer_seq_cursor_from_epoch(self, epoch: int) -> pd.Timestamp:
        """Infer a deterministic sequential cursor from an epoch index (0-based)."""
        inferred = self.train_start_date + epoch * self.seq_stride
        if inferred >= self.train_end_date:
            span_days = max(1, int((self.train_end_date - self.train_start_date).days))
            stride_days = max(1, int(self.seq_stride / pd.Timedelta(days=1)))
            off = (epoch * stride_days) % span_days
            inferred = self.train_start_date + pd.Timedelta(days=off)
        return inferred

    def _infer_seq_cursor_from_epoch_with_restart(self, epoch: int) -> pd.Timestamp:
        """Infer cursor when sequential windows are defined to RESTART at train_start_date.

        If sequential starts at epoch = E0, then epoch E0 uses train_start_date,
        epoch E0+1 uses train_start_date + stride, etc.
        """
        if self._seq_start_epoch is None:
            # Fallback to the legacy inference.
            return self._infer_seq_cursor_from_epoch(epoch)

        e0 = int(self._seq_start_epoch)
        if epoch < e0:
            return self.train_start_date

        k = epoch - e0
        inferred = self.train_start_date + k * self.seq_stride
        if inferred >= self.train_end_date:
            span_days = max(1, int((self.train_end_date - self.train_start_date).days))
            stride_days = max(1, int(self.seq_stride / pd.Timedelta(days=1)))
            off = (k * stride_days) % span_days
            inferred = self.train_start_date + pd.Timedelta(days=off)
        return inferred

    def _state_dict_legacy(self):
        return {}

    def _switch_to_sequential_for_next_epoch(self, trainer, next_epoch: int, reason: str):
        if self.mode == "sequential" or self._auto_switched:
            return

        self.mode = "sequential"
        self._auto_switched = True
        # Desired behavior: when switching from random->sequential, restart at train_start_date.
        self._seq_start_epoch = int(next_epoch)
        self._seq_cursor = self.train_start_date

        start = self._seq_cursor
        end = self._clip_end(start + self.train_window, self.train_end_date)

        if getattr(trainer, "is_global_zero", True):
            print(
                f"[ResampleDataCallback] Auto-switch TRAIN sampler -> SEQUENTIAL at epoch={next_epoch} "
                f"({reason}); next_window={start.date()}..{end.date()} stride={self.seq_stride} (restart_at_start_date)"
            )

        # on_validation_epoch_end runs after on_train_epoch_end in the same epoch,
        # so override the already-chosen next train window.
        trainer.datamodule.set_train_window(start, end)

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
            try:
                if getattr(trainer, "optimizers", None):
                    lr = float(trainer.optimizers[0].param_groups[0].get("lr", float("nan")))
                    print(f"[LR] current={lr:.6g}")
            except Exception:
                pass
        print(f"[Rank {rank}] train range: {self.train_start_date.date()} .. {self.train_end_date.date()} "
              f"win={self.train_window}")

    def on_fit_start(self, trainer, pl_module):
        """Align sequential cursor on resume.

        If we are (or will be) in sequential mode and no cursor was restored from checkpoint,
        infer cursor from the current epoch counter so resuming doesn't restart from the beginning.
        """
        epoch = self._resume_epoch(trainer)

        # Safety: changing stride_days across resume changes the epoch->date mapping.
        # Example: epoch 212 with stride=12 is ~2021-12-19, but with stride=1 it's 2015-08-01.
        if self._restored_from_ckpt and self._saved_seq_stride_days is not None:
            current_stride_days = max(1, int(self.seq_stride / pd.Timedelta(days=1)))
            saved_stride_days = int(self._saved_seq_stride_days)
            if current_stride_days != saved_stride_days:
                allow = os.environ.get("ALLOW_STRIDE_CHANGE_ON_RESUME", "0").lower() in {"1", "true", "yes"}
                if getattr(trainer, "is_global_zero", True):
                    print(
                        f"[ResampleDataCallback][WARN] stride_days mismatch on resume: "
                        f"checkpoint={saved_stride_days} current={current_stride_days}. "
                        f"This will change sequential train windows. "
                        f"Set ALLOW_STRIDE_CHANGE_ON_RESUME=1 to keep the current stride."
                    )
                if not allow:
                    # Preserve experiment continuity by honoring the checkpoint stride.
                    self.seq_stride = pd.Timedelta(days=saved_stride_days)
                    if getattr(trainer, "is_global_zero", True):
                        print(
                            f"[ResampleDataCallback][WARN] Overriding current stride_days -> {saved_stride_days} "
                            f"to match checkpoint and keep window continuity."
                        )

        # If we are at/after the configured switch epoch, ensure sequential mode immediately.
        if self.switch_to_sequential_after_epochs is not None and epoch >= self.switch_to_sequential_after_epochs:
            self.mode = "sequential"
            # IMPORTANT: Do NOT force seq_start_epoch here.
            # - Existing runs created before restart-at-start-date semantics relied on legacy
            #   cursor inference (train_start_date + epoch*stride), which matches your observed
            #   v159 window around 2020-03-23.
            # - New runs that want restart semantics will have seq_start_epoch saved in the
            #   checkpoint (or will set it during the actual switch in on_train_epoch_end).

        if self.mode == "sequential":
            # Ignore any saved seq_cursor that may be stale/corrupted; compute from epoch.
            start = self._infer_seq_cursor_from_epoch_with_restart(epoch)
            end = self._clip_end(start + self.train_window, self.train_end_date)
            trainer.datamodule.set_train_window(start, end)

            # Precompute NEXT epoch's start for the epoch-end scheduler.
            self._seq_cursor = self._infer_seq_cursor_from_epoch_with_restart(epoch + 1)

            if getattr(trainer, "is_global_zero", True):
                print(
                    f"[ResampleDataCallback] RESUME align (epoch={epoch}) mode=sequential "
                    f"current_window={start.date()}..{end.date()} next_cursor={self._seq_cursor.date()}"
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        rank = int(os.environ.get("RANK", "0"))
        if getattr(trainer, "is_global_zero", True):
            print(f"\n=== VAL   EPOCH {pl_module.current_epoch} START ===")
        print(f"[Rank {rank}] val range:   {self.val_start_date.date()} .. {self.val_end_date.date()} "
              f"win={self.val_window} resample_val={self.resample_val}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Prepare NEXT epoch's TRAIN window."""
        # Optional switch: random warmup -> sequential
        next_epoch = int(pl_module.current_epoch) + 1
        if (
            self.switch_to_sequential_after_epochs is not None
            and next_epoch >= self.switch_to_sequential_after_epochs
            and self.mode != "sequential"
        ):
            self.mode = "sequential"
            # Restart sequential coverage at train_start_date.
            self._seq_start_epoch = int(next_epoch)
            self._seq_cursor = self.train_start_date
            if getattr(trainer, "is_global_zero", True):
                print(
                    f"[ResampleDataCallback] Switching TRAIN sampler to SEQUENTIAL at epoch={next_epoch}; "
                    f"seq_cursor={self._seq_cursor.date()} stride={self.seq_stride} (restart_at_start_date)"
                )

        total_days = (self.train_end_date - self.train_start_date).days
        if self.mode == "sequential":
            # Deterministic: next_epoch window depends only on epoch index.
            next_epoch = int(pl_module.current_epoch) + 1
            start = self._infer_seq_cursor_from_epoch_with_restart(next_epoch)
            end = self._clip_end(start + self.train_window, self.train_end_date)
            self._seq_cursor = self._infer_seq_cursor_from_epoch_with_restart(next_epoch + 1)
        else:
            offset = self._rand_offset(total_days, self.train_window.days)
            start = self.train_start_date + pd.Timedelta(days=offset)
            end = self._clip_end(start + self.train_window, self.train_end_date)

        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] [Sampler - TRAIN] Next -> {start.date()} .. {end.date()}")
        self._update_datamodule(trainer, start, end, is_train=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Prepare NEXT epoch's VAL window (only if resample_val=True)."""
        # Optional auto-switch to sequential based on plateau
        if self.auto_switch_to_sequential and self.mode != "sequential" and not self._auto_switched:
            metric_val = self._to_float(trainer.callback_metrics.get(self.auto_switch_metric))
            if metric_val is not None:
                improved = metric_val < (self._auto_best - self.auto_switch_min_delta)
                if improved:
                    self._auto_best = metric_val
                    self._auto_wait = 0
                else:
                    self._auto_wait += 1

                if getattr(trainer, "is_global_zero", True):
                    print(
                        f"[ResampleDataCallback] AutoSwitch monitor={self.auto_switch_metric} "
                        f"val={metric_val:.6f} best={self._auto_best:.6f} wait={self._auto_wait}/"
                        f"{self.auto_switch_patience_epochs} min_delta={self.auto_switch_min_delta}"
                    )

                if self._auto_wait >= self.auto_switch_patience_epochs:
                    next_epoch = int(pl_module.current_epoch) + 1
                    self._switch_to_sequential_for_next_epoch(
                        trainer,
                        next_epoch=next_epoch,
                        reason=f"plateau({self.auto_switch_metric}) for {self._auto_wait} epochs",
                    )

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

    # --- checkpointing support (resume) ---
    def state_dict(self):
        return {
            "mode": self.mode,
            # Cursor for sequential train windows.
            "seq_cursor": None if self._seq_cursor is None else str(self._seq_cursor),
            # Epoch index when sequential started (used for restart-at-start-date semantics).
            "seq_start_epoch": self._seq_start_epoch,
            # Config needed to preserve epoch->date mapping across resume.
            "seq_stride_days": max(1, int(self.seq_stride / pd.Timedelta(days=1))),
            # config
            "switch_to_sequential_after_epochs": self.switch_to_sequential_after_epochs,
            "auto_switch_to_sequential": self.auto_switch_to_sequential,
            "auto_switch_metric": self.auto_switch_metric,
            "auto_switch_patience_epochs": self.auto_switch_patience_epochs,
            "auto_switch_min_delta": self.auto_switch_min_delta,
            # state
            "auto_best": self._auto_best,
            "auto_wait": self._auto_wait,
            "auto_switched": self._auto_switched,
        }

    def load_state_dict(self, state_dict):
        try:
            self.mode = str(state_dict.get("mode", self.mode)).lower()
            v = state_dict.get("seq_cursor", None)
            self._seq_cursor = None if v in (None, "None") else pd.to_datetime(v)
            self._seq_start_epoch = state_dict.get("seq_start_epoch", self._seq_start_epoch)
            saved = state_dict.get("seq_stride_days", None)
            self._saved_seq_stride_days = None if saved in (None, "None") else int(saved)
            self.switch_to_sequential_after_epochs = state_dict.get(
                "switch_to_sequential_after_epochs", self.switch_to_sequential_after_epochs
            )
            self._auto_best = float(state_dict.get("auto_best", self._auto_best))
            self._auto_wait = int(state_dict.get("auto_wait", self._auto_wait))
            self._auto_switched = bool(state_dict.get("auto_switched", self._auto_switched))
            # config (keep defaults if older checkpoints don't have these)
            self.auto_switch_to_sequential = bool(state_dict.get("auto_switch_to_sequential", self.auto_switch_to_sequential))
            self.auto_switch_metric = str(state_dict.get("auto_switch_metric", self.auto_switch_metric))
            self.auto_switch_patience_epochs = int(state_dict.get("auto_switch_patience_epochs", self.auto_switch_patience_epochs))
            self.auto_switch_min_delta = float(state_dict.get("auto_switch_min_delta", self.auto_switch_min_delta))
            self._restored_from_ckpt = True
        except Exception:
            self._restored_from_ckpt = False


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
        self._restored_from_ckpt = False

        if self.full_end_date <= self.full_start_date:
            raise ValueError(
                f"[Sequential Sampler] range invalid: {self.full_start_date.date()} .. {self.full_end_date.date()}"
            )

    def _window_for_current(self):
        start = self.current_start
        end = min(start + self.window, self.full_end_date)
        return start, end

    @staticmethod
    def _rank_str() -> str:
        return os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"

    def on_fit_start(self, trainer, pl_module):
        """Ensure cursor is aligned on resume even for older checkpoints.

        If the checkpoint was created before this callback had state_dict/load_state_dict,
        `current_start` will reset to full_start_date. In that case, infer the cursor from
        the restored epoch counter.
        """
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)

        if (not self._restored_from_ckpt) and epoch > 0:
            inferred = self.full_start_date + epoch * self.stride

            # wrap if needed
            if inferred >= self.full_end_date:
                span_days = max(1, int((self.full_end_date - self.full_start_date).days))
                off = (epoch * int(self.stride / pd.Timedelta(days=1))) % span_days
                inferred = self.full_start_date + pd.Timedelta(days=off)

            self.current_start = inferred
            if getattr(trainer, "is_global_zero", True):
                print(
                    f"[Sequential Train] Checkpoint had no cursor; inferring from epoch={epoch}: "
                    f"current_start={self.current_start.date()}"
                )

        # Set the datamodule window immediately for the (resumed) current epoch.
        # This avoids starting from 2015-01-01 when resuming from older checkpoints.
        start, end = self._window_for_current()
        trainer.datamodule.set_train_window(start, end)
        if getattr(trainer, "is_global_zero", True):
            print(f"[Sequential Train] SET  -> {start.date()} .. {end.date()} (epoch={epoch})")

    def on_train_epoch_start(self, trainer, pl_module):
        # Purely informational logging; do NOT set here.
        start, end = self._window_for_current()
        rank = self._rank_str()
        if getattr(trainer, "is_global_zero", True):
            print(f"\n[Rank {rank}] [Sequential Train] CURRENT -> {start.date()} .. {end.date()}")

    def on_train_epoch_end(self, trainer, pl_module):
        # advance cursor for next epoch
        self.current_start += self.stride
        if self.current_start >= self.full_end_date:
            print("[Sequential Train] Reached end of range; looping back.")
            self.current_start = self.full_start_date

        # Prepare NEXT epoch’s window so PL reload uses it
        start, end = self._window_for_current()
        trainer.datamodule.set_train_window(start, end)
        rank = self._rank_str()
        if getattr(trainer, "is_global_zero", True):
            print(f"[Rank {rank}] [Sequential Train] NEXT -> {start.date()} .. {end.date()}")

    # --- checkpointing support (true resume) ---
    def state_dict(self):
        return {
            "current_start": str(self.current_start),
        }

    def load_state_dict(self, state_dict):
        try:
            v = state_dict.get("current_start", None)
            self.current_start = pd.to_datetime(v) if v is not None else self.full_start_date
            self._restored_from_ckpt = v is not None
        except Exception:
            self.current_start = self.full_start_date
            self._restored_from_ckpt = False


# -----------------------------------
# Validation window scheduler
# -----------------------------------
class ValWindowCallback(pl.Callback):
    """Rotate validation windows within a given validation pool.

    Purpose
    -------
    - Use a *small* validation window (e.g., 8 days) to avoid OOM.
    - Still cover a full year (e.g., 2024) by rotating the window.
    - Avoid recomputing the validation summary every epoch by allowing
      `update_every_n_epochs > 1`.

    Notes
    -----
    - Updating happens at validation epoch end so that PL's
      `reload_dataloaders_every_n_epochs=1` will pick it up for the NEXT epoch.
    - If you use EarlyStopping on `val_loss`, rotating validation windows increases
      metric variance; increase patience / min_delta accordingly.
    """

    def __init__(
        self,
        val_start_date,
        val_end_date,
        val_window_days: int = 8,
        mode: str = "sequential",  # fixed|random|sequential
        stride_days: int = 8,
        update_every_n_epochs: int = 5,
    ):
        self.val_pool_start = pd.to_datetime(val_start_date)
        self.val_pool_end = pd.to_datetime(val_end_date)
        self.window = pd.Timedelta(days=int(val_window_days))
        self.mode = str(mode).lower()
        assert self.mode in {"fixed", "random", "sequential"}

        self.stride = pd.Timedelta(days=int(stride_days))
        self.update_every_n_epochs = max(1, int(update_every_n_epochs))

        self._seq_cursor: Optional[pd.Timestamp] = None
        # The currently active validation window start that was applied to the datamodule.
        # This is what we want to restore on resume.
        self._active_start: Optional[pd.Timestamp] = None

        if self.val_pool_end <= self.val_pool_start:
            raise ValueError(
                f"[ValWindowCallback] range invalid: {self.val_pool_start.date()} .. {self.val_pool_end.date()}"
            )
        if self.window <= pd.Timedelta(0):
            raise ValueError("val_window_days must be > 0")

    @staticmethod
    def _clip_end(new_end: pd.Timestamp, end_limit: pd.Timestamp) -> pd.Timestamp:
        return min(new_end, end_limit)

    def _pick_next(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        total_days = (self.val_pool_end - self.val_pool_start).days

        if self.mode == "fixed":
            start = self.val_pool_start
            end = self._clip_end(start + self.window, self.val_pool_end)
            return start, end

        if self.mode == "sequential":
            if self._seq_cursor is None:
                self._seq_cursor = self.val_pool_start
            start = self._seq_cursor
            end = self._clip_end(start + self.window, self.val_pool_end)
            next_start = start + self.stride
            if next_start >= self.val_pool_end:
                print("[ValWindowCallback] Reached end of val range; looping back.")
                next_start = self.val_pool_start
            self._seq_cursor = next_start
            return start, end

        # random
        max_off = max(0, total_days - self.window.days)
        off = random.randint(0, max_off) if max_off > 0 else 0
        start = self.val_pool_start + pd.Timedelta(days=off)
        end = self._clip_end(start + self.window, self.val_pool_end)
        return start, end

    def on_validation_epoch_start(self, trainer, pl_module):
        rank = int(os.environ.get("RANK", "0"))
        if getattr(trainer, "is_global_zero", True):
            print(
                f"[ValWindowCallback] pool={self.val_pool_start.date()}..{self.val_pool_end.date()} "
                f"mode={self.mode} win={self.window} stride={self.stride} update_every={self.update_every_n_epochs}"
            )
        # per-rank, show current DM window
        try:
            dm = trainer.datamodule
            print(f"[Rank {rank}] [ValWindowCallback] CURRENT dm.val={dm.hparams.val_start}..{dm.hparams.val_end}")
        except Exception:
            pass

    def on_fit_start(self, trainer, pl_module):
        """Restore active validation window on resume.

        Lightning restores callback state (via load_state_dict) but the datamodule
        does not restore its window. We therefore re-apply the active window here.
        """
        if self.mode == "fixed":
            start = self.val_pool_start
            end = self._clip_end(start + self.window, self.val_pool_end)
            trainer.datamodule.set_val_window(start, end)
            self._active_start = start
            return

        # If we have an active_start from checkpoint, restore exactly.
        if self._active_start is not None:
            start = self._active_start
            end = self._clip_end(start + self.window, self.val_pool_end)
            trainer.datamodule.set_val_window(start, end)
            # Ensure seq_cursor is consistent for the next scheduled update.
            if self.mode == "sequential":
                next_start = start + self.stride
                if next_start >= self.val_pool_end:
                    next_start = self.val_pool_start
                self._seq_cursor = next_start
            return

        # Backward compatibility: older checkpoints only had seq_cursor.
        if self.mode == "sequential" and self._seq_cursor is not None:
            # seq_cursor represents the NEXT start; infer the active window as previous stride.
            start = self._seq_cursor - self.stride
            if start < self.val_pool_start:
                # wrap to end
                span_days = max(1, int((self.val_pool_end - self.val_pool_start).days))
                stride_days = max(1, int(self.stride / pd.Timedelta(days=1)))
                # last possible aligned start within span
                off = (span_days - stride_days) % span_days
                start = self.val_pool_start + pd.Timedelta(days=off)
            end = self._clip_end(start + self.window, self.val_pool_end)
            trainer.datamodule.set_val_window(start, end)
            self._active_start = start
            return

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.mode == "fixed":
            return

        # Update cadence: compute NEXT val window only every N epochs
        # (epoch numbers are 0-based)
        next_epoch = int(pl_module.current_epoch) + 1
        if next_epoch % self.update_every_n_epochs != 0:
            return

        start, end = self._pick_next()
        self._active_start = start
        rank = int(os.environ.get("RANK", "0"))
        print(f"[Rank {rank}] [ValWindowCallback] NEXT -> {start.date()} .. {end.date()}")
        trainer.datamodule.set_val_window(start, end)

    # --- checkpointing support (true resume) ---
    def state_dict(self):
        return {
            "mode": self.mode,
            "seq_cursor": None if self._seq_cursor is None else str(self._seq_cursor),
            "active_start": None if self._active_start is None else str(self._active_start),
        }

    def load_state_dict(self, state_dict):
        try:
            self.mode = str(state_dict.get("mode", self.mode)).lower()
            v = state_dict.get("seq_cursor", None)
            self._seq_cursor = None if v in (None, "None") else pd.to_datetime(v)
            a = state_dict.get("active_start", None)
            self._active_start = None if a in (None, "None") else pd.to_datetime(a)
        except Exception:
            self._seq_cursor = None
            self._active_start = None
