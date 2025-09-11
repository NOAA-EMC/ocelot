import lightning.pytorch as pl
import pandas as pd
import random


class ResampleDataCallback(pl.Callback):
    """
    Callback to resample from separate train/val date ranges
    """

    def __init__(self, train_start_date, train_end_date, val_start_date, val_end_date, train_window_days=14, val_window_days=3):
        # Training date range
        self.train_start_date = pd.to_datetime(train_start_date)
        self.train_end_date = pd.to_datetime(train_end_date)
        self.train_window_days = pd.Timedelta(days=train_window_days)

        # Validation date range
        self.val_start_date = pd.to_datetime(val_start_date)
        self.val_end_date = pd.to_datetime(val_end_date)
        self.val_window_days = pd.Timedelta(days=val_window_days)

        # Validate date ranges during initialization
        self._check_date_ranges()

    def on_train_epoch_start(self, trainer, pl_module):
        """Sample from training date range"""
        total_train_days = (self.train_end_date - self.train_start_date).days
        max_offset = total_train_days - self.train_window_days.days
        random_day_offset = random.randint(0, max_offset) if max_offset > 0 else 0
        new_start = self.train_start_date + pd.Timedelta(days=random_day_offset)
        new_end = new_start + self.train_window_days
        new_end = self._check_end_date(new_end, self.train_end_date, "TRAIN")

        print(f"\n[Random Sampler - TRAIN] Window: {new_start.date()} to {new_end.date()}\n")
        self._update_datamodule(trainer, new_start, new_end)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Sample from validation date range"""
        total_val_days = (self.val_end_date - self.val_start_date).days
        max_offset = total_val_days - self.val_window_days.days
        random_day_offset = random.randint(0, max_offset) if max_offset > 0 else 0
        new_start = self.val_start_date + pd.Timedelta(days=random_day_offset)
        new_end = new_start + self.val_window_days
        new_end = self._check_end_date(new_end, self.val_end_date, "VAL")

        print(f"\n[Random Sampler - VAL] Window: {new_start.date()} to {new_end.date()}\n")
        self._update_datamodule(trainer, new_start, new_end)

    def _check_date_ranges(self):
        """Validate date ranges during initialization"""
        if self.train_end_date < self.train_start_date:
            raise ValueError(
                f"[Random Sampler] Training date range error: "
                f"Start date ({self.train_start_date.date()}) "
                f"exceeds updated end date ({self.train_end_date.date()})"
            )

        if self.val_end_date < self.val_start_date:
            raise ValueError(
                f"[Random Sampler] Validation date range error: "
                f"Start date ({self.val_start_date.date()}) "
                f"exceeds updated end date ({self.val_end_date.date()})"
            )

    def _check_end_date(self, new_end, end_date_limit, mode):
        """Ensure the window doesn't exceed the date range limit"""
        if new_end > end_date_limit:
            print(f"\n[Random Sampler - {mode}] Warning: End date would exceed date range. "
                  f"Setting end date to the last available date: {end_date_limit.date()}\n")
            new_end = end_date_limit
        return new_end

    def _update_datamodule(self, trainer, start_date, end_date):
        """Helper to update datamodule and re-setup"""
        datamodule = trainer.datamodule
        datamodule.hparams.start_date = start_date
        datamodule.hparams.end_date = end_date
        datamodule.setup("fit")


class SequentialDataCallback(pl.Callback):
    """
    Callback to process data sequentially, one window at a time.

    At the beginning of each training epoch, this callback advances the
    datamodule's time window to the next chronological segment of the
    full dataset.
    """

    def __init__(self, full_start_date, full_end_date, window_days=7):
        self.full_start_date = pd.to_datetime(full_start_date)
        self.full_end_date = pd.to_datetime(full_end_date)
        self.window_days = pd.Timedelta(days=window_days)
        self.current_start_date = self.full_start_date

    def on_train_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # If we've gone past the end date, loop back to the beginning
        if self.current_start_date >= self.full_end_date:
            print(
                "\n[Sequential Sampler] Reached end of dataset. Looping back to the start."
            )
            self.current_start_date = self.full_start_date

        # 1. Define the window for the current epoch
        new_start = self.current_start_date
        new_end = new_start + self.window_days

        # 2. Ensure the window doesn't exceed the full end date
        if new_end > self.full_end_date:
            new_end = self.full_end_date

        print(
            f"\n[Sequential Sampler] Preparing new window: {new_start.date()} to {new_end.date()}\n"
        )

        # 3. Update the datamodule's hyperparameters with the new time window
        datamodule = trainer.datamodule
        datamodule.hparams.start_date = new_start
        datamodule.hparams.end_date = new_end

        # 4. Re-run the setup logic for the new window
        datamodule.setup("fit")

        # 5. Advance the start date for the next epoch
        # This creates an overlapping, "sliding" window
        self.current_start_date += pd.Timedelta(days=1)
