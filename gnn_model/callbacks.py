import lightning.pytorch as pl
import pandas as pd
import random


class ResampleDataCallback(pl.Callback):
    """
    Callback to resample a new, random time window from the full dataset
    at the beginning of each training epoch.
    """

    def __init__(self, full_start_date, full_end_date, window_days=7):
        self.full_start_date = pd.to_datetime(full_start_date)
        # Adjust end date to ensure a full window can always be sampled
        self.full_end_date = pd.to_datetime(full_end_date) - pd.Timedelta(
            days=window_days
        )
        self.window_days = pd.Timedelta(days=window_days)

    def on_train_epoch_start(self, trainer, pl_module):
        # Calculate a new random start date from the full date range
        total_days = (self.full_end_date - self.full_start_date).days
        random_day_offset = random.randint(0, total_days)
        new_start = self.full_start_date + pd.Timedelta(days=random_day_offset)
        new_end = new_start + self.window_days

        print(
            f"\n[Random Sampler] Resampling new window: {new_start.date()} to {new_end.date()}\n"
        )

        # Update the datamodule with the new time window
        datamodule = trainer.datamodule
        datamodule.hparams.start_date = new_start
        datamodule.hparams.end_date = new_end

        # Re-run the setup logic for the new, smaller window of data
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
