import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os


def plot_instrument_maps(instrument_name, epoch, batch_idx, num_channels=1, data_dir="val_csv"):
    """
    Loads prediction data for a given instrument and generates map plots for each channel.

    Args:
        instrument_name (str): The name of the instrument (e.g., 'surface_obs', 'atms').
        epoch (int): The epoch number to plot.
        batch_idx (int): The batch index to plot.
        num_channels (int, optional): The number of channels for this instrument. Defaults to 1.
        data_dir (str, optional): The directory where the CSV files are located. Defaults to "val_csv".
    """
    filepath = f"{data_dir}/val_{instrument_name}_target_epoch{epoch}_batch{batch_idx}_step0.csv"

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} from {filepath} ---")
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file {filepath}. Skipping.")
        return

    # --- 2. Loop Through Channels and Plot ---
    for i in range(1, num_channels + 1):
        true_col = f"true_ch{i}"
        pred_col = f"pred_ch{i}"

        # Ensure the necessary columns exist in the DataFrame
        if not all(col in df.columns for col in [true_col, pred_col, "lon", "lat"]):
            print(f"Warning: Missing required columns for channel {i} in {filepath}. Skipping.")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        fig.suptitle(f"Instrument: {instrument_name} - Channel: {i} - Epoch: {epoch}", fontsize=16)

        # Get shared color scale limits for consistency
        vmin = min(df[true_col].min(), df[pred_col].min())
        vmax = max(df[true_col].max(), df[pred_col].max())

        # a) Plot Ground Truth
        sc1 = axes[0].scatter(df["lon"], df["lat"], c=df[true_col], cmap="jet", s=5, vmin=vmin, vmax=vmax)
        fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
        axes[0].set_title("Ground Truth")

        # b) Plot Prediction
        sc2 = axes[1].scatter(df["lon"], df["lat"], c=df[pred_col], cmap="jet", s=5, vmin=vmin, vmax=vmax)
        fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
        axes[1].set_title("Prediction")

        # c) Plot Percent Error
        # Avoid division by zero for the error calculation
        error = 100 * (df[pred_col] - df[true_col]) / df[true_col].replace(0, 1e-9)
        error_norm = TwoSlopeNorm(vmin=error.min(), vcenter=0, vmax=error.max())

        sc3 = axes[2].scatter(df["lon"], df["lat"], c=error, cmap="bwr", norm=error_norm, s=5)
        fig.colorbar(sc3, ax=axes[2], orientation="horizontal", pad=0.1).set_label("% Error")
        axes[2].set_title("Percent Error")

        for ax in axes:
            ax.set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        # --- 3. Save Figure ---
        output_filename = f"{instrument_name}_map_channel_{i}_epoch_{epoch}.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_filename)
        plt.close()
        print(f"  -> Saved plot: {output_filename}")


if __name__ == "__main__":
    # --- Configuration ---
    EPOCH_TO_PLOT = 76
    BATCH_IDX_TO_PLOT = 0

    # Define all the instruments to plot and their number of channels
    INSTRUMENTS = {"surface_obs": 1, "radiosonde": 1, "atms": 22}
    # -------------------

    for name, channels in INSTRUMENTS.items():
        plot_instrument_maps(name, EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=channels)
