import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from timing_utils import timing_resource_decorator


@timing_resource_decorator
def organize_bins_times(z, start_date, end_date, selected_satelliteId):
    """
    Organizes satellite observation times into 12-hour bins and creates input-target pairs
    for time-series prediction.

    - Reads satellite observation times and filters data for a specific week (April 1-7, 2024)
      and a selected satellite (ID 224).
    - Groups observations into 12-hour time bins.
    - Creates a mapping of input and target time indices for each bin, forming sequential
      input-target pairs for model training.

    Returns:
        dict: A dictionary where each key represents a time bin (e.g., 'bin1', 'bin2') and
              contains input-target time indices and corresponding timestamps.
    """
    # Read time and convert to pandas datetime
    time = pd.to_datetime(z["time"][:], unit="s")
    satellite_ids = z["satelliteId"][:]

    # Select data based on the given time range and satellite ID
    selected_times = np.where(
        (time >= start_date) & (time <= end_date) & (satellite_ids == selected_satelliteId)
    )[0]

    # Filter data for the specified week and satellite
    df = pd.DataFrame({"time": time[selected_times], "zar_time": z["time"][selected_times]})
    df["index"] = np.where(
        (time >= start_date) & (time <= end_date) & (satellite_ids == selected_satelliteId)
    )[0]
    df["time_bin"] = df["time"].dt.floor("12h")

    # Sort by time
    df = df.sort_values(by="zar_time")
    data_summary = {}

    # Iterate over the time bins and shift them to form input-target pairs
    unique_bins = df["time_bin"].unique()
    print("Unique time bins:", df["time_bin"].unique())

    for i in range(len(unique_bins) - 1):  # Exclude last bin (no target)
        input_times = df[df["time_bin"] == unique_bins[i]]["zar_time"].values
        target_times = df[df["time_bin"] == unique_bins[i + 1]]["zar_time"].values
        data_summary[f"bin{i+1}"] = {
            "input_time": unique_bins[i],
            "target_time": unique_bins[i + 1],
            "input_time_index": input_times,
            "target_time_index": target_times,
        }

    return data_summary


@timing_resource_decorator
def extract_features(z, data_summary):
    """
    Extracts and normalizes input and target features for each time bin in the dataset.

    Parameters:
        z (zarr.Group): The Zarr dataset containing satellite observation data.
        data_summary (dict): Dictionary containing time bins and corresponding input/target time indices.

    Returns:
        dict: Updated data_summary with additional keys:
            - 'input_features_final': Normalized input features including latitude, longitude, and sensor measurements.
            - 'target_features_final': Normalized target features including latitude, longitude, and brightness temperatures.

    Notes:
        - Uses MinMax scaling for normalization.
        - Adds latitude and longitude (converted to radians) to both input and target features.
    """

    # Initialize scalers
    minmax_scaler_input = MinMaxScaler()
    minmax_scaler_target = MinMaxScaler()

    # Extract all necessary data at once (reduces repeated Zarr indexing)
    all_times = z["time"][:]
    latitude_rad = np.radians(z["latitude"][:])[:, None]  # Convert once, reshape for stacking
    longitude_rad = np.radians(z["longitude"][:])[:, None]

    # Extract all sensor angles at once (batch indexing)
    sensor_zenith = z["sensorZenithAngle"][:]
    solar_zenith = z["solarZenithAngle"][:]
    solar_azimuth = z["solarAzimuthAngle"][:]

    # Extract all 22 BT channels efficiently
    bt_channels = np.stack([z[f"bt_channel_{i}"][:] for i in range(1, 23)], axis=1)

    for bin_name in data_summary.keys():  # Process all bins
        # Find indices for input and target times
        input_mask = np.isin(all_times, data_summary[bin_name]["input_time_index"])
        target_mask = np.isin(all_times, data_summary[bin_name]["target_time_index"])

        # Prepare input features (batch extraction, avoid repeated indexing)
        input_features_orig = np.column_stack(
            [
                sensor_zenith[input_mask],
                solar_zenith[input_mask],
                solar_azimuth[input_mask],
                bt_channels[input_mask],
            ]
        )
        input_features_normalized = minmax_scaler_input.fit_transform(input_features_orig)

        input_features_final = input_features_normalized

        # Metadata = lat, lon, and original (non-normalized) angles
        input_metadata = np.column_stack([
            latitude_rad[input_mask],
            longitude_rad[input_mask],
            sensor_zenith[input_mask],
            solar_zenith[input_mask],
            solar_azimuth[input_mask],
        ])

        # Prepare target features
        target_features_orig = bt_channels[target_mask]
        target_features_normalized = minmax_scaler_target.fit_transform(target_features_orig)

        # Final target = only normalized BTs
        target_features_final = target_features_normalized
        # Metadata = lat/lon only (you can include angles if needed)
        target_metadata = np.column_stack([
            latitude_rad[target_mask],
            longitude_rad[target_mask],
            sensor_zenith[target_mask],
            solar_zenith[target_mask],
            solar_azimuth[target_mask],
        ])
        # Convert to tensors at the end
        data_summary[bin_name]["input_features_final"] = torch.tensor(
            input_features_final, dtype=torch.float32
        )
        data_summary[bin_name]["target_features_final"] = torch.tensor(
            target_features_final, dtype=torch.float32
        )
        data_summary[bin_name]["input_metadata"] = torch.tensor(
            input_metadata, dtype=torch.float32
        )
        data_summary[bin_name]["target_metadata"] = torch.tensor(
            target_metadata, dtype=torch.float32
        )

        # Store min/max values for later unnormalization
        data_summary[bin_name]["target_scaler_min"] = minmax_scaler_target.data_min_
        data_summary[bin_name]["target_scaler_max"] = minmax_scaler_target.data_max_
    return data_summary
