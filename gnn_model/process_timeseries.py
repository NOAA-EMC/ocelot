import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from timing_utils import timing_resource_decorator

@timing_resource_decorator
def organize_bins_times(z, start_date, end_date, selected_satids):
    """
    Organizes satellite observation times into time bins and creates input-target pairs
    for time-series prediction.

    - Reads observation times and filters data for a specific period of time
    - Groups observations into time bins.
    - Creates a mapping of input and target time indices for each bin, forming sequential
      input-target pairs for model training.

    Returns:
    dict: A dictionary where each key represents a time bin (e.g., 'bin1', 'bin2') and
            contains input-target time indices and corresponding timestamps.
    """
    # Read time and convert to pandas datetime
    time = pd.to_datetime(z["time"][:], unit="s")
    satids = z["satelliteId"][:]

    # Select data based on the given time range and satellite ID
    time_cond=(time >= start_date) & (time < end_date)
    if selected_satids is None:
        selected_times = np.where(time_cond)[0]
    else:
        if not isinstance(selected_satids, list):
            selected_satids = [selected_satids]
        sat_mask = np.isin(satids, selected_satids)
        selected_times = np.where(time_cond & sat_mask)[0]

    df = pd.DataFrame({"time": time[selected_times], "zar_time": z["time"][selected_times], "index": selected_times})
    df["time_bin"] = df["time"].dt.floor("12h")

    # Sort by time
    df = df.sort_values(by="zar_time")

    unique_bins = df["time_bin"].unique()
    print("Filtered observation times:")
    print("  - Start:", df["time"].min())
    print("  - End:", df["time"].max())
    print("Unique time bins:", unique_bins)
    print(df["time_bin"].value_counts().sort_index())  # show how full each bin is

    data_summary = {}
    for i in range(len(unique_bins) - 1):  # Exclude last bin (no target)
        input_indices = df[df["time_bin"] == unique_bins[i]]["index"].values
        target_indices = df[df["time_bin"] == unique_bins[i + 1]]["index"].values
        data_summary[f"bin{i+1}"] = {
            "input_time": unique_bins[i],
            "target_time": unique_bins[i + 1],
            "input_time_index": input_indices,
            "target_time_index": target_indices,
        }
    print(f"Created {len(data_summary)} input-target bin pairs.")
    return data_summary

@timing_resource_decorator
def extract_features(z, data_summary):
    """
    Loads and normalizes input and target features for each time bin individually.

    This function processes one bin at a time to minimize memory usage. For each bin,
    it extracts only the necessary indices from the Zarr dataset, normalizes the features
    using MinMax scaling, and attaches both the normalized features and metadata
    (e.g., lat/lon and angles) to the corresponding entry in `data_summary`.

    This per-bin loading strategy ensures compatibility with large-scale, distributed
    training by avoiding full-array preloading into memory.

    Parameters:
        z (zarr.Group): The Zarr dataset containing observation data.
        data_summary (dict): Dictionary containing input and target indices for each bin.

    Returns:
        dict: Updated data_summary with:
            - 'input_features_final': Normalized input features.
            - 'target_features_final': Normalized target features.
            - 'input_metadata', 'target_metadata': Metadata arrays.
            - 'target_scaler_min', 'target_scaler_max': Min/max for unnormalization.

    """

    for bin_name in data_summary.keys():
        input_idx = data_summary[bin_name]["input_time_index"]
        target_idx = data_summary[bin_name]["target_time_index"]

        if len(input_idx) == 0 or len(target_idx) == 0:
            print(f"Skipping bin {bin_name} because input or target is empty.")
            continue

        # === Extract only necessary points for this bin ===
        lat_rad_input = np.radians(z["latitude"][input_idx])[:, None]
        lon_rad_input = np.radians(z["longitude"][input_idx])[:, None]
        lat_rad_target = np.radians(z["latitude"][target_idx])[:, None]
        lon_rad_target = np.radians(z["longitude"][target_idx])[:, None]

        # Compute sine and cosine of latitude and longitude
        sin_lat = np.sin(lat_rad_input)
        cos_lat = np.cos(lat_rad_input)
        sin_lon = np.sin(lon_rad_input)
        cos_lon = np.cos(lon_rad_input)

        # Compute time of the year
        input_times = z["time"][input_idx][:]
        input_timestamps = pd.to_datetime(input_times, unit='s')
        input_dayofyear = np.array([
            (timestamp.timetuple().tm_yday - 1 +
            (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400)/
            365.24219
            for timestamp in input_timestamps])[:, None]

        sensor_zenith_input = z["sensorZenithAngle"][input_idx][:, None]
        solar_zenith_input = z["solarZenithAngle"][input_idx][:, None]
        solar_azimuth_input = z["solarAzimuthAngle"][input_idx][:, None]

        sensor_zenith_target = z["sensorZenithAngle"][target_idx][:, None]
        solar_zenith_target = z["solarZenithAngle"][target_idx][:, None]
        solar_azimuth_target = z["solarAzimuthAngle"][target_idx][:, None]

        bt_input = np.stack([z[f"bt_channel_{i}"][input_idx] for i in range(1, 23)], axis=1)
        bt_target = np.stack([z[f"bt_channel_{i}"][target_idx] for i in range(1, 23)], axis=1)

        # === Normalize features ===
        input_features_orig = np.column_stack([
                sensor_zenith_input,
                solar_zenith_input,
                solar_azimuth_input,
                bt_input,
            ])
        input_scaler = MinMaxScaler()
        input_features_norm = input_scaler.fit_transform(input_features_orig)
        target_features_orig = bt_target
        target_scaler = MinMaxScaler()
        target_features_norm = target_scaler.fit_transform(target_features_orig)

        # === Input Feature data ===
        input_features_final = np.hstack([
                sin_lat,
                cos_lat,
                sin_lon,
                cos_lon,
                input_dayofyear,
                input_features_norm,
            ])

        # === Metadata ===
        input_metadata = np.column_stack([
                lat_rad_input,
                lon_rad_input,
                sensor_zenith_input, #[:, None],
                solar_zenith_input,  #[:, None],
                solar_azimuth_input, #[:, None],
            ])
        target_metadata = np.column_stack([
                lat_rad_target,
                lon_rad_target,
                sensor_zenith_target, #[:, None],
                solar_zenith_target,  #[:, None],
                solar_azimuth_target, #[:, None],
            ])

        # === Save ===
        data_summary[bin_name]["input_features_final"] = torch.tensor(input_features_final, dtype=torch.float32)
        data_summary[bin_name]["target_features_final"] = torch.tensor(target_features_norm, dtype=torch.float32)
        data_summary[bin_name]["input_metadata"] = torch.tensor(input_metadata, dtype=torch.float32)
        data_summary[bin_name]["target_metadata"] = torch.tensor(target_metadata, dtype=torch.float32)
        # Store min/max values for later unnormalization
        data_summary[bin_name]["target_scaler_min"] = target_scaler.data_min_
        data_summary[bin_name]["target_scaler_max"] = target_scaler.data_max_

        # Save lat/lon degrees separately for CSV and evaluation
        data_summary[bin_name]["input_lat_deg"] = z["latitude"][input_idx]
        data_summary[bin_name]["input_lon_deg"] = z["longitude"][input_idx]
        data_summary[bin_name]["target_lat_deg"] = z["latitude"][target_idx]
        data_summary[bin_name]["target_lon_deg"] = z["longitude"][target_idx]

        print(f"[{bin_name}] input_features_final shape: {input_features_final.shape}")
        print(f"[{bin_name}] target_features_final shape: {target_features_norm.shape}")
    return data_summary
