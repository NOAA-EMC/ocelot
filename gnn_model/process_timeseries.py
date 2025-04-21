import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from timing_utils import timing_resource_decorator


@timing_resource_decorator
def organize_bins_times(z, start_date, end_date, selected_satelliteId):
    """
    Organizes satellite observation times into 12-hour bins and creates input-target pairs.
    """
    # Read time and convert to pandas datetime
    time = pd.to_datetime(z["time"][:], unit="s")
    satellite_ids = z["satelliteId"][:]

    # Select data based on the given time range and satellite ID
    selected_times = np.where(
        (time >= start_date) & (time <= end_date) & (satellite_ids == selected_satelliteId)
    )[0]

    df = pd.DataFrame({
        "time": time[selected_times],
        "zar_time": z["time"][selected_times],
        "index": selected_times
    })
    df["time_bin"] = df["time"].dt.floor("12h")

    # Sort by time
    df = df.sort_values(by="zar_time")

    unique_bins = df["time_bin"].unique()
    print("Unique time bins:", unique_bins)

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

    return data_summary


@timing_resource_decorator
def extract_features(z, data_summary):
    """
    Efficiently loads and normalizes features using only the necessary indices per bin.
    """
    minmax_scaler_input = MinMaxScaler()
    minmax_scaler_target = MinMaxScaler()

    for bin_name in data_summary.keys():
        input_idx = data_summary[bin_name]["input_time_index"]
        target_idx = data_summary[bin_name]["target_time_index"]

        # Load only the required indices
        lat_rad_input = np.radians(z["latitude"][input_idx])[:, None]
        lon_rad_input = np.radians(z["longitude"][input_idx])[:, None]
        lat_rad_target = np.radians(z["latitude"][target_idx])[:, None]
        lon_rad_target = np.radians(z["longitude"][target_idx])[:, None]

        sensor_zenith_input = z["sensorZenithAngle"][input_idx]
        solar_zenith_input = z["solarZenithAngle"][input_idx]
        solar_azimuth_input = z["solarAzimuthAngle"][input_idx]

        sensor_zenith_target = z["sensorZenithAngle"][target_idx]
        solar_zenith_target = z["solarZenithAngle"][target_idx]
        solar_azimuth_target = z["solarAzimuthAngle"][target_idx]
        
        # Load BT channels only for those indices
        bt_input = np.stack([z[f"bt_channel_{i}"][input_idx] for i in range(1, 23)], axis=1)
        bt_target = np.stack([z[f"bt_channel_{i}"][target_idx] for i in range(1, 23)], axis=1)

        # Normalize features
        input_features_orig = np.column_stack([
            sensor_zenith_input,
            solar_zenith_input,
            solar_azimuth_input,
            bt_input,
        ])
        input_features_norm = minmax_scaler_input.fit_transform(input_features_orig)

        target_features_orig = bt_target
        target_features_norm = minmax_scaler_target.fit_transform(target_features_orig)

        input_metadata = np.column_stack([
            lat_rad_input,
            lon_rad_input,
            sensor_zenith_input[:, None],
            solar_zenith_input[:, None],
            solar_azimuth_input[:, None],
        ])
        target_metadata = np.column_stack([
            lat_rad_target,
            lon_rad_target,
            sensor_zenith_target[:, None],
            solar_zenith_target[:, None],
            solar_azimuth_target[:, None],
        ])

        print(f"[{bin_name}] input_features_final shape: {input_features_norm.shape}")
        print(f"[{bin_name}] target_features_final shape: {target_features_norm.shape}")

        data_summary[bin_name]["input_features_final"] = torch.tensor(input_features_norm, dtype=torch.float32)
        data_summary[bin_name]["target_features_final"] = torch.tensor(target_features_norm, dtype=torch.float32)
        data_summary[bin_name]["input_metadata"] = torch.tensor(input_metadata, dtype=torch.float32)
        data_summary[bin_name]["target_metadata"] = torch.tensor(target_metadata, dtype=torch.float32)
        data_summary[bin_name]["target_scaler_min"] = minmax_scaler_target.data_min_
        data_summary[bin_name]["target_scaler_max"] = minmax_scaler_target.data_max_

    return data_summary
