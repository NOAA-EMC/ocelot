import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from timing_utils import timing_resource_decorator


@timing_resource_decorator
def organize_bins_times(
    z_dict, start_date, end_date, observation_config, window_size="12h"
):
    """
    Bin definition: a bin consists of a pair of input and targets, each covers window_size.
    Organizes satellite observation times into time bins and creates input-target pairs
    for time-series prediction.

    Args:
        z_dict (dict): Dictionary containing observation data for different types
        start_date (datetime): Start date for filtering data
        end_date (datetime): End date for filtering data
        observation_config (dict): Configuration for different observation types
        window_size (str, optional): Size of input (or target) of each bin. Accepts pandas offset strings:
            - 'h' or 'H': hours (e.g., '6h' for 6 hours)
            Default is '12h' (12 hours)

    Returns:
        dict: A dictionary where each key represents a time bin (e.g., 'bin1', 'bin2') and
              contains input-target time indices and corresponding timestamps.

    Raises:
        ValueError: If window_size format is invalid
    """
    delta_satellite = 25
    delta_surface = 20
    # Validate window_size format
    valid_units = ["h", "H"]
    if not any(window_size.endswith(unit) for unit in valid_units):
        raise ValueError(
            f"Invalid window_size format: {window_size}\n"
            f"Must end with one of: {valid_units}\n"
            f"Examples: '6h' for 6 hours"
        )
    data_summary = {}
    for obs_type in observation_config.keys():
        for key in observation_config[obs_type].keys():
            z = z_dict[obs_type][key]
            # Read time and convert to pandas datetime
            time = pd.to_datetime(z["time"][:], unit="s")
            # Select data based on the given time range and satellite ID
            time_cond = (time >= start_date) & (time < end_date)

            if obs_type == "satellite":
                sat_ids = observation_config[obs_type][key]["sat_ids"]
                available_sats = z["satelliteId"][:]
                assert isinstance(sat_ids, list), (
                    f"Configuration error: satellite IDs must be a list, got {type(sat_ids)} instead.\n"
                    f"Key: {key}, Value: {sat_ids}"
                )

                invalid_sats = [sid for sid in sat_ids if sid not in available_sats]
                if invalid_sats:
                    raise ValueError(
                        f"Error in {obs_type} configuration for {key}:\n"
                        f"Invalid satellite IDs found: {invalid_sats}\n"
                        f"Available satellite IDs: {np.unique(available_sats)}\n"
                        f"Please check your observation configuration."
                    )

                sat_ids_mask = np.isin(z["satelliteId"][:], sat_ids)
                selected_times = np.where(time_cond & sat_ids_mask)[0]
            else:
                selected_times = np.where(time_cond)[0]

            df = pd.DataFrame(
                {
                    "time": time[selected_times],
                    "zar_time": z["time"][selected_times],
                    "index": selected_times,
                }
            )
            df["time_window"] = df["time"].dt.floor(window_size)

            # Sort by time
            df = df.sort_values(by="zar_time")

            unique_time_windows = df["time_window"].unique()
            print("Filtered observation times:")
            print("  - Start:", df["time"].min())
            print("  - End:", df["time"].max())
            print(
                df["time_window"].value_counts().sort_index()
            )  # show how full each bin is

            for i in range(
                len(unique_time_windows) - 1
            ):  # Exclude last bin (no target)
                bin_name = f"bin{i+1}"
                input_indices = df[df["time_window"] == unique_time_windows[i]][
                    "index"
                ].values
                target_indices = df[df["time_window"] == unique_time_windows[i + 1]][
                    "index"
                ].values

                # Apply the fixed-rate subsampling
                if obs_type == "satellite":
                    input_indices = input_indices[::delta_satellite]
                    target_indices = target_indices[::delta_satellite]
                else:  # For conventional data
                    input_indices = input_indices[::delta_surface]
                    target_indices = target_indices[::delta_surface]

                # Initialize nested dictionaries if they don't exist
                if bin_name not in data_summary:
                    data_summary[bin_name] = {}
                if obs_type not in data_summary[bin_name]:
                    data_summary[bin_name][obs_type] = {}

                data_summary[bin_name][obs_type][key] = {
                    "input_time": unique_time_windows[i],  # datetime
                    "target_time": unique_time_windows[i + 1],
                    "input_time_index": input_indices,
                    "target_time_index": target_indices,
                }
            print(f"Created {len(data_summary)} bins (pair of input-target).")
    return data_summary


@timing_resource_decorator
def extract_features(z_dict, data_summary, bin_name, observation_config):
    """
    Loads and normalizes input and target features for each time bin individually.

    This function processes one bin at a time to minimize memory usage. For each bin,
    it extracts only the necessary indices from the Zarr dataset, normalizes the features
    using MinMax scaling, and attaches both the normalized features and metadata
    (e.g., lat/lon and angles) to the corresponding entry in `data_summary`.

    This per-bin loading strategy ensures compatibility with large-scale, distributed
    training by avoiding full-array preloading into memory.

    Parameters:
        z_dict (dict): Dictionary containing Zarr datasets for each observation type.
        data_summary (dict): Dictionary containing input and target indices for each bin.
        observation_config (dict): Configuration for observation types.

    Returns:
        dict: Updated data_summary with:
            - 'input_features_final': Normalized input features.
            - 'target_features_final': Normalized target features.
            - 'input_metadata', 'target_metadata': Metadata arrays.
            - 'target_scaler_min', 'target_scaler_max': Min/max for unnormalization.

    """

    # for bin_name in data_summary.keys():  # Process bins in order
    print(f"\nProcessing {bin_name}...")
    for obs_type in data_summary[bin_name].keys():
        for inst_name in data_summary[bin_name][obs_type].keys():
            print(f"obs: {obs_type}: {inst_name}")
            z = z_dict[obs_type][inst_name]

            data_summary_bin = data_summary[bin_name][obs_type][inst_name]
            input_idx = data_summary_bin["input_time_index"]
            target_idx = data_summary_bin["target_time_index"]

            if len(input_idx) == 0 or len(target_idx) == 0:
                print(f"Skipping bin {bin_name} because input or target is empty.")
                continue

            if inst_name == 'surface_obs':
                print(f'original input surface_obs, {len(input_idx)}')
                print(f'original target surface_obs, {len(target_idx)}')
                input_surface_t = z['virtualTemperature'][input_idx]
                target_surface_t = z['virtualTemperature'][target_idx]
                input_surface_t_idx = input_surface_t[:] < 100
                target_surface_t_idx = target_surface_t[:] < 100
                input_idx = input_idx[input_surface_t_idx]
                target_idx = target_idx[target_surface_t_idx]
                print(f'final input surface_obs, {len(input_idx)}')
                print(f'final target surface_obs, {len(input_idx)}')

            # === Extract only necessary points for this bin ===
            lat_rad_input = np.radians(z["latitude"][input_idx])[:, None]
            lon_rad_input = np.radians(z["longitude"][input_idx])[:, None]
            input_sin_lat = np.sin(lat_rad_input)
            input_cos_lat = np.cos(lat_rad_input)
            input_sin_lon = np.sin(lon_rad_input)
            input_cos_lon = np.cos(lon_rad_input)

            # Compute time of the year
            input_times = z["time"][input_idx][:]
            input_timestamps = pd.to_datetime(input_times, unit="s")
            input_dayofyear = np.array(
                [
                    (
                        timestamp.timetuple().tm_yday
                        - 1
                        + (
                            timestamp.hour * 3600
                            + timestamp.minute * 60
                            + timestamp.second
                        )
                        / 86400
                    )
                    / 365.24219
                    for timestamp in input_timestamps
                ]
            )[:, None]

            # Time of day as fraction [0, 1]
            input_time_fraction = np.array(
                [
                    (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400
                    for ts in input_timestamps
                ]
            )

            input_sin_time = np.sin(2 * np.pi * input_time_fraction)[:, None]
            input_cos_time = np.cos(2 * np.pi * input_time_fraction)[:, None]

            # Ensure all raw data is loaded as float32 to prevent gradient issues.
            input_features_raw = np.column_stack(
                [
                    z[key][input_idx]
                    for key in observation_config[obs_type][inst_name]["features"]
                ]
            ).astype(np.float32)
            input_metadata_raw = np.column_stack(
                [
                    z[key][input_idx]
                    for key in observation_config[obs_type][inst_name]["metadata"]
                ]
            ).astype(np.float32)

            # --- Process Target Features and Metadata ---
            target_times = z["time"][target_idx][:]
            target_timestamps = pd.to_datetime(target_times, unit="s")
            target_time_fraction = np.array(
                [
                    (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400
                    for ts in target_timestamps
                ]
            )
            target_sin_time = np.sin(2 * np.pi * target_time_fraction)[:, None]
            target_cos_time = np.cos(2 * np.pi * target_time_fraction)[:, None]

            lat_rad_target = np.radians(z["latitude"][target_idx])[:, None]
            lon_rad_target = np.radians(z["longitude"][target_idx])[:, None]
            target_sin_lat = np.sin(lat_rad_target)
            target_cos_lat = np.cos(lat_rad_target)
            target_sin_lon = np.sin(lon_rad_target)
            target_cos_lon = np.cos(lon_rad_target)

            target_features_raw = np.column_stack(
                [
                    z[key][target_idx]
                    for key in observation_config[obs_type][inst_name]["features"]
                ]
            ).astype(np.float32)

            target_metadata_raw = np.column_stack(
                [
                    z[key][target_idx]
                    for key in observation_config[obs_type][inst_name]["metadata"]
                ]
            ).astype(np.float32)

            all_features_raw = np.concatenate(
                [input_features_raw, target_features_raw], axis=0
            )

            # Normalize features
            bin_scaler = StandardScaler()
            all_features_norm = bin_scaler.fit_transform(all_features_raw)
            n_input = input_features_raw.shape[0]
            input_features_norm = all_features_norm[:n_input]
            target_features_norm = all_features_norm[n_input:]

            if obs_type == "satellite":
                # Normalize to encode input metadata angles
                input_metadata_rad = np.deg2rad(input_metadata_raw)
                input_metadata_cos = np.cos(input_metadata_rad)
                input_metadata = input_metadata_cos
                # Normalize target metadata angles for the decoder
                target_metadata_rad = np.deg2rad(target_metadata_raw)
                target_metadata_cos = np.cos(target_metadata_rad)
                target_metadata_norm = target_metadata_cos
                # Assemble final input features for satellite
                input_features_final = np.column_stack(
                    [
                        input_sin_lat,
                        input_cos_lat,
                        input_sin_lon,
                        input_cos_lon,
                        input_dayofyear,
                        input_metadata,
                        input_features_norm,
                    ]
                )

                # Assemble final target features for satellite (with scan angle)
                scan_angle = target_metadata_cos[:, 0:1]
                target_features_final = np.column_stack(
                    [target_features_norm, scan_angle]
                )

                # Assemble target metadata for plotting
                target_metadata = np.column_stack(
                    [lat_rad_target, lon_rad_target, target_metadata_norm]
                )

            else:
                # --- CONVENTIONAL-SPECIFIC LOGIC ---
                features_scaler = StandardScaler()
                input_metadata_norm = features_scaler.fit_transform(input_metadata_raw)
                target_scaler = StandardScaler()
                target_metadata_norm = target_scaler.fit_transform(target_metadata_raw)

                input_metadata = input_metadata_norm  # For conventional, it's just the normalized metadata

                # Assemble final input features for conventional
                input_features_final = np.column_stack(
                    [
                        input_sin_lat,
                        input_cos_lat,
                        input_sin_lon,
                        input_cos_lon,
                        input_dayofyear,
                        input_metadata,
                        input_features_norm,
                    ]
                )

                # The target is JUST the normalized features
                target_features_final = target_features_norm

                # Assemble target metadata for plotting
                target_metadata = np.column_stack(
                    [lat_rad_target, lon_rad_target, target_metadata_norm]
                )

            # --- Assemble Final Data for the Bin ---
            data_summary_bin["input_features_final"] = torch.tensor(
                input_features_final, dtype=torch.float32
            )
            data_summary_bin["target_features_final"] = torch.tensor(
                target_features_final, dtype=torch.float32
            )

            # This can be simplified as it's the same for both
            input_metadata_for_graph = np.column_stack([lat_rad_input, lon_rad_input])
            data_summary_bin["input_metadata"] = torch.tensor(
                input_metadata_for_graph, dtype=torch.float32
            )
            data_summary_bin["target_metadata"] = torch.tensor(
                target_metadata, dtype=torch.float32
            )

            # Save lat/lon degrees separately for CSV and evaluation
            data_summary_bin["input_lat_deg"] = z["latitude"][input_idx]
            data_summary_bin["input_lon_deg"] = z["longitude"][input_idx]
            data_summary_bin["target_lat_deg"] = z["latitude"][target_idx]
            data_summary_bin["target_lon_deg"] = z["longitude"][target_idx]

            print(
                f"[{bin_name}] input_features_final shape: {input_features_final.shape}"
            )
            print(
                f"[{bin_name}] target_features_final shape: {target_features_final.shape}"
            )

    return data_summary
