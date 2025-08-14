import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from timing_utils import timing_resource_decorator


@timing_resource_decorator
def organize_bins_times(z_dict, start_date, end_date, observation_config, window_size="12h"):
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
    delta_satellite = observation_config.get("pipeline", {}).get("subsample", {}).get("satellite", 25)
    delta_surface = observation_config.get("pipeline", {}).get("subsample", {}).get("conventional", 20)
    # Validate window_size format
    valid_units = ["h", "H"]
    if not any(window_size.endswith(unit) for unit in valid_units):
        raise ValueError(f"Invalid window_size format: {window_size}\n" f"Must end with one of: {valid_units}\n" f"Examples: '6h' for 6 hours")
    data_summary = {}
    for obs_type in observation_config.keys():
        for key in observation_config[obs_type].keys():
            z = z_dict[obs_type][key]
            time = pd.to_datetime(z["time"][:], unit="s")
            time_cond = (time >= start_date) & (time < end_date)

            if obs_type == "satellite":
                sat_ids = observation_config[obs_type][key]["sat_ids"]
                available_sats = z["satelliteId"][:]
                assert isinstance(sat_ids, list), (
                    f"Configuration error: satellite IDs must be a list, got {type(sat_ids)} instead.\n" f"Key: {key}, Value: {sat_ids}"
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
            print(df["time_window"].value_counts().sort_index())  # show how full each bin is

            for i in range(len(unique_time_windows) - 1):  # Exclude last bin (no target)
                bin_name = f"bin{i+1}"
                input_indices = df[df["time_window"] == unique_time_windows[i]]["index"].values
                target_indices = df[df["time_window"] == unique_time_windows[i + 1]]["index"].values

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
    using z-score, and attaches both the normalized features and metadata
    (e.g., lat/lon and angles) to the corresponding entry in `data_summary`.

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

            # Get the QC filter configuration for the current instrument
            qc_filters = observation_config[obs_type][inst_name].get("qc_filters")

            # Apply quality control based on the instrument name
            if qc_filters:
                print(f"Applying QC for {inst_name}...")
                valid_input_mask = np.ones(len(input_idx), dtype=bool)
                valid_target_mask = np.ones(len(target_idx), dtype=bool)

                # Sequentially apply each filter defined in the config
                for var, valid_range in qc_filters.items():
                    # Load the data for the current QC variable
                    input_qc_data = z[var][input_idx]
                    target_qc_data = z[var][target_idx]

                    # Update the masks. A row is only kept if it passes ALL checks.
                    valid_input_mask &= (input_qc_data > valid_range[0]) & (input_qc_data < valid_range[1])
                    valid_target_mask &= (target_qc_data > valid_range[0]) & (target_qc_data < valid_range[1])

                # Apply the mask to the indices
                input_idx = input_idx[valid_input_mask]
                target_idx = target_idx[valid_target_mask]

                if len(input_idx) == 0 or len(target_idx) == 0:
                    print(f"Skipping bin {bin_name} for {inst_name} after QC filtering.")
                    continue

            # --- Load ALL Raw Data ---
            input_features_raw = np.column_stack([z[key][input_idx] for key in observation_config[obs_type][inst_name]["features"]]).astype(
                np.float32
            )

            input_metadata_raw = np.column_stack([z[key][input_idx] for key in observation_config[obs_type][inst_name]["metadata"]]).astype(
                np.float32
            )
            input_lat_raw = z["latitude"][input_idx]
            input_lon_raw = z["longitude"][input_idx]
            input_times_raw = z["time"][input_idx]

            target_features_raw = np.column_stack([z[key][target_idx] for key in observation_config[obs_type][inst_name]["features"]]).astype(
                np.float32
            )

            target_metadata_raw = np.column_stack([z[key][target_idx] for key in observation_config[obs_type][inst_name]["metadata"]]).astype(
                np.float32
            )

            target_lat_raw = z["latitude"][target_idx]
            target_lon_raw = z["longitude"][target_idx]
            target_times_raw = z["time"][target_idx]

            # --- Replace Fill Values with NaN ---
            # The fill value used in the Zarr dataset
            FILL_VALUE = 3.402823e38
            input_features_raw[input_features_raw >= FILL_VALUE] = np.nan
            input_metadata_raw[input_metadata_raw >= FILL_VALUE] = np.nan
            target_features_raw[target_features_raw >= FILL_VALUE] = np.nan
            target_metadata_raw[target_metadata_raw >= FILL_VALUE] = np.nan

            # Create a combined array to find any row with a NaN value
            input_combined = np.concatenate([input_features_raw, input_metadata_raw], axis=1)
            valid_input_mask = ~np.isnan(input_combined).any(axis=1)

            # Apply this mask to ALL input-related arrays
            input_idx = input_idx[valid_input_mask]
            input_features_raw = input_features_raw[valid_input_mask]
            input_metadata_raw = input_metadata_raw[valid_input_mask]
            input_lat_raw = input_lat_raw[valid_input_mask]
            input_lon_raw = input_lon_raw[valid_input_mask]
            input_times_clean = input_times_raw[valid_input_mask]

            target_combined = np.concatenate([target_features_raw, target_metadata_raw], axis=1)
            valid_target_mask = ~np.isnan(target_combined).any(axis=1)

            # Apply the mask to ALL target-related arrays
            target_idx = target_idx[valid_target_mask]
            target_features_raw = target_features_raw[valid_target_mask]
            target_metadata_raw = target_metadata_raw[valid_target_mask]
            target_lat_raw = target_lat_raw[valid_target_mask]
            target_lon_raw = target_lon_raw[valid_target_mask]
            target_times_clean = target_times_raw[valid_target_mask]

            # If after filtering, any array is empty, skip this bin
            if input_features_raw.shape[0] == 0 or target_features_raw.shape[0] == 0:
                print(f"Skipping bin {bin_name} for {inst_name} after NaN removal.")
                continue

            # --- Create the final feature arrays using the CLEANED data ---
            lat_rad_input = np.radians(input_lat_raw)[:, None]
            lon_rad_input = np.radians(input_lon_raw)[:, None]
            input_sin_lat = np.sin(lat_rad_input)
            input_cos_lat = np.cos(lat_rad_input)
            input_sin_lon = np.sin(lon_rad_input)
            input_cos_lon = np.cos(lon_rad_input)

            lat_rad_target = np.radians(target_lat_raw)[:, None]
            lon_rad_target = np.radians(target_lon_raw)[:, None]
            # --- Create Time Features ---
            input_timestamps = pd.to_datetime(input_times_clean, unit="s")
            input_dayofyear = np.array(
                [
                    (timestamp.timetuple().tm_yday - 1 + (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400) / 365.24219
                    for timestamp in input_timestamps
                ]
            )[:, None]

            # Time of day as fraction [0, 1]
            input_time_fraction = np.array([(ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400 for ts in input_timestamps])

            input_sin_time = np.sin(2 * np.pi * input_time_fraction)[:, None]
            input_cos_time = np.cos(2 * np.pi * input_time_fraction)[:, None]

            target_timestamps = pd.to_datetime(target_times_clean, unit="s")
            target_time_fraction = np.array([(ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400 for ts in target_timestamps])
            target_sin_time = np.sin(2 * np.pi * target_time_fraction)[:, None]
            target_cos_time = np.cos(2 * np.pi * target_time_fraction)[:, None]

            # If after filtering, any array is empty, skip this bin
            if input_features_raw.shape[0] == 0 or target_features_raw.shape[0] == 0:
                print(f"Skipping bin {bin_name} for {inst_name} after NaN removal.")
                continue

            all_features_raw = np.concatenate([input_features_raw, target_features_raw], axis=0)

            # Normalize features
            # Check the standard deviation of the raw features
            std_dev = np.std(all_features_raw, axis=0)
            if np.any(std_dev == 0):
                print(f"WARNING: Bin {bin_name} for {inst_name} contains constant data...")
                all_features_norm = np.zeros_like(all_features_raw, dtype=np.float32)
            else:
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
                        input_sin_time,
                        input_cos_time,
                        input_dayofyear,
                        input_metadata,
                        input_features_norm,
                    ]
                )

                # Assemble final target features for satellite (with scan angle)
                scan_angle = target_metadata_cos[:, 0:1]
                target_features_final = target_features_norm

                # Assemble target metadata for plotting
                target_metadata = np.column_stack([lat_rad_target, lon_rad_target, target_metadata_norm])

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
                        input_sin_time,
                        input_cos_time,
                        input_dayofyear,
                        input_metadata,
                        input_features_norm,
                    ]
                )

                # The target is JUST the normalized features
                target_features_final = target_features_norm

                # Assemble target metadata for plotting
                target_metadata = np.column_stack([lat_rad_target, lon_rad_target, target_metadata_norm])

            # --- Assemble Final Data for the Bin ---
            data_summary_bin["input_features_final"] = torch.tensor(input_features_final, dtype=torch.float32)
            data_summary_bin["target_features_final"] = torch.tensor(target_features_final, dtype=torch.float32)

            input_metadata_for_graph = np.column_stack([lat_rad_input, lon_rad_input])
            data_summary_bin["input_metadata"] = torch.tensor(input_metadata_for_graph, dtype=torch.float32)
            data_summary_bin["target_metadata"] = torch.tensor(target_metadata, dtype=torch.float32)
            data_summary_bin["scan_angle"] = torch.tensor(scan_angle, dtype=torch.float32)

            # Save lat/lon degrees separately for CSV and evaluation
            data_summary_bin["input_lat_deg"] = z["latitude"][input_idx]
            data_summary_bin["input_lon_deg"] = z["longitude"][input_idx]
            data_summary_bin["target_lat_deg"] = z["latitude"][target_idx]
            data_summary_bin["target_lon_deg"] = z["longitude"][target_idx]
            NAME2ID = {"atms": 0, "surface_obs": 1, "radiosonde": 2}
            data_summary_bin["instrument_id"] = NAME2ID[inst_name]

            print(f"[{bin_name}] input_features_final shape: {input_features_final.shape}")
            print(f"[{bin_name}] target_features_final shape: {target_features_final.shape}")

    return data_summary
