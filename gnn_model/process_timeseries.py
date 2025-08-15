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
                    "zar_time": (z["zar_time"][selected_times] if "zar_time" in z else z["time"][selected_times]),
                    "index": selected_times,
                }
            )

            df["time_window"] = df["time"].dt.floor(window_size)

            # Sort by time
            df = df.sort_values(by="zar_time")

            unique_time_windows = df["time_window"].unique()
            if df.empty:
                print(f"No observations for {obs_type}.{key} in the window " f"{start_date} → {end_date}")
                continue  # skip to next key
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
    Adds per-channel masks for conventional targets so features can be missing independently.
    """
    print(f"\nProcessing {bin_name}...")
    for obs_type in list(data_summary[bin_name].keys()):
        for inst_name in list(data_summary[bin_name][obs_type].keys()):
            z = z_dict[obs_type][inst_name]

            data_summary_bin = data_summary[bin_name][obs_type][inst_name]
            input_idx = np.asarray(data_summary_bin["input_time_index"])
            target_idx = np.asarray(data_summary_bin["target_time_index"])
            orig_in, orig_tg = input_idx.size, target_idx.size

            if len(input_idx) == 0 or len(target_idx) == 0:
                del data_summary[bin_name][obs_type][inst_name]
                continue

            # Get the QC filter configuration for the current instrument
            obs_cfg = observation_config[obs_type][inst_name]
            qc_filters = obs_cfg.get("qc_filters") or obs_cfg.get("qc")

            # Apply quality control based on the instrument name
            if qc_filters:
                print(f"Applying QC for {inst_name}...")
                valid_input_mask = np.ones(len(input_idx), dtype=bool)
                valid_target_mask = np.ones(len(target_idx), dtype=bool)

                for var, cfg in qc_filters.items():
                    # range handling
                    rng = None
                    if isinstance(cfg, dict):
                        rng = cfg.get("range", cfg.get("valid_range"))
                    elif isinstance(cfg, (list, tuple)) and len(cfg) == 2:
                        rng = cfg

                    if rng is not None:
                        if var not in z:
                            print(f"[QC WARNING] '{var}' not in z; skipping range filter.")
                        else:
                            lo, hi = rng
                            in_vals = z[var][input_idx]
                            tg_vals = z[var][target_idx]
                            valid_input_mask &= (in_vals >= lo) & (in_vals <= hi)
                            valid_target_mask &= (tg_vals >= lo) & (tg_vals <= hi)

                    # QM flags
                    if isinstance(cfg, dict) and "qm_flag_col" in cfg and "keep" in cfg:
                        flag_col = cfg["qm_flag_col"]
                        if flag_col in z:
                            in_flags = z[flag_col][input_idx]
                            tg_flags = z[flag_col][target_idx]
                            has_valid = (in_flags >= 0).any() or (tg_flags >= 0).any()
                            if has_valid:
                                keep_set = set(cfg["keep"])
                                valid_input_mask &= np.isin(in_flags, list(keep_set)) | (in_flags < 0)
                                valid_target_mask &= np.isin(tg_flags, list(keep_set)) | (tg_flags < 0)
                        else:
                            print(f"[QC] {inst_name}: no valid {flag_col}; skipping QM filter")

                input_idx = input_idx[valid_input_mask]
                target_idx = target_idx[valid_target_mask]
                print(f"[{bin_name}][{inst_name}] QC kept {input_idx.size}/{orig_in} (input), {target_idx.size}/{orig_tg} (target)")

                if input_idx.size == 0 or target_idx.size == 0:
                    del data_summary[bin_name][obs_type][inst_name]
                    continue

            # --- Load ALL Raw Data ---
            feat_keys = observation_config[obs_type][inst_name]["features"]
            meta_keys = observation_config[obs_type][inst_name]["metadata"]

            input_features_raw = np.column_stack([z[k][input_idx] for k in feat_keys]).astype(np.float32)
            input_metadata_raw = np.column_stack([z[k][input_idx] for k in meta_keys]).astype(np.float32)
            input_lat_raw = z["latitude"][input_idx]
            input_lon_raw = z["longitude"][input_idx]
            input_times_raw = z["time"][input_idx]

            target_features_raw = np.column_stack([z[k][target_idx] for k in feat_keys]).astype(np.float32)
            target_metadata_raw = np.column_stack([z[k][target_idx] for k in meta_keys]).astype(np.float32)
            target_lat_raw = z["latitude"][target_idx]
            target_lon_raw = z["longitude"][target_idx]
            target_times_raw = z["time"][target_idx]

            # --- Replace Fill Values with NaN ---
            FILL_VALUE = 3.402823e38
            input_features_raw[input_features_raw >= FILL_VALUE] = np.nan
            input_metadata_raw[input_metadata_raw >= FILL_VALUE] = np.nan
            target_features_raw[target_features_raw >= FILL_VALUE] = np.nan
            target_metadata_raw[target_metadata_raw >= FILL_VALUE] = np.nan

            # Build masks to drop rows with bad *metadata* (keep NaNs in target features for masking)
            # INPUT: we still require both features+metadata to be present (as before)
            input_combined = np.concatenate([input_features_raw, input_metadata_raw], axis=1)
            valid_input_mask = ~np.isnan(input_combined).any(axis=1)

            # TARGET: require metadata only; allow per-channel NaNs in features
            valid_target_mask_meta = ~np.isnan(target_metadata_raw).any(axis=1)

            # Apply masks to arrays
            input_idx = input_idx[valid_input_mask]
            input_features_raw = input_features_raw[valid_input_mask]
            input_metadata_raw = input_metadata_raw[valid_input_mask]
            input_lat_raw = input_lat_raw[valid_input_mask]
            input_lon_raw = input_lon_raw[valid_input_mask]
            input_times_clean = input_times_raw[valid_input_mask]

            target_idx = target_idx[valid_target_mask_meta]
            target_features_raw = target_features_raw[valid_target_mask_meta]
            target_metadata_raw = target_metadata_raw[valid_target_mask_meta]
            target_lat_raw = target_lat_raw[valid_target_mask_meta]
            target_lon_raw = target_lon_raw[valid_target_mask_meta]
            target_times_clean = target_times_raw[valid_target_mask_meta]

            # If after filtering, any array is empty, skip this bin
            if input_features_raw.shape[0] == 0 or target_features_raw.shape[0] == 0:
                del data_summary[bin_name][obs_type][inst_name]
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
                    (timestamp.timetuple().tm_yday - 1 +
                     (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400) / 365.24219
                    for timestamp in input_timestamps
                ]
            )[:, None]
            input_time_fraction = np.array([(ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400 for ts in input_timestamps])
            input_sin_time = np.sin(2 * np.pi * input_time_fraction)[:, None]
            input_cos_time = np.cos(2 * np.pi * input_time_fraction)[:, None]

            target_timestamps = pd.to_datetime(target_times_clean, unit="s")
            target_time_fraction = np.array([(ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400 for ts in target_timestamps])
            target_sin_time = np.sin(2 * np.pi * target_time_fraction)[:, None]
            target_cos_time = np.cos(2 * np.pi * target_time_fraction)[:, None]

            # ---------------- Normalization ----------------
            if obs_type == "satellite":
                all_features_raw = np.concatenate([input_features_raw, target_features_raw], axis=0)
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

                # Normalize to encode input metadata angles
                input_metadata_rad = np.deg2rad(input_metadata_raw)
                input_metadata_cos = np.cos(input_metadata_rad)
                input_metadata = input_metadata_cos
                # Normalize target metadata angles for the decoder
                target_metadata_rad = np.deg2rad(target_metadata_raw)
                target_metadata_cos = np.cos(target_metadata_rad)

                input_features_final = np.column_stack(
                    [
                        input_sin_lat, input_cos_lat,
                        input_sin_lon, input_cos_lon,
                        input_sin_time, input_cos_time,
                        input_dayofyear,
                        input_metadata,
                        input_features_norm,
                    ]
                )
                scan_angle = target_metadata_cos[:, 0:1]
                target_features_final = target_features_norm
                target_metadata = np.column_stack([lat_rad_target, lon_rad_target, target_metadata_cos])

            else:
                # --------- CONVENTIONAL: NaN-safe per-channel standardization + mask ---------
                # Compute per-channel mean/std over combined input+target, ignoring NaNs in target
                combined = np.vstack([input_features_raw, target_features_raw])  # may include NaNs (from target)
                means = np.nanmean(combined, axis=0)
                stds = np.nanstd(combined, axis=0)
                stds[(stds == 0) | ~np.isfinite(stds)] = 1.0
                means[~np.isfinite(means)] = 0.0

                # Standardize
                input_features_norm = (input_features_raw - means) / stds
                target_features_norm = (target_features_raw - means) / stds  # NaNs preserved here

                # Metadata normalization (unchanged)
                features_scaler = StandardScaler()
                input_metadata_norm = features_scaler.fit_transform(input_metadata_raw)
                target_scaler = StandardScaler()
                target_metadata_norm = target_scaler.fit_transform(target_metadata_raw)

                # Final input features for conventional
                input_metadata = input_metadata_norm
                input_features_final = np.column_stack(
                    [
                        input_sin_lat, input_cos_lat,
                        input_sin_lon, input_cos_lon,
                        input_sin_time, input_cos_time,
                        input_dayofyear,
                        input_metadata,
                        input_features_norm,
                    ]
                )

                # ---- Build target mask BEFORE filling NaNs ----
                target_channel_mask = ~np.isnan(target_features_norm)  # shape [N, C]
                # Replace NaNs with 0 AFTER standardization to keep tensor dense
                target_features_final = np.nan_to_num(target_features_norm, nan=0.0).astype(np.float32)

                # Target metadata for plotting
                target_metadata = np.column_stack([lat_rad_target, lon_rad_target, target_metadata_norm])
                scan_angle = np.zeros((target_features_final.shape[0], 1), dtype=np.float32)

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

            # Add per-channel mask ONLY for conventional (surface_obs/radiosonde)
            if obs_type != "satellite":
                # Ensure boolean torch tensor
                data_summary_bin["target_channel_mask"] = torch.tensor(
                    target_channel_mask.astype(bool), dtype=torch.bool
                )

            print(f"[{bin_name}] input_features_final shape: {input_features_final.shape}")
            print(f"[{bin_name}] target_features_final shape: {target_features_final.shape}")

        if not data_summary[bin_name].get(obs_type):  # all instruments removed
            del data_summary[bin_name][obs_type]

    return data_summary
