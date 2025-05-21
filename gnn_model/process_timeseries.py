import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from timing_utils import timing_resource_decorator


@timing_resource_decorator
def organize_bins_times(z_dict, start_date, end_date, observation_config, bin_size='12h'):
    """
    Organizes satellite observation times into time bins and creates input-target pairs
    for time-series prediction.

    Args:
        z_dict (dict): Dictionary containing observation data for different types
        start_date (datetime): Start date for filtering data
        end_date (datetime): End date for filtering data
        observation_config (dict): Configuration for different observation types
        bin_size (str, optional): Size of time bins. Accepts pandas offset strings:
            - 'h' or 'H': hours (e.g., '6h' for 6 hours)
            Default is '12h' (12 hours)

    Returns:
        dict: A dictionary where each key represents a time bin (e.g., 'bin1', 'bin2') and
              contains input-target time indices and corresponding timestamps.

    Raises:
        ValueError: If bin_size format is invalid
    """
    # Validate bin_size format
    valid_units = ['h', 'H']
    if not any(bin_size.endswith(unit) for unit in valid_units):
        raise ValueError(
            f"Invalid bin_size format: {bin_size}\n"
            f"Must end with one of: {valid_units}\n"
            f"Examples: '6h' for 6 hours")
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
                available_sats = z["satelliteId"][:]  # TODO other ways to get available sats
                assert isinstance(sat_ids, list), (
                    f"Configuration error: satellite IDs must be a list, got {type(sat_ids)} instead.\n"
                    f"Key: {key}, Value: {sat_ids}")

                invalid_sats = [sid for sid in sat_ids if sid not in available_sats]
                if invalid_sats:
                    raise ValueError(
                        f"Error in {obs_type} configuration for {key}:\n"
                        f"Invalid satellite IDs found: {invalid_sats}\n"
                        f"Available satellite IDs: {np.unique(available_sats)}\n"
                        f"Please check your observation configuration.")

                sat_ids_mask = np.isin(z["satelliteId"][:], sat_ids)
                selected_times = np.where(time_cond & sat_ids_mask)[0]
            else:
                selected_times = np.where(time_cond)[0]

            df = pd.DataFrame({"time": time[selected_times], "zar_time": z["time"][selected_times],
                               "index": selected_times})
            df["time_bin"] = df["time"].dt.floor(bin_size)

            # Sort by time
            df = df.sort_values(by="zar_time")

            unique_bins = df["time_bin"].unique()
            print("Filtered observation times:")
            print("  - Start:", df["time"].min())
            print("  - End:", df["time"].max())
            print("Unique time bins:", unique_bins)
            print(df["time_bin"].value_counts().sort_index())  # show how full each bin is

            for i in range(len(unique_bins) - 1):  # Exclude last bin (no target)
                bin_name = f"bin{i+1}"
                input_indices = df[df["time_bin"] == unique_bins[i]]["index"].values
                target_indices = df[df["time_bin"] == unique_bins[i + 1]]["index"].values

                # Initialize nested dictionaries if they don't exist
                if bin_name not in data_summary:
                    data_summary[bin_name] = {}
                if obs_type not in data_summary[bin_name]:
                    data_summary[bin_name][obs_type] = {}

                data_summary[bin_name][obs_type][key] = {
                    "input_time": unique_bins[i],       # datetime
                    "target_time": unique_bins[i + 1],
                    "input_time_index": input_indices,
                    "target_time_index": target_indices,
                }
            print(f"Created {len(data_summary)} input-target bin pairs.")
    return data_summary


@timing_resource_decorator
def extract_features(z_dict, data_summary, observation_config):
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

    for bin_name in data_summary.keys():  # Process bins in order
        print(f"\nProcessing {bin_name}...")
        for obs_type in data_summary[bin_name].keys():
            for inst_name in data_summary[bin_name][obs_type].keys():
                print(f'obs: {obs_type}: {inst_name}')
                z = z_dict[obs_type][inst_name]
                data_summary_bin = data_summary[bin_name][obs_type][inst_name]
                input_idx = data_summary_bin["input_time_index"]
                target_idx = data_summary_bin["target_time_index"]

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
                     (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second) / 86400) / 365.24219
                    for timestamp in input_timestamps])[:, None]

                if obs_type == "satellite":
                    metadata_keys = observation_config[obs_type][inst_name]["metadata"]
                    metadata_input = np.column_stack([z[key][input_idx] for key in metadata_keys])
                    metadata_target = np.column_stack([z[key][target_idx] for key in metadata_keys])

                feature_input = np.column_stack([z[key][input_idx]
                                                 for key in observation_config[obs_type][inst_name]["features"]])
                feature_target = np.column_stack([z[key][target_idx]
                                                  for key in observation_config[obs_type][inst_name]["features"]])

                # === Normalize features ===
                if obs_type == "satellite":
                    input_features_orig = np.column_stack([
                            sin_lat,
                            cos_lat,
                            sin_lon,
                            cos_lon,
                            input_dayofyear,
                            metadata_input,
                            feature_input,
                        ])
                else:
                    input_features_orig = np.column_stack([
                        sin_lat,
                        cos_lat,
                        sin_lon,
                        cos_lon,
                        input_dayofyear,
                        feature_input,
                    ])
                input_scaler = MinMaxScaler()
                input_features_norm = input_scaler.fit_transform(input_features_orig)
                target_features_orig = feature_target
                target_scaler = MinMaxScaler()
                target_features_norm = target_scaler.fit_transform(target_features_orig)

                # === Input Feature data ===
                input_features_final = np.column_stack([
                        sin_lat,
                        cos_lat,
                        sin_lon,
                        cos_lon,
                        input_dayofyear,
                        input_features_norm,
                    ])

                # === Metadata ===
                if obs_type == "satellite":
                    input_metadata = np.column_stack([
                            lat_rad_input,
                            lon_rad_input,
                            metadata_input
                        ])
                    target_metadata = np.column_stack([
                            lat_rad_target,
                            lon_rad_target,
                            metadata_target
                        ])
                else:
                    input_metadata = np.column_stack([
                            lat_rad_input,
                            lon_rad_input,
                        ])
                    target_metadata = np.column_stack([
                            lat_rad_target,
                            lon_rad_target,
                        ])

                # === Save ===
                data_summary_bin["input_features_final"] = torch.tensor(input_features_final, dtype=torch.float32)
                data_summary_bin["target_features_final"] = torch.tensor(target_features_norm, dtype=torch.float32)
                data_summary_bin["input_metadata"] = torch.tensor(input_metadata, dtype=torch.float32)
                data_summary_bin["target_metadata"] = torch.tensor(target_metadata, dtype=torch.float32)
                # Store min/max values for later unnormalization
                data_summary_bin["target_scaler_min"] = target_scaler.data_min_
                data_summary_bin["target_scaler_max"] = target_scaler.data_max_

                # Save lat/lon degrees separately for CSV and evaluation
                data_summary_bin["input_lat_deg"] = z["latitude"][input_idx]
                data_summary_bin["input_lon_deg"] = z["longitude"][input_idx]
                data_summary_bin["target_lat_deg"] = z["latitude"][target_idx]
                data_summary_bin["target_lon_deg"] = z["longitude"][target_idx]

                print(f"[{bin_name}] input_features_final shape: {input_features_final.shape}")
                print(f"[{bin_name}] target_features_final shape: {target_features_norm.shape}")
    return data_summary


def flatten_data_summary(data_summary):
    """Flatten data_summary from extract_features by padding missing columns with zeros
    and adding instrument IDs as additional features.
    
    Args:
        data_summary (dict): Dictionary of bin data from extract_features
        
    Returns:
        dict: Flattened data with consistent features across all bins and instrument IDs,
              along with an instrument mapping dictionary
    """
    # Find maximum feature dimensions across all bins and collect unique instruments
    max_input_features = 0
    max_target_features = 0
    max_input_metadata = 0
    max_target_metadata = 0
    unique_instruments = set()
    
    # First pass: find maximum dimensions and collect instruments
    for bin_name, bin_data in data_summary.items():
        for obs_type in bin_data.keys():
            for inst_name in bin_data[obs_type].keys():
                unique_instruments.add((obs_type, inst_name))
                curr_data = bin_data[obs_type][inst_name]
                
                if 'input_features_final' in curr_data:
                    max_input_features = max(max_input_features, curr_data['input_features_final'].shape[1])
                if 'target_features_final' in curr_data:
                    max_target_features = max(max_target_features, curr_data['target_features_final'].shape[1])
                if 'input_metadata' in curr_data:
                    max_input_metadata = max(max_input_metadata, curr_data['input_metadata'].shape[1])
                if 'target_metadata' in curr_data:
                    max_target_metadata = max(max_target_metadata, curr_data['target_metadata'].shape[1])
    
    # Create instrument ID mapping
    instrument_mapping = {}
    for idx, (obs_type, inst_name) in enumerate(sorted(unique_instruments)):
        instrument_mapping[f"{obs_type}_{inst_name}"] = idx
    
    num_instruments = len(instrument_mapping)
    flattened_data = {}
    
    # Second pass: pad and flatten
    for bin_name, bin_data in data_summary.items():
        bin_flat = {}
         # Create one-hot encoding for instrument ID
        input_inst_encoding = torch.tensor([])
        target_inst_encoding = torch.tensor([])
        
        for obs_type in bin_data.keys():
            for inst_name in bin_data[obs_type].keys():
                curr_data = bin_data[obs_type][inst_name]
                inst_id = instrument_mapping[f"{obs_type}_{inst_name}"]
                
                # Pad input features and add instrument encoding
                if 'input_features_final' in curr_data:
                    current_features = curr_data['input_features_final']
                    if current_features.shape[1] < max_input_features:
                        padding = torch.zeros(
                            (current_features.shape[0], max_input_features - current_features.shape[1]),
                            dtype=current_features.dtype,
                            device=current_features.device
                        )
                        padded_features = torch.cat([current_features, padding], dim=1)
                    else:
                        padded_features = current_features
                    
                    # Add instrument encodingt
                    inst_tensor = torch.full((current_features.shape[0], 1), 
                                             fill_value=instrument_mapping[inst_id],
                                             dtype=current_features.dtype,
                                             device=current_features.device)
                    input_inst_encoding = 
                    bin_flat['input_features_final'] = torch.cat([padded_features, inst_tensor], dim=1)
                
                # Pad target features
                if 'target_features_final' in curr_data:
                    current_features = curr_data['target_features_final']
                    if current_features.shape[1] < max_target_features:
                        padding = torch.zeros(
                            (current_features.shape[0], max_target_features - current_features.shape[1]),
                            dtype=current_features.dtype,
                            device=current_features.device
                        )
                        bin_flat['target_features_final'] = torch.cat([current_features, padding], dim=1)
                    else:
                        bin_flat['target_features_final'] = current_features
                
                # Pad input metadata
                if 'input_metadata' in curr_data:
                    current_metadata = curr_data['input_metadata']
                    if current_metadata.shape[1] < max_input_metadata:
                        padding = torch.zeros(
                            (current_metadata.shape[0], max_input_metadata - current_metadata.shape[1]),
                            dtype=current_metadata.dtype,
                            device=current_metadata.device
                        )
                        bin_flat['input_metadata'] = torch.cat([current_metadata, padding], dim=1)
                    else:
                        bin_flat['input_metadata'] = current_metadata
                
                # Pad target metadata
                if 'target_metadata' in curr_data:
                    current_metadata = curr_data['target_metadata']
                    if current_metadata.shape[1] < max_target_metadata:
                        padding = torch.zeros(
                            (current_metadata.shape[0], max_target_metadata - current_metadata.shape[1]),
                            dtype=current_metadata.dtype,
                            device=current_metadata.device
                        )
                        bin_flat['target_metadata'] = torch.cat([current_metadata, padding], dim=1)
                    else:
                        bin_flat['target_metadata'] = current_metadata
                
                # Copy non-tensor data directly
                for key in ['target_scaler_min', 'target_scaler_max', 'input_lat_deg',
                           'input_lon_deg', 'target_lat_deg', 'target_lon_deg']:
                    if key in curr_data:
                        bin_flat[key] = curr_data[key]
        
        flattened_data[bin_name] = bin_flat
    
    # Add instrument mapping to the flattened data
    flattened_data['instrument_mapping'] = instrument_mapping
    
    return flattened_data
