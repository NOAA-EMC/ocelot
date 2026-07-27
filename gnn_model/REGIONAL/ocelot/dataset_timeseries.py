from collections import OrderedDict

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from ocelot.timing_utils import timing_resource_decorator
from ocelot.logger import logger

import pyarrow.parquet as pq

from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import os
import numpy as np
from ocelot.utils import DEFAULT_DTYPE
from ocelot.constants import (
    FEATURES_NORM_OUTLIER_THRESHOLD,
    SECONDS_PER_DAY,
    YEAR_PROGRESS_DAYS,
)


import numpy as np
from typing import Literal

def _subsample_by_mode(
    indices: np.ndarray,
    mode: Literal["stride", "random", "none"],
    stride: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Efficiently return a **sorted**, subsampled view of `indices` according to `mode`.

    Parameters
    ----------
    indices : np.ndarray
        Input 1D array of indices to subsample. Does not need to be sorted.
    mode : {"stride", "random", "none"}
        Subsampling strategy:
        - "stride": keep every `stride`-th element (after sorting)
        - "random": keep approximately 1/`stride` of elements, chosen uniformly at random (no replacement)
        - "none"  : keep all elements
    stride : int
        Subsampling stride or inverse sampling fraction. Must be > 0.
    seed : int | None, optional
        Random seed for reproducibility when `mode="random"`.

    Returns
    -------
    np.ndarray
        Sorted subsampled indices.
    """
    idx = np.asarray(indices)
    n = idx.size
    if n == 0:
        return idx

    stride = max(1, int(stride))

    # Fast path — skip unnecessary computation
    if mode == "none" or stride == 1:
        # Sort only once if needed
        return idx if np.all(idx[:-1] <= idx[1:]) else np.sort(idx)

    if mode == "stride":
        # Sort once, then stride efficiently
        sorted_idx = np.sort(idx, kind="mergesort")
        return sorted_idx[::stride]

    if mode == "random":
        k = max(1, int(np.ceil(n / stride)))
        rng = np.random.default_rng(seed)
        take = rng.choice(n, size=k, replace=False)
        # Sorting here ensures deterministic output order (ascending indices)
        return np.sort(idx.take(take))

    raise ValueError(f"Invalid subsampling mode: {mode!r}. Must be one of 'stride', 'random', or 'none'.")



def extract_features_from_df(data_summary, observation_config, feature_stats):
    """
    Loads and normalizes features from a dictionary of DataFrames for each time bin.

    This function processes data for a single time bin. It extracts features and
    metadata from the provided DataFrames, normalizes them, and attaches the
    resulting tensors back to the `data_summary` dictionary.

    Parameters:
        data_summary (dict): A dictionary where keys are observation types
            (e.g., 'satellite') and values are dictionaries mapping instrument names
            to DataFrames.
        observation_config (dict): Configuration specifying which features and
            metadata to use for each observation type.

    Returns:
        dict: Updated data_summary with feature and metadata tensors.
    """
    for obs_type in data_summary.keys():
        for inst_name in data_summary[obs_type].keys():
            data_bin = data_summary[obs_type][inst_name]
            if 'dataframe' not in data_bin or data_bin['dataframe'].empty:
                logger.warning(f"Empty dataframe for {inst_name}, skipping.")
                continue

            df = data_bin.pop('dataframe')

            # === Apply satellite ID filtering if configured ===
            sat_ids_config = observation_config[obs_type][inst_name].get('sat_ids')
            if sat_ids_config:
                possible_cols = ["sat_id", "satelliteIdentifier", "satelliteId"]
                sat_id_field = next((c for c in possible_cols if c in df.columns), None)
                if sat_id_field is None:
                    logger.warning(f"No satellite ID column found in {inst_name} for filtering.")
                else:
                    original_rows = len(df)
                    df = df[df[sat_id_field].isin(sat_ids_config)]
                    logger.info(
                        f"Applied satellite ID filtering on {inst_name}: "
                        f"kept {len(df)} of {original_rows} rows."
                    )


            # === Apply level selection if configured (e.g., for radiosonde) ===
            level_selection_config = observation_config[obs_type][inst_name].get('level_selection')
            if level_selection_config:
                filter_col = level_selection_config['filter_col']
                levels_to_keep = level_selection_config['levels']

                original_rows = len(df)
                df = df[df[filter_col].isin(levels_to_keep)].copy()
                print(f"Applied level selection on {inst_name}: kept {len(df)} of {original_rows} rows.")

            # === Drop rows with NaN in specified columns ===
            dropna_cols = observation_config[obs_type][inst_name].get('dropna_cols')
            if dropna_cols:
                original_rows = len(df)
                df =df.dropna(subset=dropna_cols)
                print(f"Dropped NaN rows for {inst_name}: kept {len(df)} of {original_rows} rows.")

            # === Drop rows matching specified values ===
            drop_rows = observation_config[obs_type][inst_name].get('drop_rows')
            if drop_rows:
                original_rows = len(df)
                for col, values in drop_rows.items():
                    df = df[~df[col].isin(values)]
                print(f"Dropped value-matched rows for {inst_name}: kept {len(df)} of {original_rows} rows.")

            # === Log column statistics ===
            logger.debug(f"\n=== Column Statistics for {inst_name} ===")
            logger.debug(f"Total rows: {len(df)}")
            logger.debug(f"Total columns: {len(df.columns)}")
            logger.debug("\nColumn-wise statistics:")
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_data = df[col]
                    logger.debug(f"  {col}:")
                    logger.debug(f"    count: {col_data.count()}, "
                          f"mean: {col_data.mean():.4f}, "
                          f"std: {col_data.std():.4f}, "
                          f"min: {col_data.min():.4f}, "
                          f"max: {col_data.max():.4f}, "
                          f"NaN: {col_data.isna().sum()}")
                else:
                    logger.debug(f"  {col}: non-numeric (dtype: {df[col].dtype})")
            logger.debug("=" * 50)

            # === Subsample the DataFrame if configured ===
            subsample_config = observation_config[obs_type][inst_name].get('subsample', {})
            subsample_mode = subsample_config.get('mode', 'none')

            if subsample_mode != 'none':
                subsample_fraction = subsample_config.get('fraction', 1.0)
                stride = 1 / subsample_fraction if subsample_fraction > 0 else float('inf')
                seed = subsample_config.get('seed')

                original_indices = df.index.to_numpy()
                subsampled_indices = _subsample_by_mode(
                    original_indices, mode=subsample_mode, stride=stride, seed=seed
                )
                df = df.loc[subsampled_indices]
                print(f"Subsampled {inst_name} from {len(original_indices)} to {len(df)} rows (mode: {subsample_mode}, fraction: {subsample_fraction})")

            # Save linear pressure for radiosonde before log conversion (level selection, range_by_level, mean_std_by_level use linear hPa)
            pressure_linear = None
            if inst_name == 'radiosonde':
                level_sel = observation_config[obs_type][inst_name].get('level_selection')
                if level_sel:
                    pressure_col_linear = level_sel.get('filter_col', 'airPressure')
                    if pressure_col_linear in df.columns:
                        pressure_linear = np.asarray(df[pressure_col_linear].values, dtype=float).copy()

            # === Log-transform air pressure if present ===
            for col in ['airPressure', 'pressure', 'pressureMeanSeaLevel_bufr']:
                if col in df.columns:
                    pressure = df[col].values
                    # Use a mask to avoid taking the log of non-positive values or NaNs
                    valid_mask = ~np.isnan(pressure) & (pressure > 0)
                    log_pressure = np.full(pressure.shape, np.nan)
                    log_pressure[valid_mask] = np.log(pressure[valid_mask])
                    df[col] = log_pressure

            # === Extract coordinates and time ===
            lat_rad = np.radians(df["latitude"].values)[:, None]
            lon_rad = np.radians(df["longitude"].values)[:, None]

            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            sin_lon = np.sin(lon_rad)
            cos_lon = np.cos(lon_rad)

            # Handle both 'time' and 'obs_time' column names
            time_col = 'obs_time' if 'obs_time' in df.columns else 'time'
            seconds_since_epoch = np.asarray(df[time_col].values, dtype=np.float64)
            ts_index = pd.to_datetime(df[time_col].values, unit='s')
            if not isinstance(ts_index, pd.DatetimeIndex):
                ts_index = pd.DatetimeIndex(ts_index)
            sec_of_day = np.mod(seconds_since_epoch, SECONDS_PER_DAY)
            doy_frac = ts_index.dayofyear.to_numpy(dtype=np.float64) - 1.0 + sec_of_day / SECONDS_PER_DAY
            two_pi = 2.0 * np.pi
            year_progress = (doy_frac / YEAR_PROGRESS_DAYS) * two_pi
            # Local solar day fraction: Greenwich + longitude as fraction of a full turn (then 2π for sin/cos)
            day_progress_greenwich = np.mod(seconds_since_epoch, SECONDS_PER_DAY) / SECONDS_PER_DAY
            longitude_deg = df["longitude"].values.astype(np.float64)
            longitude_offsets = np.deg2rad(longitude_deg) / two_pi
            # Per-row element-wise mod (avoids (N,1)+(N,) → (N,N) numpy broadcast)
            day_progress_frac = np.mod(day_progress_greenwich + longitude_offsets, 1.0).astype(np.float32)
            day_progress = day_progress_frac * two_pi
            year_progress_sin = np.sin(year_progress)
            year_progress_cos = np.cos(year_progress)
            day_progress_sin = np.sin(day_progress)
            day_progress_cos = np.cos(day_progress)
            # Cyclical time encodings in features_aux only (not in metadata tensor).
            time_meta = np.column_stack(
                [year_progress_sin, year_progress_cos, day_progress_sin, day_progress_cos]
            )

            # === Calculate derived features (e.g., wind components) ===
            feature_keys = observation_config[obs_type][inst_name]["features"]
            if 'wind_u' in feature_keys and 'wind_v' in feature_keys:
                if 'windSpeed' in df.columns and 'windDirection' in df.columns:
                    # Meteorological wind direction is 'from', so we use negative sine/cosine
                    # and convert degrees to radians.
                    wind_dir_rad = np.deg2rad(df['windDirection'].values)
                    wind_speed = df['windSpeed'].values
                    df['wind_u'] = -wind_speed * np.sin(wind_dir_rad)
                    df['wind_v'] = -wind_speed * np.cos(wind_dir_rad)
                    print(f"Calculated wind_u and wind_v for {inst_name}.")

            # === Extract features and metadata ===

            features = df[feature_keys].values
            logger.debug(f"[{inst_name}] After subsampling - df shape: {df.shape}, features shape: {features.shape}")

            # === Create validity mask and apply QC Filters ===
            # The mask is True for valid (non-NaN) data, False for NaN.
            features_valid_mask = ~np.isnan(features)
            logger.debug(f"[{inst_name}] features_valid_mask shape: {features_valid_mask.shape}")

            qc_filters = observation_config[obs_type][inst_name].get('qc_filters', {})
            if qc_filters:
                print(f"Applying QC filters for {inst_name}...")
                feature_pos_map = {key: i for i, key in enumerate(feature_keys)}

                for var, cfg in qc_filters.items():
                    pos = feature_pos_map.get(var)
                    if pos is None:
                        continue # This QC var is not in the feature list

                    # --- Range Check ---
                    if 'range' in cfg:
                        if inst_name == 'radiosonde' and 'range_by_level' in cfg:
                            # Per-level range: use linear pressure (saved before log conversion); levels are in hPa
                            level_selection_config = observation_config[obs_type][inst_name].get('level_selection')
                            pressure_col = level_selection_config.get('filter_col', 'airPressure') if level_selection_config else 'airPressure'
                            if pressure_linear is not None:
                                pressure = pressure_linear
                                range_by_level = cfg['range_by_level']
                                levels_arr = np.array(sorted(range_by_level.keys(), reverse=True))
                                diff = np.abs(pressure[:, np.newaxis] - levels_arr[np.newaxis, :])
                                level_idx = np.argmin(diff, axis=1)
                                row_levels = levels_arr[level_idx]
                                for level in range_by_level:
                                    mask = (row_levels == level)
                                    if not np.any(mask):
                                        continue
                                    lo, hi = range_by_level[level]
                                    features_valid_mask[mask, pos] &= (features[mask, pos] >= lo) & (features[mask, pos] <= hi)
                                # Rows whose nearest level is not in the dict use global range
                                known = np.isin(row_levels, list(range_by_level.keys()))
                                if not np.all(known):
                                    lo, hi = cfg['range']
                                    features_valid_mask[~known, pos] &= (features[~known, pos] >= lo) & (features[~known, pos] <= hi)
                            else:
                                lo, hi = cfg['range']
                                features_valid_mask[:, pos] &= (features[:, pos] >= lo) & (features[:, pos] <= hi)
                        else:
                            lo, hi = cfg['range']
                            features_valid_mask[:, pos] &= (features[:, pos] >= lo) & (features[:, pos] <= hi)

                    # --- Flag Check ---
                    if 'qm_flag_col' in cfg and 'keep' in cfg:
                        flag_col = cfg['qm_flag_col']
                        keep_flags = set(cfg['keep'])
                        if flag_col in df.columns:
                            flags = df[flag_col].values.astype(float)
                            # If keep_nan is set (number), replace NaN in flags with that number before applying keep
                            keep_nan_val = cfg.get('keep_nan')
                            if keep_nan_val is not None:
                                flags = np.where(np.isnan(flags), float(keep_nan_val), flags)
                            # Keep values where the flag is in the 'keep' list or where the flag itself is invalid (e.g., < 0)
                            keep_mask = np.isin(flags, list(keep_flags)) | (flags < 0)
                            features_valid_mask[:, pos] &= keep_mask

            data_bin["features_valid_mask"] = torch.tensor(features_valid_mask, dtype=torch.bool)

            # Set all invalid values (NaNs or QC failures) to 0 based on the mask
            features[~features_valid_mask] = 0



            # === Normalize features using pre-computed stats ===
            stats = feature_stats.get(inst_name, {})
            if features.shape[0] == 0:
                logger.warning("Skipping empty feature batch.")
                continue

            if inst_name == 'radiosonde' and 'mean_std_by_level' in stats:
                # Per-level mean/std: use linear pressure (saved before log conversion); levels are in hPa
                if pressure_linear is not None:
                    pressure = pressure_linear
                    by_level = stats['mean_std_by_level']
                    levels_arr = np.array(sorted(next(iter(by_level.values())).keys(), reverse=True))
                    diff = np.abs(pressure[:, np.newaxis] - levels_arr[np.newaxis, :])
                    row_levels = levels_arr[np.argmin(diff, axis=1)]
                    features_norm = np.empty_like(features, dtype=features.dtype)
                    for level in levels_arr:
                        mask = (row_levels == level)
                        if not np.any(mask):
                            continue
                        means = np.array([
                            by_level.get(key, {}).get(level, stats.get(key, [0, 1]))[0]
                            for key in feature_keys
                        ])
                        stds = np.array([
                            by_level.get(key, {}).get(level, stats.get(key, [0, 1]))[1]
                            for key in feature_keys
                        ])
                        stds = np.where(stds > 0, stds, 1.0)
                        features_norm[mask, :] = (features[mask, :] - means) / stds
                    # Rows whose level has no stats (shouldn't happen) use global
                    covered = np.zeros(features.shape[0], dtype=bool)
                    for level in levels_arr:
                        covered |= (row_levels == level)
                    if not np.all(covered):
                        means = np.array([stats.get(key, [0, 1])[0] for key in feature_keys])
                        stds = np.array([stats.get(key, [0, 1])[1] for key in feature_keys])
                        stds = np.where(stds > 0, stds, 1.0)
                        features_norm[~covered, :] = (features[~covered, :] - means) / stds
                    logger.info(f'stats used for normalize for {inst_name} (by level)')
                else:
                    means = np.array([stats.get(key, [0, 1])[0] for key in feature_keys])
                    stds = np.array([stats.get(key, [0, 1])[1] for key in feature_keys])
                    stds = np.where(stds > 0, stds, 1.0)
                    features_norm = (features - means) / stds
                    logger.info(f'stats used for normalize for {inst_name}, {means}, {stds}')
            else:
                means = np.array([stats.get(key, [0, 1])[0] for key in feature_keys])
                stds = np.array([stats.get(key, [0, 1])[1] for key in feature_keys])
                logger.info(f'stats used for normalize for {inst_name}, {means}, {stds}')
                features_scaler = StandardScaler()
                features_scaler.mean_ = means
                features_scaler.scale_ = np.where(stds > 0, stds, 1.0)
                features_norm = features_scaler.transform(features)



            features_aux = np.column_stack([sin_lat, cos_lat, sin_lon, cos_lon, time_meta])

            if obs_type == "satellite":
                metadata_keys = observation_config[obs_type][inst_name]["metadata"]
                metadata_deg = df[metadata_keys].values
                
                # === Create validity mask for metadata ===
                # The mask is True for valid (non-NaN) data, False for NaN.
                metadata_valid_mask = ~np.isnan(metadata_deg)
                data_bin["metadata_valid_mask"] = torch.tensor(metadata_valid_mask, dtype=torch.bool)
                
                metadata_rad = np.deg2rad(metadata_deg)
                features_meta_norm = np.cos(metadata_rad)
                if inst_name in ('ssmis', ):
                    # TODO might need improve here for more general cases!
                    print("SSMIS: using raw metadata without cosine transform.")
                    features_meta_norm[:, 1] = metadata_deg[:, 1]
                
                metadata = np.column_stack([lat_rad, lon_rad, features_meta_norm])
                data_bin["features_meta"] = torch.tensor(features_meta_norm, dtype=DEFAULT_DTYPE)
            else:
                # Check if metadata keys are provided in the config
                if "metadata" in observation_config[obs_type][inst_name]:
                    metadata_keys = observation_config[obs_type][inst_name]["metadata"]
                    metadata_values = df[metadata_keys].values

                    # Normalize metadata using pre-computed stats
                    meta_means = np.array([stats.get(key, [0, 1])[0] for key in metadata_keys])
                    meta_stds = np.array([stats.get(key, [0, 1])[1] for key in metadata_keys])

                    # Create and configure the scaler
                    metadata_scaler = StandardScaler()
                    metadata_scaler.mean_ = meta_means
                    metadata_scaler.scale_ = meta_stds

                    # Apply the transformation
                    features_meta_norm = metadata_scaler.transform(metadata_values)

                    metadata = np.column_stack([lat_rad, lon_rad, features_meta_norm])
                    data_bin["features_meta"] = torch.tensor(features_meta_norm, dtype=DEFAULT_DTYPE)
                else:
                    metadata = np.column_stack([lat_rad, lon_rad])

            # === Clip outliers and mark invalid: norm > threshold -> 0 and invalid ===
            invalid_high = features_norm > FEATURES_NORM_OUTLIER_THRESHOLD
            features_norm[invalid_high] = 0
            # Update valid mask: positions where norm was > 3 are now invalid
            valid_mask_np = data_bin["features_valid_mask"].numpy()
            valid_mask_np &= ~invalid_high
            data_bin["features_valid_mask"] = torch.tensor(valid_mask_np, dtype=torch.bool)

            # === Diagnostic: log per-feature valid fraction after all QC/clipping ===
            n_total = valid_mask_np.shape[0]
            per_feature = [valid_mask_np[:, fi].sum() for fi in range(len(feature_keys))]
            n_all_valid = int(np.all(valid_mask_np, axis=1).sum())
            feature_summary = ", ".join(
                f"{k}={v}/{n_total}({100*v//max(n_total,1)}%)"
                for k, v in zip(feature_keys, per_feature)
            )
            logger.warning(
                f"[QC] {inst_name}: {n_all_valid}/{n_total} obs all-features-valid | "
                f"per-feature: {feature_summary}"
            )

            # === Save Tensors ===
            data_bin["features_aux"] = torch.tensor(features_aux, dtype=DEFAULT_DTYPE)
            data_bin["features_norm"] = torch.tensor(features_norm, dtype=DEFAULT_DTYPE)
            data_bin["metadata"] = torch.tensor(metadata, dtype=DEFAULT_DTYPE)

            # Save lat/lon degrees separately for evaluation
            data_bin["lat_deg"] = df["latitude"].values
            data_bin["lon_deg"] = df["longitude"].values
            
            logger.debug(f"[{inst_name}] Final shapes - lat_deg: {len(data_bin['lat_deg'])}, "
                        f"lon_deg: {len(data_bin['lon_deg'])}, "
                        f"features_valid_mask: {data_bin['features_valid_mask'].shape}")

            logger.info(f"[{obs_type}/{inst_name}] features_norm shape: {features_norm.shape}")
    logger.info("Features extracted and normalized.")
    return data_summary


def generate_timestamps(start_mmdd, end_mmdd, year=2024, delta_time: int = 12):
    """
    Generates timestamps at a fixed hourly interval between a start and end date.

    Args:
        start_mmdd (str): Start date in 'MM/DD' format.
        end_mmdd (str): End date in 'MM/DD' format.
        year (int): The year to use.
        delta_time (int): Interval in hours between timestamps (e.g. 1, 6, or 12).

    Returns:
        List[datetime]: A list of datetime objects.
    """
    start_date = datetime.strptime(f"{year}/{start_mmdd}", "%Y/%m/%d")
    end_date = datetime.strptime(f"{year}/{end_mmdd}", "%Y/%m/%d")

    timestamps = []
    current = start_date
    # Add timedelta(days=1) to make the end date inclusive.
    while current < end_date + timedelta(days=1):
        timestamps.append(current)
        current += timedelta(hours=delta_time)

    return timestamps


def generate_binned_timestamp_list(start_mmdd, end_mmdd, year=2024, delta_time: int = 12):
    """
    Generates a list of uniquely named bins for each timestamp at the given interval.

    Args:
        start_mmdd (str): Start date in 'MM/DD' format.
        end_mmdd (str): End date in 'MM/DD' format.
        year (int): The year to use.
        delta_time (int): Interval in hours between timestamps (e.g. 1, 6, or 12).

    Returns:
        List[str]: A list of bin name strings in 'day_half=YYYY-MM-DD_HH' format.
    """
    timestamps = generate_timestamps(start_mmdd, end_mmdd, year, delta_time)
    ts_series = pd.to_datetime(timestamps)
    return ts_series.strftime("day_half=%Y-%m-%d_%H").tolist()


class ParquetDataManager:
    """
    Manages loading and processing of data from Parquet files.

    This class lazily loads and processes data for a given time bin when first
    requested, then caches the result for subsequent requests.
    """
    _NWP_CYCLES = [0, 6, 12, 18]

    def __init__(self, data_dir: str, observation_config, feature_stats, fill_values=None, delta_time: int = 12):
        self.data_dir = data_dir
        self.observation_config = observation_config
        self.feature_stats = feature_stats
        self.fill_values = fill_values if fill_values is not None else [-999, -9999, 9.96921e+36]
        self.delta_time = delta_time
        self.processed_data_cache = OrderedDict()
        self.cache_size = 4

    def _cycles_for_hour(self, hour_int: int) -> list:
        """Return cycle strings covered by a bin starting at hour_int.

        For 1h bins the hour itself is the cycle; for 6h/12h bins cycles are
        the NWP model run times (00, 06, 12, 18) that fall within the bin window.
        """
        if self.delta_time == 1:
            return [f"{hour_int:02d}"]
        cycles = [c for c in self._NWP_CYCLES if hour_int <= c < hour_int + self.delta_time]
        return [f"{c:02d}" for c in cycles]

    def get_data_for_bin(self, bin_name: str):
        """
        Retrieves processed data for a specific bin, loading from Parquet files.
        Caches the processed data for the bin.
        """
        if bin_name in self.processed_data_cache:
            print(f'Using cache for bin {bin_name}')
            return self.processed_data_cache[bin_name]

        print(f'Processing bin: {bin_name}')
        data_summary_bin = {}
        print(f'observation_config: {self.observation_config.keys()}')
        for obs_type in self.observation_config.keys():
            data_summary_bin[obs_type] = {}
            for inst_name in self.observation_config[obs_type].keys():
                file_base = self.observation_config[obs_type][inst_name].get('zarr_name', inst_name)
                # Extract date and hour from bin_name (format: day_half=YYYY-MM-DD_HH)
                bin_parts = bin_name.split('=')[1]  # Get YYYY-MM-DD_HH
                date_part = bin_parts.rsplit('_', 1)[0]  # Get YYYY-MM-DD
                hour = bin_parts.rsplit('_', 1)[1]  # Get HH
                year = date_part.split('-')[0]
                
                # Determine which NWP cycles to read based on the bin hour and delta_time.
                cycles = self._cycles_for_hour(int(hour))
                print(f'hour: {hour}, cycles: {cycles}, date: {date_part}, year: {year},bin_name: {bin_name}')
                # Try reading from new two-level partition structure
                dfs = []
                new_structure_found = False
                for cycle in cycles:
                    dataset_path = os.path.join(
                        self.data_dir, 
                        f'{file_base}_{year}.parquet',
                        f'date={date_part}',
                        f'cycle={cycle}'
                    )
                    print(dataset_path)
                    try:
                        df_cycle = pd.read_parquet(dataset_path)
                        # Replace fill values with NaN
                        if self.fill_values:
                            df_cycle = df_cycle.replace(self.fill_values, np.nan)
                        dfs.append(df_cycle)
                        new_structure_found = True
                    except Exception as e:
                        print(f"Info: Could not load from new structure {dataset_path}. Error: {e}")
                        continue
                
                # If new structure not found, fall back to old single-partition structure
                if not new_structure_found:
                    dataset_path = os.path.join(self.data_dir, file_base + '.parquet', bin_name)
                    print(f"Trying old structure: {dataset_path}")
                    try:
                        df = pd.read_parquet(dataset_path)
                        # Replace fill values with NaN
                        if self.fill_values:
                            df = df.replace(self.fill_values, np.nan)
                        dfs = [df]
                    except Exception as e:
                        print(f"Warning: Could not load Parquet data from {dataset_path} for bin {bin_name}. Error: {e}")
                        continue
                
                if not dfs:
                    print(f"Warning: No data loaded for {inst_name} in bin {bin_name}")
                    continue
                
                try:
                    # Combine data from all partitions (or single partition for old structure)
                    df = pd.concat(dfs, ignore_index=True)
                    
                    if inst_name == 'raw_surface_obs':
                        df_clean = df.dropna(subset=["latitude", "longitude", "airTemperature"])
                    else:
                        df_clean = df.dropna(subset=["latitude", "longitude"])

                    if df_clean.empty:
                        print(f"Warning: No data for obs_type {obs_type} in bin {bin_name}")
                        continue

                    # Instruments are columns in the dataframe, so we distribute the same dataframe
                    # to each instrument configured for this obs_type.
                    data_summary_bin[obs_type][inst_name] = {'dataframe': df_clean}

                except Exception as e:
                    print(f"Warning: Could not process data for {inst_name} in bin {bin_name}. Error: {e}")
                    continue

        empty_list = [
            (obs_type_name, inst_name)
            for obs_type_name, obs_dict in data_summary_bin.items()
            for inst_name, entry in obs_dict.items()
            if entry['dataframe'].empty
        ]

        any_empty = len(empty_list) > 0

        # if any_empty:
        #     print("At least one DataFrame is empty:", empty_list)
        #     return None
        # else:
        #     print("All DataFrames have data")
        #     if not data_summary_bin:
        #         print(f"No data loaded for bin {bin_name}")
        #         return None

        # Extract features for the loaded data
        processed_bin_data = extract_features_from_df(data_summary_bin, self.observation_config, self.feature_stats)

        # Cache the processed data for this bin
        self.processed_data_cache[bin_name] = processed_bin_data

        # Enforce the cache size limit (FIFO)
        if len(self.processed_data_cache) > self.cache_size:
            self.processed_data_cache.popitem(last=False)

        return processed_bin_data
