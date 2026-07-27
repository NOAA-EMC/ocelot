import traceback
import torch
import numpy as np
import os
from ocelot.vis import (
    plot_hist, plot_map, plot_scatter_with_diagonal,
    plot_vertical_profile
)
from ocelot.observation_config import load_observation_config


def load_debug_data(filepath):
    """
    Loads the debug data file.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None

    print(f"Loading data from {filepath}...")
    data = torch.load(filepath, weights_only=False)
    return data['batch'], data['predictions']


def get_valid_mask(batch, node_type, channel_idx):
    if hasattr(batch[node_type], "valid_mask"):
        return batch[node_type].valid_mask[:, channel_idx]
    y = batch[node_type].y
    return torch.ones(y.shape[0], dtype=torch.bool, device=y.device)


def extract_channel_data(
    batch, predictions, target_node_type, channel_idx, add_background=True
):
    """
    Extracts valid ground truth and prediction data for a specific channel.

    Args:
        batch: HeteroData batch containing ground truth
        predictions: Dictionary of predictions
        target_node_type: Node type to extract (e.g., 'satellite_atms_target')
        channel_idx: Channel index
        add_background: If True (default), reconstruct full analysis values
                       by adding back the background (ges) for state
                       variables using the residual formulation. If False,
                       leave values in residual/increment space.

    Returns:
        tuple: (valid_ground_truth, valid_prediction, valid_mask,
                valid_lat, valid_lon) as numpy arrays.
               valid_lat and valid_lon will be None if pos is not
               available
    """
    node = batch[target_node_type]
    ground_truth = node.y
    predicted_values = predictions[target_node_type]

    # If the increment was rescaled by offline stats (--normalize_increment), undo
    # that rescaling first so both tensors are back in ges-comparable units:
    # anal_norm - ges_norm = y * increment_std + increment_mean
    if hasattr(node, 'increment_std') and node.increment_std is not None:
        increment_mean = node.increment_mean
        increment_std = node.increment_std
        ground_truth = ground_truth * increment_std + increment_mean
        predicted_values = predicted_values * increment_std + increment_mean

    # State variables use residual formulation: y = anal - ges (DA increment).
    # Reconstruct full analysis values so obs_stats (full-field scale) apply
    # correctly.
    if add_background and hasattr(node, 'ges') and node.ges is not None:
        background = node.ges
        ground_truth = ground_truth + background
        predicted_values = predicted_values + background

    valid_mask = get_valid_mask(batch, target_node_type, channel_idx)
    valid_prediction = (
        predicted_values[:, channel_idx][valid_mask].numpy().flatten()
    )
    valid_ground_truth = (
        ground_truth[:, channel_idx][valid_mask].numpy().flatten()
    )

    # Extract lat/lon from batch.pos if available
    valid_lat = None
    valid_lon = None
    if hasattr(node, 'pos') and node.pos is not None:
        pos = node.pos.numpy()
        if pos.shape[1] >= 2:
            mask_np = valid_mask.numpy()
            valid_lon = pos[mask_np, 0]  # First column is longitude
            valid_lat = pos[mask_np, 1]  # Second column is latitude

    return valid_ground_truth, valid_prediction, valid_mask, valid_lat, valid_lon


def iterate_channels_and_plot(
    batch, predictions, instrument_name, plot_func, obs_config=None,
    add_background=True, **kwargs
):
    """
    Iterates over all channels for a given instrument and applies
    the plotting function.

    Args:
        batch: HeteroData batch
        predictions: Dictionary of predictions
        instrument_name: Name of the instrument (e.g., 'satellite_atms')
        plot_func: Plotting function to apply to each channel
        obs_config: Observation configuration dictionary containing
                   'features' list
        add_background: If True (default), reconstruct full analysis
                       values by adding back the background (ges).
                       If False, plot in residual/increment space.
        **kwargs: Additional arguments to pass to plot_func
    """
    target_node_type = f'{instrument_name}_target'

    if target_node_type in predictions:
        target_node_types = [target_node_type]
        print(f"Plotting for instrument: {instrument_name}")
    else:
        print(
            f"Error: Ground truth for '{target_node_type}' not found "
            f"in predictions: {predictions.keys()}."
        )
        return

    for target_node_type in target_node_types:
        if target_node_type not in batch.y_dict:
            print(
                f"Error: Ground truth for '{target_node_type}' not "
                f"found in batch data."
            )
            return

        if target_node_type not in predictions:
            print(f"Error: Predictions for '{target_node_type}' not found.")
            return

        num_chs = predictions[target_node_type].shape[1]

        # Get feature names from config if available
        feature_names = None
        if obs_config and 'features' in obs_config:
            feature_names = obs_config['features']
            if len(feature_names) != num_chs:
                print(
                    f"Warning: Number of features ({len(feature_names)}) "
                    f"doesn't match number of channels ({num_chs})"
                )
                feature_names = None

        # Special case for radiosonde: group by airPressure
        node = batch[target_node_type]
        is_radiosonde = (
            'radiosonde' in instrument_name.lower() and
            hasattr(node, 'x') and node.x is not None
        )
        if is_radiosonde:
            # Get airPressure from metadata (stored in .x)
            # Find the index of airPressure in metadata
            airpressure_idx = 0  # Default to first column
            if obs_config and 'metadata' in obs_config:
                metadata_keys = obs_config.get('metadata', [])
                if 'airPressure' in metadata_keys:
                    airpressure_idx = metadata_keys.index('airPressure')

            airpressure = node.x[:, airpressure_idx].numpy()

            # Group by airPressure values (use unique values or bin them)
            unique_pressures = np.unique(airpressure)
            # Filter out NaN values
            unique_pressures = unique_pressures[~np.isnan(unique_pressures)]

            print(
                f"Grouping radiosonde data by airPressure: "
                f"{len(unique_pressures)} unique pressure levels"
            )

            # If plotting vertical profiles, call with all data first
            # (before grouping)
            if plot_func == _plot_vertical_profile_channel:
                # Extract all channel data without filtering by pressure
                if feature_names:
                    for i, feature_name in enumerate(feature_names):
                        (valid_ground_truth, valid_prediction, valid_mask,
                         valid_lat, valid_lon) = extract_channel_data(
                            batch, predictions, target_node_type, i,
                            add_background=add_background
                        )
                        plot_func(
                            batch, target_node_type, i,
                            valid_ground_truth, valid_prediction,
                            valid_mask, valid_lat, valid_lon,
                            instrument_name, feature_name=feature_name,
                            obs_config=obs_config, **kwargs
                        )
                else:
                    for i in range(num_chs):
                        feature_name = f"ch{i}"
                        (valid_ground_truth, valid_prediction, valid_mask,
                         valid_lat, valid_lon) = extract_channel_data(
                            batch, predictions, target_node_type, i,
                            add_background=add_background
                        )
                        plot_func(
                            batch, target_node_type, i,
                            valid_ground_truth, valid_prediction,
                            valid_mask, valid_lat, valid_lon,
                            instrument_name, feature_name=feature_name,
                            obs_config=obs_config, **kwargs
                        )
                return  # Don't do grouping for vertical profiles

            # Plot for each pressure group
            for pressure_val in unique_pressures:
                pressure_mask = (airpressure == pressure_val)
                pressure_mask_torch = torch.from_numpy(pressure_mask)

                # Create a filtered batch for this pressure level
                filtered_batch = batch.clone()
                filtered_batch[target_node_type].y = (
                    batch[target_node_type].y[pressure_mask_torch]
                )
                filtered_batch[target_node_type].x = (
                    batch[target_node_type].x[pressure_mask_torch]
                )
                if (hasattr(node, 'pos') and node.pos is not None):
                    filtered_batch[target_node_type].pos = (
                        node.pos[pressure_mask_torch]
                    )
                if hasattr(node, 'valid_mask'):
                    filtered_batch[target_node_type].valid_mask = (
                        node.valid_mask[pressure_mask_torch]
                    )

                # Create filtered predictions
                filtered_predictions = {
                    target_node_type: (
                        predictions[target_node_type][pressure_mask_torch]
                    )
                }

                # Update kwargs to include pressure level info
                plot_kwargs = kwargs.copy()
                plot_kwargs['pressure_level'] = pressure_val
                plot_kwargs['group_label'] = (
                    f'airPressure_{pressure_val:.2f}'
                )

                # Iterate over channels for this pressure group
                if feature_names:
                    for i, feature_name in enumerate(feature_names):
                        (valid_ground_truth, valid_prediction, valid_mask,
                         valid_lat, valid_lon) = extract_channel_data(
                            filtered_batch, filtered_predictions,
                            target_node_type, i, add_background=add_background
                        )
                        plot_func(
                            filtered_batch, target_node_type, i,
                            valid_ground_truth, valid_prediction,
                            valid_mask, valid_lat, valid_lon,
                            instrument_name, feature_name=feature_name,
                            **plot_kwargs
                        )
                else:
                    for i in range(num_chs):
                        feature_name = f"ch{i}"
                        (valid_ground_truth, valid_prediction, valid_mask,
                         valid_lat, valid_lon) = extract_channel_data(
                            filtered_batch, filtered_predictions,
                            target_node_type, i, add_background=add_background
                        )
                        plot_func(
                            filtered_batch, target_node_type, i,
                            valid_ground_truth, valid_prediction,
                            valid_mask, valid_lat, valid_lon,
                            instrument_name, feature_name=feature_name,
                            **plot_kwargs
                        )
        else:
            # Standard case: iterate over all channels without grouping
            if feature_names:
                for i, feature_name in enumerate(feature_names):
                    (valid_ground_truth, valid_prediction, valid_mask,
                     valid_lat, valid_lon) = extract_channel_data(
                        batch, predictions, target_node_type, i,
                        add_background=add_background
                    )
                    plot_func(
                        batch, target_node_type, i,
                        valid_ground_truth, valid_prediction, valid_mask,
                        valid_lat, valid_lon, instrument_name,
                        feature_name=feature_name, **kwargs
                    )
            else:
                for i in range(num_chs):
                    feature_name = f"ch{i}"
                    (valid_ground_truth, valid_prediction, valid_mask,
                     valid_lat, valid_lon) = extract_channel_data(
                        batch, predictions, target_node_type, i,
                        add_background=add_background
                    )
                    plot_func(
                        batch, target_node_type, i,
                        valid_ground_truth, valid_prediction, valid_mask,
                        valid_lat, valid_lon, instrument_name,
                        feature_name=feature_name, **kwargs
                    )


def plot_predictions_vs_ground_truth(
    batch, predictions, instrument_name='satellite_atms',
    exp_name='debug_analysis', obs_stats=None, obs_config=None,
    version='version_0', add_background=True, **kwargs
):
    """
    Creates a scatter plot of predictions vs. ground truth for a
    specific instrument.

    Args:
        batch: HeteroData batch containing ground truth
        predictions: Dictionary of predictions
        instrument_name: Name of the instrument to plot
        exp_name: Experiment name for output directory
        obs_stats: dict mapping feature_name -> [mean, std] (optional)
        obs_config: Observation configuration dictionary (optional)
        version: Version name for output directory (optional)
        add_background: If True (default), plot full analysis values
                       (background + increment). If False, plot in
                       residual/increment space.
    """
    suffix = 'fullfield' if add_background else 'residual'
    iterate_channels_and_plot(
        batch, predictions, instrument_name, plot_scatter_with_diagonal,
        obs_config=obs_config, exp_name=f'{exp_name}_{suffix}',
        obs_stats=obs_stats, version=version, add_background=add_background
    )


def _plot_histogram_channel(
    _batch, _target_node_type, channel_idx,
    valid_ground_truth, valid_prediction, _valid_mask,
    _valid_lat, _valid_lon, instrument_name,
    feature_name='ch0', exp_name='debug_analysis', obs_stats=None,
    version='version_0', **kwargs
):
    """Helper function to plot histogram for a single channel.

    Note: Parameters prefixed with _ are required by the common
    interface but not used in this specific plotting function.
    """
    # Prepare data for plot_hist: [ground_truth, predictions]
    data = np.array([valid_ground_truth, valid_prediction])

    # obs_stats is a dict keyed by feature_name -> [mean, std]
    if isinstance(obs_stats, dict):
        obs_stats = obs_stats.get(feature_name)

    # Use provided obs_stats or calculate from data
    if obs_stats is None:
        combined_mean = np.mean(data)
        combined_std = np.std(data)
        obs_stats = [combined_mean, combined_std]

    # Create histogram plot using feature name
    var_name = f'{instrument_name}_{feature_name}'
    plot_hist(
        var_name,
        data,
        obs_stats,
        exp_name=exp_name,
        version=version,
        **kwargs)
    print(f"Histogram saved for {var_name}")


def plot_histogram_comparison(
    batch, predictions, instrument_name='satellite_atms',
    exp_name='debug_analysis', obs_stats=None, obs_config=None,
    version='version_0', add_background=True, **kwargs
):
    """
    Creates histogram plots comparing predictions vs. ground truth
    for each channel.

    Args:
        batch: HeteroData batch containing ground truth
        predictions: Dictionary of predictions
        instrument_name: Name of the instrument to plot
        exp_name: Experiment name for output directory
        obs_stats: dict mapping feature_name -> [mean, std] (optional)
        obs_config: Observation configuration dictionary (optional)
        version: Version name for output directory (optional)
        add_background: If True (default), plot full analysis values
                       (background + increment). If False, plot in
                       residual/increment space.
    """
    suffix = 'fullfield' if add_background else 'residual'
    iterate_channels_and_plot(
        batch, predictions, instrument_name, _plot_histogram_channel,
        obs_config=obs_config, exp_name=f'{exp_name}_{suffix}',
        obs_stats=obs_stats, version=version, add_background=add_background
    )


def _plot_vertical_profile_channel(
        batch,
        target_node_type,
        channel_idx,
        valid_ground_truth,
        valid_prediction,
        valid_mask,
        _valid_lat,
        _valid_lon,
        instrument_name,
        feature_name='ch0',
        exp_name='debug_analysis',
        obs_stats=None,
        version='version_0',
        obs_config=None,
        **kwargs):
    """Helper function to plot vertical profile for radiosonde data.

    Extracts and prepares data, then calls the plotting function in vis.py.

    Note: Parameters prefixed with _ are required by the common interface
    but not used in this specific plotting function.
    """
    # Only plot vertical profiles for radiosonde
    if 'radiosonde' not in instrument_name.lower():
        return

    # Check if pressure data is available
    node = batch[target_node_type]
    if not hasattr(node, 'x') or node.x is None:
        print(
            f"Warning: No metadata (pressure) found for "
            f"'{target_node_type}'. Skipping vertical profile."
        )
        return

    # Get airPressure from metadata (stored in .x)
    airpressure_idx = 0  # Default to first column
    if obs_config and 'metadata' in obs_config:
        metadata_keys = obs_config.get('metadata', [])
        if 'airPressure' in metadata_keys:
            airpressure_idx = metadata_keys.index('airPressure')

    # Get pressure values for all valid data points
    pressure_all = node.x[:, airpressure_idx].numpy()
    if isinstance(valid_mask, torch.Tensor):
        mask_np = valid_mask.numpy()
    else:
        mask_np = valid_mask
    pressure_valid = pressure_all[mask_np]

    # Check if we have valid pressure data
    if len(pressure_valid) == 0 or np.all(np.isnan(pressure_valid)):
        print(
            f"Warning: No valid pressure data for '{target_node_type}'. "
            f"Skipping vertical profile."
        )
        return

    # Get unique pressure levels and sort them (descending)
    unique_pressures = np.unique(pressure_valid)
    unique_pressures = unique_pressures[~np.isnan(unique_pressures)]
    unique_pressures = np.sort(unique_pressures)[::-1]

    if len(unique_pressures) < 2:
        print(
            f"Warning: Need at least 2 pressure levels for vertical "
            f"profile. Found {len(unique_pressures)}. Skipping."
        )
        return

    # Group data by pressure level and compute statistics
    gt_by_pressure = []
    pred_by_pressure = []
    pressure_levels = []

    for p_level in unique_pressures:
        # Find indices where pressure matches this level
        p_mask = np.abs(pressure_valid - p_level) < 1e-6

        if np.sum(p_mask) > 0:
            gt_values = valid_ground_truth[p_mask]
            pred_values = valid_prediction[p_mask]

            # Compute mean and std for this pressure level
            gt_mean = np.mean(gt_values)
            pred_mean = np.mean(pred_values)
            gt_std = np.std(gt_values)
            pred_std = np.std(pred_values)

            pressure_levels.append(p_level)
            gt_by_pressure.append({'mean': gt_mean, 'std': gt_std})
            pred_by_pressure.append({'mean': pred_mean, 'std': pred_std})

    if len(pressure_levels) == 0:
        print(
            f"Warning: No valid data grouped by pressure. "
            f"Skipping vertical profile."
        )
        return

    # Prepare data arrays
    pressure_levels = np.array(pressure_levels)
    gt_means = np.array([d['mean'] for d in gt_by_pressure])
    gt_stds = np.array([d['std'] for d in gt_by_pressure])
    pred_means = np.array([d['mean'] for d in pred_by_pressure])
    pred_stds = np.array([d['std'] for d in pred_by_pressure])

    # Convert standardized values back to original scale
    # Reverse standardization: original = standardized * std + mean
    # obs_stats is a dict keyed by feature_name -> [mean, std]
    if isinstance(obs_stats, dict):
        obs_stats = obs_stats.get(feature_name)
    if obs_stats is not None:
        var_mean, var_std = obs_stats[0], obs_stats[1]
        gt_means = gt_means * var_std + var_mean
        pred_means = pred_means * var_std + var_mean
        gt_stds = gt_stds * var_std
        pred_stds = pred_stds * var_std

    # Convert pressure from standardized log scale back to linear scale (hPa)
    # Pressure is stored as standardized log(pressure), so:
    # 1. Convert from standardized: original_log = standardized * std + mean
    # 2. Convert from log scale: original = exp(original_log)
    pressure_stats = kwargs.get('pressure_stats', None)
    if pressure_stats is None and 'feature_stats' in kwargs:
        # Try to get pressure stats from feature_stats if available
        feature_stats = kwargs.get('feature_stats')
        if 'radiosonde' in feature_stats:
            pressure_stats = feature_stats['radiosonde'].get('airPressure')

    if pressure_stats is not None:
        # Convert from standardized log scale to log scale
        pressure_mean, pressure_std = pressure_stats[0], pressure_stats[1]
        pressure_levels = pressure_levels * pressure_std + pressure_mean

    # Convert from log scale to linear scale (hPa)
    pressure_levels = np.exp(pressure_levels)

    # Call the plotting function from vis.py
    var_name = f'{instrument_name}_{feature_name}'
    title = f'Vertical Profile: {instrument_name} - {feature_name}'
    plot_vertical_profile(
        var_name, pressure_levels, gt_means, gt_stds, pred_means, pred_stds,
        title=title, exp_name=exp_name, version=version, **kwargs
    )
    print(f"Vertical profile saved for {var_name}")


def _plot_spatial_map_channel(
    _batch, target_node_type, channel_idx,
    valid_ground_truth, valid_prediction, _valid_mask,
    valid_lat, valid_lon, instrument_name,
    feature_name='ch0', exp_name='debug_analysis', obs_stats=None,
    version='version_0', **kwargs
):
    """Helper function to plot spatial map for a single channel.

    Note: Parameters prefixed with _ are required by the common
    interface but not used in this specific plotting function.
    """
    # Check if lat/lon are available
    if valid_lat is None or valid_lon is None:
        print(
            f"Warning: No latitude/longitude found for "
            f"'{target_node_type}'. Skipping spatial map."
        )
        return

    # Prepare data for plot_map: [ground_truth, predictions]
    z = np.array([valid_ground_truth, valid_prediction])

    # obs_stats is a dict keyed by feature_name -> [mean, std]
    if isinstance(obs_stats, dict):
        obs_stats = obs_stats.get(feature_name)

    # Use provided obs_stats or calculate from data
    if obs_stats is None:
        combined_mean = np.mean(z)
        combined_std = np.std(z)
        obs_stats = [combined_mean, combined_std]

    # Create spatial map plot using feature name
    var_name = f'{instrument_name}_{feature_name}'
    title = f'{instrument_name} - {feature_name}'
    plot_map(
        var_name, valid_lon, valid_lat, z, obs_stats, title=title,
        exp_name=exp_name, version=version, **kwargs
    )
    print(f"Spatial map saved for {var_name}")


def plot_spatial_map(
    batch, predictions, instrument_name='satellite_atms',
    exp_name='debug_analysis', obs_stats=None, obs_config=None,
    version='version_0', add_background=True, **kwargs
):
    """
    Creates spatial map plots comparing predictions vs. ground truth
    for each channel. Requires latitude and longitude information in
    the batch.

    Args:
        batch: HeteroData batch containing ground truth and coordinates
        predictions: Dictionary of predictions
        instrument_name: Name of the instrument to plot
        exp_name: Experiment name for output directory
        obs_stats: dict mapping feature_name -> [mean, std] (optional)
        obs_config: Observation configuration dictionary (optional)
        version: Version name for output directory (optional)
        add_background: If True (default), plot full analysis values
                       (background + increment). If False, plot in
                       residual/increment space.
    """
    suffix = 'fullfield' if add_background else 'residual'
    iterate_channels_and_plot(
        batch, predictions, instrument_name, _plot_spatial_map_channel,
        obs_config=obs_config, exp_name=f'{exp_name}_{suffix}',
        obs_stats=obs_stats, version=version, add_background=add_background
    )


def plot_vertical_profile_sonde(
    batch, predictions, instrument_name='radiosonde',
    exp_name='debug_analysis', obs_stats=None, obs_config=None,
    version='version_0', add_background=True, **kwargs
):
    """
    Creates vertical profile plots for radiosonde data.
    Plots pressure (y-axis, inverted) vs variable value (x-axis) for
    each channel.

    Args:
        batch: HeteroData batch containing ground truth
        predictions: Dictionary of predictions
        instrument_name: Name of the instrument to plot
                        (should be radiosonde)
        exp_name: Experiment name for output directory
        obs_stats: dict mapping feature_name -> [mean, std] (optional)
        obs_config: Observation configuration dictionary (optional)
        version: Version name for output directory (optional)
        add_background: If True (default), plot full analysis values
                       (background + increment). If False, plot in
                       residual/increment space.
    """
    if 'radiosonde' not in instrument_name.lower():
        print(
            f"Warning: plot_vertical_profile is designed for "
            f"radiosonde data. Skipping for {instrument_name}."
        )
        return

    # Pass feature_stats through kwargs
    plot_kwargs = kwargs.copy()
    if 'feature_stats' in kwargs:
        plot_kwargs['feature_stats'] = kwargs['feature_stats']

    suffix = 'fullfield' if add_background else 'residual'
    iterate_channels_and_plot(
        batch, predictions, instrument_name, _plot_vertical_profile_channel,
        obs_config=obs_config, exp_name=f'{exp_name}_{suffix}',
        obs_stats=obs_stats, version=version, add_background=add_background,
        **plot_kwargs
    )


def get_latest_debug_file(debug_dir, rank=0, pattern_prefix='debug_data'):
    """
    Find the most recent debug file in the directory based on epoch and step numbers.

    Args:
        debug_dir: Directory containing debug files
        rank: Rank number to search for (default: 0)
        pattern_prefix: Prefix of debug files (default: 'debug_data')

    Returns:
        str: Filename of the most recent debug file, or None if not found

    Example:
        Files: debug_data_epoch_39_step_480_rank0.pt, debug_data_epoch_38_step_460_rank0.pt
        Returns: debug_data_epoch_39_step_480_rank0.pt
    """
    import glob
    import re

    if not os.path.exists(debug_dir):
        print(f"Error: Directory {debug_dir} does not exist")
        return None

    # Pattern: debug_data_epoch_{epoch}_step_{step}_rank{rank}.pt
    pattern = os.path.join(
        debug_dir, f'{pattern_prefix}_epoch_*_step_*_rank{rank}.pt'
    )
    files = glob.glob(pattern)

    if not files:
        print(f"No debug files found matching pattern: {pattern}")
        return None

    # Extract epoch and step numbers from filenames
    file_info = []
    for filepath in files:
        filename = os.path.basename(filepath)
        # Match pattern: debug_data_epoch_{epoch}_step_{step}_rank{rank}
        match = re.search(
            r'epoch_(\d+)_step_(\d+)_rank(\d+)', filename
        )
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))
            file_info.append((epoch, step, filename))

    if not file_info:
        print("No valid debug files found with epoch/step information")
        return None

    # Sort by epoch (descending), then by step (descending)
    file_info.sort(key=lambda x: (x[0], x[1]), reverse=True)

    latest_file = file_info[0][2]
    latest_epoch = file_info[0][0]
    latest_step = file_info[0][1]

    print(f"Found {len(file_info)} debug file(s)")
    print(
        f"Latest file: {latest_file} (epoch={latest_epoch}, step={latest_step})")

    return latest_file


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze debug outputs from training')
    parser.add_argument('--exp_name', type=str, default='test_multi_ddp',
                        help='Experiment name (subdirectory in debug_outputs)')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank number to analyze (default: 0)')
    parser.add_argument('--instrument', type=str, default='satellite_atms',
                        help='Instrument to analyze (default: satellite_atms)')
    parser.add_argument(
        '--debug_base_dir', type=str,
        default=(
            '/scratch3/NCEPDEV/da/Xin.C.Jin/git/ocelot3/debug_outputs'
        ),
        help='Base directory for debug outputs'
    )
    parser.add_argument('--analysis_name', type=str, default='debug_analysis',
                        help='Name for output analysis directory')
    parser.add_argument(
        '--version',
        type=str,
        default='version_0',
        help='Version name for output directory (e.g., v1, version_0)')

    args = parser.parse_args()

    # --- Set paths based on arguments ---
    debug_base_dir = args.debug_base_dir
    exp_name = args.exp_name
    rank = args.rank

    debug_dir = os.path.join(debug_base_dir, exp_name, args.version)

    print(f"Analyzing experiment: {exp_name}")
    print(f"Debug directory: {debug_dir}")
    print(f"Rank: {rank}")
    print()

    # Automatically find the latest debug file
    debug_filename = get_latest_debug_file(debug_dir, rank=rank)

    if debug_filename is None:
        print("No debug file found. Exiting.")
        exit(1)

    debug_file_path = os.path.join(debug_dir, debug_filename)

    batch_data, predictions_data = load_debug_data(debug_file_path)
    # Load the configuration
    observation_config, feature_stats, fill_values, _instrument_weights, _increment_stats = load_observation_config(
        exp_type="regional_da", config_name="urma")

    if batch_data and predictions_data:
        # --- Set analysis parameters ---
        instrument_to_plot = args.instrument
        analysis_name = f"{args.analysis_name}/{exp_name}"

        # Split instrument name to get obs_type and instrument_name
        # e.g., 'satellite_atms' -> obs_type='satellite',
        # instrument_name='atms'
        parts = instrument_to_plot.split('_', 1)
        print(parts)
        if len(parts) == 2:
            obs_type, instrument_name = parts
        else:
            print(
                f"Error: Could not parse instrument '{instrument_to_plot}'. "
                f"Expected format: 'obs_type_instrument_name'"
            )
            exit(1)

        # Get the configuration and stats for this instrument
        try:
            obs_config = observation_config[obs_type][instrument_name]
        except KeyError as e:
            traceback.print_exc()
            print(
                f"Error: Could not find configuration for {instrument_to_plot}: {e}"
            )
            print(
                f"Available obs types: {list(observation_config.keys())}"
            )
            if obs_type in observation_config:
                print(
                    f"Available instruments for {obs_type}: "
                    f"{list(observation_config[obs_type].keys())}"
                )
            # If we can't find the config, exit with an error
            exit(1)

        # Try to get stats for the first feature; fall back gracefully if
        # missing
        obs_stats = None
        try:
            first_feature = obs_config['features'][0]
            # feature_stats is usually keyed by instrument_name; for some cases
            # (e.g. radiosonde) it may be keyed by obs_type instead.
            stats_dict = feature_stats.get(
                instrument_name) or feature_stats.get(obs_type, {})
            print(stats_dict)
            obs_stats = stats_dict
            # if first_feature in stats_dict:
            #     obs_stats = stats_dict[first_feature]
            print(f"\nInstrument: {instrument_to_plot}")
            print(f"  Obs type: {obs_type}")
            print(f"  Instrument name: {instrument_name}")
            print(f"  First feature: {first_feature}")
            if obs_stats is not None:
                print(f"  Stats (mean, std): {obs_stats}")
            else:
                print("  Stats: not found, will compute from data in plotting.")
        except Exception as e:
            traceback.print_exc()
            print(
                f"Warning: could not determine obs_stats for {instrument_to_plot}: {e}")
            obs_stats = None

        # Create individual plots (comment out any you don't need).
        # Each is generated twice: once with the background (ges) added
        # back for full-analysis-field values, and once in the raw
        # residual/increment space the model was trained on.
        for add_background in (True, False):
            space_label = 'full-field' if add_background else 'residual'

            print(f"\n=== Creating Scatter Plots ({space_label}) ===")
            plot_predictions_vs_ground_truth(
                batch_data, predictions_data, instrument_to_plot,
                analysis_name, obs_stats, obs_config, args.version,
                add_background=add_background
            )

            print(f"\n=== Creating Histogram Comparisons ({space_label}) ===")
            plot_histogram_comparison(
                batch_data, predictions_data, instrument_to_plot,
                analysis_name, obs_stats, obs_config, args.version,
                add_background=add_background
            )

            print(f"\n=== Creating Spatial Maps ({space_label}) ===")
            plot_spatial_map(
                batch_data, predictions_data, instrument_to_plot,
                analysis_name, obs_stats, obs_config, args.version,
                add_background=add_background
            )

            print(f"\n=== Creating Vertical Profiles ({space_label}) ===")
            plot_vertical_profile_sonde(
                batch_data, predictions_data, instrument_to_plot,
                analysis_name, obs_stats, obs_config, args.version,
                feature_stats=feature_stats, add_background=add_background
            )

        print("\n=== All plots completed! ===")
