# Third-party
import matplotlib.pyplot as plt
import numpy as np
import functools
import os
import traceback

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def catch_plotting_errors(func):
    """A decorator to catch and print exceptions from plotting functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(
                f"--- Error in plotting function '{func.__name__}': {e} ---"
            )
            traceback.print_exc()
            return None
    return wrapper


def standard_scale(data, stats):
    data[0] = stats[1] * data[0] + stats[0]
    data[1] = stats[1] * data[1] + stats[0]
    return data

    
@catch_plotting_errors
def plot_hist(
    var_name, data, obs_stats, exp_name="default_experiment",
    version="version_0", **kwargs
):
    data = standard_scale(data, obs_stats)
    n = len(data[0])
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    mx = np.max(data, axis=1)
    mn = np.min(data, axis=1)

    # Calculate difference statistics
    data_diff = data[1] - data[0]
    mean_diff = np.mean(data_diff)
    std_diff = np.std(data_diff)
    mx_diff = np.max(data_diff)
    mn_diff = np.min(data_diff)

    # Make proper bin sizes using the equation max-min/sqrt(n). Then
    # extend the bin range to 4x the standard deviation
    binsize = 0.25
    print('binsize: ', binsize)
    # bins = np.arange(mean-(4*std),mean+(4*std),binsize)

    # Now plot figure
    fig = plt.figure(figsize=(18,8))

    labels = ['Ground Truth', 'Prediction', 'Difference']

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        delta = obs_stats[1] * 3

        if i < 2:
            # Plot Ground Truth or Prediction
            d = data[i]
            lo, hi = np.nanmin(d), np.nanmax(d)
            if lo == hi:  # constant predictions → expand range so bins have non-zero width
                lo, hi = lo - abs(lo) * 0.01 - 1e-6, hi + abs(hi) * 0.01 + 1e-6
            ax.hist(d, bins=20, range=(lo, hi))
            ax.set_xlim(obs_stats[0] - delta, obs_stats[0] + delta)
            text = (
                f' total: {n}\n mean: {mean[i]:.4f}\n std: {std[i]:.4f}\n '
                f'max: {mx[i]:.4f}\n min: {mn[i]:.4f}'
            )
        else:
            # Plot Difference
            lo, hi = np.nanmin(data_diff), np.nanmax(data_diff)
            if lo == hi:
                lo, hi = lo - abs(lo) * 0.01 - 1e-6, hi + abs(hi) * 0.01 + 1e-6
            ax.hist(data_diff, bins=20, range=(lo, hi))
            ax.set_xlim(-delta, delta)
            text = (
                f' total: {n}\n mean: {mean_diff:.4f}\n '
                f'std: {std_diff:.4f}\n max: {mx_diff:.4f}\n '
                f'min: {mn_diff:.4f}'
            )

        # Add labels
        plt.xlabel(var_name)
        plt.ylabel('Count')
        plt.title(f'{var_name} - {labels[i]}', fontsize=14)
        ax.text(0.2, 0.7, text, transform=ax.transAxes, fontsize=12)
        # data = df[var_name]
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.hist(data)
        # plt.title(var_name)
    dpi=150
    plt.tight_layout()
    output_dir = f'figures/{exp_name}/{version}'
    os.makedirs(output_dir, exist_ok=True)
    # Add pressure level to filename if present
    pressure_suffix = ''
    if 'pressure_level' in kwargs:
        pressure_val = kwargs['pressure_level']
        pressure_suffix = f'_pressure_{pressure_val:.2f}'
    pngfile = f'{output_dir}/hist_{var_name}{pressure_suffix}.png'
    fig.savefig(pngfile)
    plt.close()


@catch_plotting_errors
def plot_map(
    var_name, x, y, z, obs_stats, title='title',
    exp_name="default_experiment", version="version_0", **kwargs
):
    z = standard_scale(z, obs_stats)
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    zmin = np.min(z, axis=1)
    zmax = np.max(z, axis=1)
    zstd = np.std(z, axis=1)
    zavg = np.mean(z, axis=1)
    # Compute difference
    z_diff = z[1] - z[0]
    zmin_diff = np.min(z_diff)
    zmax_diff = np.max(z_diff)
    zstd_diff = np.std(z_diff)
    zavg_diff = np.mean(z_diff)
    # zcnt = z.size()
    # Set colorbar
    delta = obs_stats[1] * 3
    cmax =  obs_stats[0]  + delta
    cmin =  obs_stats[0] - delta
    
    cmap=plt.get_cmap('jet')
    
    # Set plot variable unit
    units = 'W m-2 sr-1 m'
    units = 'K'

    fig = plt.figure(figsize=(18,8))
    
    labels = ['Ground Truth', 'Prediction', 'Difference']
    
    for i in range(3):
        # Initialize the plot pointing to the project
        ax = fig.add_subplot(
            1, 3, i+1, projection=ccrs.PlateCarree()
        )
        
        # Get data to plot
        if i < 2:
            z_plot = z[i]
            vmin, vmax = cmin, cmax
            zmax_val = zmax[i]
            zmin_val = zmin[i]
            zavg_val = zavg[i]
            zstd_val = zstd[i]
        else:
            z_plot = z_diff
            zmax_val = zmax_diff
            zmin_val = zmin_diff
            zavg_val = zavg_diff
            zstd_val = zstd_diff
            vmin, vmax = -3 * zstd_diff, 3 * zstd_diff              
            cmap=plt.get_cmap('RdBu_r')

            
        
        # Get scatter data
        sc = ax.scatter(
            x, y, c=z_plot, s=5, marker="o", linewidth=3, alpha=1.0,
            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap,
            norm=None, edgecolor='none', antialiased=True
        )
        
        # Plot colorbar
        cbar = fig.colorbar(
            sc, ax=ax, orientation="horizontal", pad=0.1, fraction=0.15,
            aspect=40, extend='both'
        )
        cbar.ax.set_xlabel(units, fontsize=10, loc='right')
        
        # Plot globally
        # ax.set_global()

        # Add land and ocean
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        
        # Add gridlines
        gline = ax.gridlines(
            draw_labels=True, dms=True, x_inline=False, y_inline=False,
            color='lightgray', alpha=0.5, linewidth=1.0, linestyle='--'
        )
        gline.top_labels = False
        gline.right_labels = False
        
        # Get title and png file from the input filename
        
        subplot_title = f'{title} - {labels[i]}'
        # Add figure labels
        ax.set_title(subplot_title, pad=15, fontsize=18)
        #   text = f"Total Count: {zcnt:0.0f}     Max: {zmax:0.3f}     "
        #   f"Min: {zmin:0.3f}     Mean: {zavg:0.3f}     Std: {zstd:0.3f} "
        #   f"{units}"
        text = (
            f"     Max: {zmax_val:0.3f}     Min: {zmin_val:0.3f}     "
            f"Mean: {zavg_val:0.3f}     Std: {zstd_val:0.3f}"
        )
        ax.text(
            0.05, -0.15, text, transform=ax.transAxes, va='bottom',
            fontsize=10
        )

    dpi = 150
    plt.tight_layout()
    output_dir = f'figures/{exp_name}/{version}'
    os.makedirs(output_dir, exist_ok=True)
    # Add pressure level to filename if present
    pressure_suffix = ''
    if 'pressure_level' in kwargs:
        pressure_val = kwargs['pressure_level']
        pressure_suffix = f'_pressure_{pressure_val:.2f}'
    pngfile = f'{output_dir}/map_{var_name}{pressure_suffix}.png'
    fig.savefig(pngfile)
    plt.close()


@catch_plotting_errors
def plot_scatter_with_diagonal(
    _batch, _target_node_type, channel_idx,
    valid_ground_truth, valid_prediction, _valid_mask,
    _valid_lat, _valid_lon, instrument_name,
    exp_name='debug_analysis', **kwargs
):
    """
    Creates a scatter plot with a 1:1 diagonal line.

    Args:
        channel_idx: channel index
        valid_ground_truth: numpy array of ground truth values
        valid_prediction: numpy array of predicted values
        instrument_name: name of the instrument
        exp_name: experiment name for output directory

    Note: Parameters prefixed with _ are required by the common interface
    but not used in this specific plotting function.

    Returns:
        plot_filename: path to the saved plot
    """
    fig = plt.figure(figsize=(8, 6))

    # Scatter plot of predictions vs. ground truth
    plt.scatter(
        valid_ground_truth, valid_prediction, alpha=0.5,
        label='Predictions'
    )

    # Add a 1:1 line for reference
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(
        lims, lims, 'r--', alpha=0.75, zorder=0,
        label='Ideal 1:1 Line'
    )

    feature_name = kwargs.get('feature_name', f"ch{channel_idx}")
    plt.title(
        f'Prediction vs. Ground Truth for {instrument_name} '
        f'{feature_name}'
    )
    plt.xlabel('Ground Truth (Normalized)')
    plt.ylabel('Predictions (Normalized)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()

    # Save the plot to a file
    version = kwargs.get('version', 'version_0')
    output_dir = f'figures/{exp_name}/{version}'
    os.makedirs(output_dir, exist_ok=True)
    safe_feature_name = (
        str(feature_name).replace('/', '_').replace(' ', '_')
    )
    # Add pressure level to filename if present
    pressure_suffix = ''
    if 'pressure_level' in kwargs:
        pressure_val = kwargs['pressure_level']
        pressure_suffix = f'_pressure_{pressure_val:.2f}'
    plot_filename = (
        f'{output_dir}/{instrument_name}_prediction_vs_truth_'
        f'{safe_feature_name}{pressure_suffix}.png'
    )
    fig.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()
    plt.close()

    return plot_filename


@catch_plotting_errors
def plot_vertical_profile(
    var_name, pressure_levels, gt_means, gt_stds, pred_means, pred_stds,
    title='Vertical Profile', exp_name="default_experiment",
    version="version_0", **kwargs
):
    """
    Creates a vertical profile plot showing pressure vs variable value.
    
    Args:
        var_name: Variable name for the plot
        pressure_levels: Array of pressure levels (will be on y-axis,
                        inverted)
        gt_means: Ground truth mean values at each pressure level
        gt_stds: Ground truth standard deviations at each pressure level
        pred_means: Prediction mean values at each pressure level
        pred_stds: Prediction standard deviations at each pressure level
        title: Plot title
        exp_name: Experiment name for output directory
        version: Version name for output directory
        **kwargs: Additional arguments (e.g., pressure_level for filename
                 suffix)
    """
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Plot ground truth with error bars
    ax.plot(
        gt_means, pressure_levels, 'b-o', label='Ground Truth',
        linewidth=2, markersize=6
    )
    ax.fill_betweenx(
        pressure_levels, gt_means - gt_stds, gt_means + gt_stds,
        alpha=0.2, color='blue', label='GT ±1σ'
    )
    
    # Plot predictions with error bars
    ax.plot(
        pred_means, pressure_levels, 'r-s', label='Prediction',
        linewidth=2, markersize=6
    )
    ax.fill_betweenx(
        pressure_levels, pred_means - pred_stds, pred_means + pred_stds,
        alpha=0.2, color='red', label='Pred ±1σ'
    )
    
    # Invert y-axis so higher pressure is at bottom
    # (typical for vertical profiles)
    ax.invert_yaxis()
    
    # Labels and title
    xlabel = (
        var_name.split('_')[-1] if '_' in var_name else var_name
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Pressure (hPa)', fontsize=12)
    ax.set_title(f'{title}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = f'figures/{exp_name}/{version}'
    os.makedirs(output_dir, exist_ok=True)
    # Add pressure level to filename if present
    pressure_suffix = ''
    if 'pressure_level' in kwargs:
        pressure_val = kwargs['pressure_level']
        pressure_suffix = f'_pressure_{pressure_val:.2f}'
    pngfile = (
        f'{output_dir}/vertical_profile_{var_name}{pressure_suffix}.png'
    )
    fig.savefig(pngfile, dpi=150)
    plt.close()
    
    return pngfile
