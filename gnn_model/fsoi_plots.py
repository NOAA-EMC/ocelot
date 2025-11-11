"""
Visualization tools for FSOI (Forecast Sensitivity to Observation Impact) analysis.

This module provides plotting functions to analyze and visualize FSOI results:
- Spatial maps showing observation impact distribution
- Channel-wise FSOI comparisons
- Time series of FSOI values
- Instrument comparison plots

Author: Azadeh Gholoubi
Date: November 2025
"""

import os
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def plot_fsoi_spatial_map(
    stats_df: pd.DataFrame,
    instrument: Optional[str] = None,
    channel: Optional[str] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    point_size: int = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    projection=ccrs.PlateCarree(),
):
    """
    Create a spatial map of FSOI values.

    Args:
        stats_df: DataFrame with FSOI statistics (from aggregate_fsoi_statistics)
        instrument: Filter by specific instrument (optional)
        channel: Filter by specific channel (optional)
        save_path: Path to save figure
        title: Plot title
        cmap: Colormap for FSOI values
        point_size: Size of scatter points
        vmin, vmax: Color scale limits (auto if None)
        projection: Cartopy projection
    """
    # Filter data
    df = stats_df.copy()
    if instrument is not None:
        df = df[df["instrument"] == instrument]
    if channel is not None:
        df = df[df["channel"] == channel]

    if len(df) == 0:
        print(f"[FSOI Plot] No data for instrument={instrument}, channel={channel}")
        return

    # Get coordinates and FSOI values
    lats = df["lat"].values
    lons = df["lon"].values
    fsoi = df["fsoi_value"].values

    # Remove invalid values
    valid = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(fsoi)
    lats, lons, fsoi = lats[valid], lons[valid], fsoi[valid]

    if len(fsoi) == 0:
        print(f"[FSOI Plot] No valid FSOI values to plot")
        return

    # Determine color scale (symmetric around zero)
    if vmin is None or vmax is None:
        abs_max = np.nanpercentile(np.abs(fsoi), 95)  # Use 95th percentile for robustness
        if abs_max == 0:
            abs_max = 1.0
        vmin = -abs_max
        vmax = abs_max

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": projection})

    # Plot FSOI values
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    scatter = ax.scatter(
        lons, lats, c=fsoi, s=point_size, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), alpha=0.7, edgecolors="none"
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label("FSOI Value (Positive = Beneficial)", fontsize=12)

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
    ax.set_global()

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Title
    if title is None:
        title_parts = ["FSOI Spatial Distribution"]
        if instrument:
            title_parts.append(f"- {instrument}")
        if channel:
            title_parts.append(f"- {channel}")
        title = " ".join(title_parts)

    ax.set_title(title, fontsize=14, pad=20)

    # Statistics annotation
    mean_fsoi = np.mean(fsoi)
    std_fsoi = np.std(fsoi)
    beneficial = np.sum(fsoi > 0)
    detrimental = np.sum(fsoi < 0)

    stats_text = f"Mean: {mean_fsoi:.2e}\n"
    stats_text += f"Std: {std_fsoi:.2e}\n"
    stats_text += f"Beneficial: {beneficial} ({100*beneficial/len(fsoi):.1f}%)\n"
    stats_text += f"Detrimental: {detrimental} ({100*detrimental/len(fsoi):.1f}%)"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[FSOI Plot] Saved to {save_path}")

    plt.close()


def plot_fsoi_by_instrument(
    stats_df: pd.DataFrame, save_path: Optional[str] = None, title: str = "FSOI by Instrument"
):
    """
    Create bar plot comparing mean FSOI across instruments.

    Args:
        stats_df: DataFrame with FSOI statistics
        save_path: Path to save figure
        title: Plot title
    """
    from fsoi import summarize_fsoi_by_instrument

    summary = summarize_fsoi_by_instrument(stats_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean FSOI
    instruments = summary["instrument"].values
    mean_fsoi = summary["fsoi_value_mean"].values
    std_fsoi = summary["fsoi_value_std"].values

    colors = ["green" if x > 0 else "red" for x in mean_fsoi]

    axes[0].barh(instruments, mean_fsoi, xerr=std_fsoi, color=colors, alpha=0.7)
    axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Mean FSOI Value", fontsize=12)
    axes[0].set_title("Mean FSOI by Instrument", fontsize=13)
    axes[0].grid(axis="x", alpha=0.3)

    # Plot 2: Total FSOI (cumulative impact)
    total_fsoi = summary["fsoi_value_sum"].values
    colors = ["green" if x > 0 else "red" for x in total_fsoi]

    axes[1].barh(instruments, total_fsoi, color=colors, alpha=0.7)
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Total FSOI Value", fontsize=12)
    axes[1].set_title("Total FSOI by Instrument", fontsize=13)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[FSOI Plot] Saved to {save_path}")

    plt.close()


def plot_fsoi_by_channel(
    stats_df: pd.DataFrame, instrument: str, save_path: Optional[str] = None, title: Optional[str] = None
):
    """
    Create bar plot showing FSOI for each channel of an instrument.

    Args:
        stats_df: DataFrame with FSOI statistics
        instrument: Instrument name
        save_path: Path to save figure
        title: Plot title
    """
    from fsoi import summarize_fsoi_by_channel

    summary = summarize_fsoi_by_channel(stats_df, instrument)

    if len(summary) == 0:
        print(f"[FSOI Plot] No data for instrument {instrument}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    channels = summary["channel"].values
    mean_fsoi = summary["fsoi_value_mean"].values
    std_fsoi = summary["fsoi_value_std"].values

    # Plot 1: Mean FSOI per channel
    colors = ["green" if x > 0 else "red" for x in mean_fsoi]
    x_pos = np.arange(len(channels))

    axes[0].bar(x_pos, mean_fsoi, yerr=std_fsoi, color=colors, alpha=0.7)
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(channels, rotation=45, ha="right")
    axes[0].set_ylabel("Mean FSOI Value", fontsize=12)
    axes[0].set_title(f"Mean FSOI by Channel - {instrument}", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)

    # Plot 2: Total FSOI per channel
    total_fsoi = summary["fsoi_value_sum"].values
    colors = ["green" if x > 0 else "red" for x in total_fsoi]

    axes[1].bar(x_pos, total_fsoi, color=colors, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(channels, rotation=45, ha="right")
    axes[1].set_ylabel("Total FSOI Value", fontsize=12)
    axes[1].set_title(f"Total FSOI by Channel - {instrument}", fontsize=13)
    axes[1].grid(axis="y", alpha=0.3)

    if title is None:
        title = f"FSOI Analysis - {instrument}"
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[FSOI Plot] Saved to {save_path}")

    plt.close()


def plot_fsoi_distribution(
    stats_df: pd.DataFrame,
    instrument: Optional[str] = None,
    save_path: Optional[str] = None,
    title: str = "FSOI Value Distribution",
):
    """
    Create histogram of FSOI value distribution.

    Args:
        stats_df: DataFrame with FSOI statistics
        instrument: Filter by specific instrument (optional)
        save_path: Path to save figure
        title: Plot title
    """
    df = stats_df.copy()
    if instrument is not None:
        df = df[df["instrument"] == instrument]

    fsoi = df["fsoi_value"].values
    fsoi = fsoi[np.isfinite(fsoi)]

    if len(fsoi) == 0:
        print(f"[FSOI Plot] No valid FSOI values for distribution plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(fsoi, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0].axvline(0, color="red", linewidth=2, linestyle="--", label="Zero Impact")
    axes[0].axvline(np.mean(fsoi), color="orange", linewidth=2, label=f"Mean = {np.mean(fsoi):.2e}")
    axes[0].set_xlabel("FSOI Value", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("FSOI Distribution", fontsize=13)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Cumulative distribution
    sorted_fsoi = np.sort(fsoi)
    cumulative = np.arange(1, len(sorted_fsoi) + 1) / len(sorted_fsoi)

    axes[1].plot(sorted_fsoi, cumulative, linewidth=2, color="steelblue")
    axes[1].axvline(0, color="red", linewidth=2, linestyle="--", label="Zero Impact")
    axes[1].axhline(0.5, color="orange", linewidth=1, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("FSOI Value", fontsize=12)
    axes[1].set_ylabel("Cumulative Probability", fontsize=12)
    axes[1].set_title("Cumulative Distribution", fontsize=13)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[FSOI Plot] Saved to {save_path}")

    plt.close()


def plot_high_impact_observations(
    stats_df: pd.DataFrame,
    top_n: int = 50,
    impact_type: str = "beneficial",
    save_path: Optional[str] = None,
    projection=ccrs.PlateCarree(),
):
    """
    Plot locations of highest impact observations.

    Args:
        stats_df: DataFrame with FSOI statistics
        top_n: Number of top observations to show
        impact_type: "beneficial" or "detrimental"
        save_path: Path to save figure
        projection: Cartopy projection
    """
    from fsoi import identify_high_impact_observations

    high_impact = identify_high_impact_observations(stats_df, top_n=top_n, impact_type=impact_type)

    if len(high_impact) == 0:
        print(f"[FSOI Plot] No high impact observations found")
        return

    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": projection})

    # Plot all observations in gray
    ax.scatter(
        stats_df["lon"],
        stats_df["lat"],
        c="lightgray",
        s=5,
        alpha=0.3,
        transform=ccrs.PlateCarree(),
        label="All observations",
    )

    # Highlight high-impact observations
    color = "green" if impact_type == "beneficial" else "red"
    scatter = ax.scatter(
        high_impact["lon"],
        high_impact["lat"],
        c=high_impact["fsoi_value"],
        s=50,
        cmap="Greens" if impact_type == "beneficial" else "Reds",
        transform=ccrs.PlateCarree(),
        edgecolors="black",
        linewidths=0.5,
        label=f"Top {top_n} {impact_type}",
        zorder=10,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label(f"FSOI Value ({impact_type})", fontsize=12)

    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_global()

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"Top {top_n} {impact_type.capitalize()} Observations", fontsize=14, pad=20)
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[FSOI Plot] Saved to {save_path}")

    plt.close()


def create_fsoi_summary_report(
    stats_df: pd.DataFrame, output_dir: str, epoch: int = 0, batch_idx: int = 0, instruments: Optional[list] = None
):
    """
    Create a comprehensive FSOI analysis report with multiple plots.

    Args:
        stats_df: DataFrame with FSOI statistics
        output_dir: Directory to save plots
        epoch: Epoch number for naming
        batch_idx: Batch index for naming
        instruments: List of instruments to analyze (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[FSOI Report] Generating comprehensive FSOI analysis for epoch {epoch}, batch {batch_idx}")

    # Overall statistics
    print(f"[FSOI Report] Total observations: {len(stats_df)}")
    print(f"[FSOI Report] Mean FSOI: {stats_df['fsoi_value'].mean():.4e}")
    print(f"[FSOI Report] Std FSOI: {stats_df['fsoi_value'].std():.4e}")

    # 1. FSOI by instrument comparison
    plot_fsoi_by_instrument(
        stats_df, save_path=os.path.join(output_dir, f"fsoi_by_instrument_epoch{epoch}_batch{batch_idx}.png")
    )

    # 2. Overall FSOI distribution
    plot_fsoi_distribution(
        stats_df, save_path=os.path.join(output_dir, f"fsoi_distribution_epoch{epoch}_batch{batch_idx}.png")
    )

    # 3. Spatial map of all observations
    plot_fsoi_spatial_map(
        stats_df,
        save_path=os.path.join(output_dir, f"fsoi_spatial_all_epoch{epoch}_batch{batch_idx}.png"),
        title=f"FSOI Spatial Distribution - Epoch {epoch}",
    )

    # 4. High impact observations
    plot_high_impact_observations(
        stats_df,
        top_n=100,
        impact_type="beneficial",
        save_path=os.path.join(output_dir, f"fsoi_high_beneficial_epoch{epoch}_batch{batch_idx}.png"),
    )

    plot_high_impact_observations(
        stats_df,
        top_n=100,
        impact_type="detrimental",
        save_path=os.path.join(output_dir, f"fsoi_high_detrimental_epoch{epoch}_batch{batch_idx}.png"),
    )

    # 5. Per-instrument analysis
    if instruments is None:
        instruments = stats_df["instrument"].unique()

    for inst in instruments:
        inst_df = stats_df[stats_df["instrument"] == inst]
        if len(inst_df) == 0:
            continue

        print(f"[FSOI Report] Analyzing {inst} ({len(inst_df)} observations)")

        # Channel-wise analysis
        plot_fsoi_by_channel(
            stats_df, instrument=inst, save_path=os.path.join(output_dir, f"fsoi_channels_{inst}_epoch{epoch}_batch{batch_idx}.png")
        )

        # Spatial map for this instrument
        plot_fsoi_spatial_map(
            stats_df,
            instrument=inst,
            save_path=os.path.join(output_dir, f"fsoi_spatial_{inst}_epoch{epoch}_batch{batch_idx}.png"),
            title=f"FSOI Spatial Distribution - {inst} - Epoch {epoch}",
        )

    print(f"[FSOI Report] Report generation complete. Files saved to {output_dir}")
