#!/usr/bin/env python3
"""
Plot relative contribution of ALL observation types to radiosonde temperature
forecast error at different pressure levels.

Shows:
- All satellite channels as separate columns
- Conventional obs types (radiosonde, aircraft, surface)
- Pressure levels as rows
- Heatmap showing relative contribution at each level

Author: Azadeh Gholoubi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
FSOI_CSV = 'FSOI/fsoi_outputs/radiosonde_temp_impact/csv/fsoi_by_channel.csv'
OUTPUT_DIR = Path('FSOI/fsoi_outputs/radiosonde_temp_impact/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard pressure levels
STANDARD_PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

# Instrument display names
INSTRUMENT_NAMES = {
    'radiosonde': 'Radiosonde',
    'aircraft': 'Aircraft',
    'surface_obs': 'Surface',
    'amsua': 'AMSU-A',
    'atms': 'ATMS',
    'ssmis': 'SSMIS',
    'seviri_asr': 'SEVIRI ASR',
    'seviri_csr': 'SEVIRI CSR',
    'avhrr': 'AVHRR',
    'ascat': 'ASCAT'
}

# Channel names for satellites
CHANNEL_NAMES = {
    'amsua': {i: f'AMSU-A ch{i+1}' for i in range(15)},
    'atms': {i: f'ATMS ch{i+1}' for i in range(22)},
    'ssmis': {i: f'SSMIS ch{i+1}' for i in range(24)},
    'seviri_asr': {i: f'SEVIRI ASR ch{4+i}' for i in range(8)},
    'seviri_csr': {i: f'SEVIRI CSR ch{4+i}' for i in range(8)},
    'avhrr': {i: f'AVHRR ch{i+1}' for i in range(5)},
    'ascat': {0: 'ASCAT u-wind', 1: 'ASCAT v-wind'},
    'radiosonde': {0: 'Radiosonde Temp', 1: 'Radiosonde Dewpt', 2: 'Radiosonde U', 3: 'Radiosonde V'},
    'aircraft': {0: 'Aircraft Temp', 1: 'Aircraft Humid', 2: 'Aircraft U', 3: 'Aircraft V'},
    'surface_obs': {0: 'Surface Temp', 1: 'Surface Dewpt', 2: 'Surface U', 3: 'Surface V', 4: 'Surface Pres'}
}


def _ensure_pressure_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a pressure_hpa column using pressure_level_idx if present."""
    df = df.copy()
    if 'pressure_hpa' in df.columns and df['pressure_hpa'].notna().any():
        return df

    if 'pressure_level_idx' in df.columns:
        # Map index to standard levels; unknown -> NaN
        level_map = {i: p for i, p in enumerate(STANDARD_PRESSURE_LEVELS)}
        df['pressure_hpa'] = df['pressure_level_idx'].map(level_map)
    else:
        df['pressure_hpa'] = np.nan
    return df


def load_and_prepare_data():
    """Load FSOI data and prepare for analysis."""
    print(f"Loading data from {FSOI_CSV}...")
    df = pd.read_csv(FSOI_CSV)
    print(f"Loaded {len(df)} records")

    # Ensure pressure_hpa exists (can be synthesized from pressure_level_idx)
    df = _ensure_pressure_column(df)

    # Check if pressure stratification exists
    has_pressure = 'pressure_hpa' in df.columns and df['pressure_hpa'].notna().any()

    if has_pressure:
        print("✓ Pressure level stratification FOUND")
        unique_pressures = sorted([p for p in df['pressure_hpa'].unique() if pd.notna(p)], reverse=True)
        print(f"  Pressure levels present: {unique_pressures}")
    else:
        print("✗ No pressure stratification in data")
        return None, False

    return df, has_pressure


def create_instrument_channel_label(row):
    """Create a label for instrument-channel combination."""
    inst = row['instrument']
    ch = int(row['channel'])

    if inst in CHANNEL_NAMES and ch in CHANNEL_NAMES[inst]:
        return CHANNEL_NAMES[inst][ch]
    else:
        return f"{INSTRUMENT_NAMES.get(inst, inst)} ch{ch}"


def plot_relative_contribution_by_pressure(df):
    """Create heatmap showing relative contribution of each observation type at different pressure levels."""
    # Filter to only records with pressure info
    df_pressure = df[df['pressure_hpa'].notna()].copy()

    # Create instrument-channel labels
    df_pressure['obs_type'] = df_pressure.apply(create_instrument_channel_label, axis=1)

    # Aggregate by pressure level and observation type
    # Use sum_impact to show total contribution
    pivot_data = df_pressure.groupby(['pressure_hpa', 'obs_type'])['sum_impact'].sum().reset_index()

    # Create pivot table for heatmap
    heatmap_data = pivot_data.pivot(index='pressure_hpa', columns='obs_type', values='sum_impact')

    # Sort by pressure (high to low = surface to space)
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Fill NaN with 0 (means no observations at that level)
    heatmap_data = heatmap_data.fillna(0)

    # Calculate relative contribution (% of total at each level)
    # Positive values = harmful, negative = helpful
    relative_contrib = heatmap_data.copy()

    print("\n" + "="*80)
    print("TOTAL IMPACT BY OBSERVATION TYPE (across all pressure levels)")
    print("="*80)
    total_by_type = heatmap_data.sum(axis=0).sort_values(ascending=False)
    print(total_by_type.to_string())

    # Sort columns by total impact
    sorted_cols = total_by_type.index.tolist()
    relative_contrib = relative_contrib[sorted_cols]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [3, 1]})

    # Plot 1: Heatmap of contributions
    # Use diverging colormap: red=harmful, blue=helpful
    vmax = np.abs(relative_contrib.values).max()
    vmin = -vmax

    im = ax1.imshow(
        relative_contrib.values,
        cmap='RdBu_r',  # Red for positive (harmful), Blue for negative (helpful)
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Total Impact (harmful = positive, helpful = negative)', fontsize=11)

    # Set ticks and labels
    ax1.set_xticks(np.arange(len(relative_contrib.columns)))
    ax1.set_yticks(np.arange(len(relative_contrib.index)))
    ax1.set_xticklabels(relative_contrib.columns, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(relative_contrib.index, rotation=0, fontsize=10)

    ax1.set_xlabel('Observation Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pressure Level (hPa)', fontsize=12, fontweight='bold')
    ax1.set_title('Relative Contribution of Each Observation Type by Pressure Level\n' +
                  'Target: Radiosonde Temperature Forecast Error',
                  fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax1.set_xticks(np.arange(len(relative_contrib.columns)) - 0.5, minor=True)
    ax1.set_yticks(np.arange(len(relative_contrib.index)) - 0.5, minor=True)
    ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Plot 2: Vertical profile of total impact by pressure
    total_by_pressure = relative_contrib.sum(axis=1)

    ax2.plot(total_by_pressure.values, total_by_pressure.index,
             marker='o', linewidth=2, markersize=8, color='darkblue')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Pressure Level (hPa)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Total Impact', fontsize=12, fontweight='bold')
    ax2.set_title('Total Impact\nby Pressure', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # Add helpful/harmful labels
    ax2.text(0.98, 0.98, 'Harmful →', transform=ax2.transAxes,
             ha='right', va='top', fontsize=10, color='red', fontweight='bold')
    ax2.text(0.02, 0.98, '← Helpful', transform=ax2.transAxes,
             ha='left', va='top', fontsize=10, color='blue', fontweight='bold')

    plt.tight_layout()

    output_file = OUTPUT_DIR / 'all_instruments_by_pressure_level.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def plot_top_contributors_by_level(df):
    """
    Create a focused plot showing top contributors at key pressure levels.
    """
    # Filter to only records with pressure info
    df_pressure = df[df['pressure_hpa'].notna()].copy()

    # Create instrument-channel labels
    df_pressure['obs_type'] = df_pressure.apply(create_instrument_channel_label, axis=1)

    # Key pressure levels to examine
    key_levels = [1000, 850, 700, 500, 300, 250, 200, 150]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for idx, pressure in enumerate(key_levels):
        ax = axes[idx]

        # Get data for this pressure level
        level_data = df_pressure[df_pressure['pressure_hpa'] == pressure].copy()

        if len(level_data) == 0:
            ax.text(
                0.5,
                0.5,
                f'No data\nat {pressure} hPa',
                ha='center',
                va='center',
                fontsize=12,
            )
            ax.set_title(f'{pressure} hPa', fontsize=12, fontweight='bold')
            continue

        # Aggregate by observation type
        impact_by_type = level_data.groupby('obs_type')['sum_impact'].sum().sort_values()

        # Plot top 15 (or all if fewer)
        top_n = min(15, len(impact_by_type))

        # Get top harmful and top helpful
        top_harmful = impact_by_type.nlargest(8)
        top_helpful = impact_by_type.nsmallest(7)

        # Combine and sort
        to_plot = pd.concat([top_helpful, top_harmful]).sort_values()

        # Color code: red for harmful, blue for helpful
        colors = ['red' if x > 0 else 'blue' for x in to_plot.values]

        to_plot.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Total Impact', fontsize=10)
        ax.set_title(f'{pressure} hPa', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Adjust y-axis labels
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    plt.suptitle('Top Contributors to Radiosonde Temperature Forecast Error at Key Pressure Levels\n' +
                 '(Red = Harmful, Blue = Helpful)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = OUTPUT_DIR / 'top_contributors_by_pressure_level.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_satellite_vs_conventional(df):
    """
    Compare satellite vs conventional observations by pressure level.
    """
    df_pressure = df[df['pressure_hpa'].notna()].copy()

    # Categorize observations
    satellite_types = ['amsua', 'atms', 'ssmis', 'seviri_asr', 'seviri_csr', 'avhrr']
    conventional_types = ['radiosonde', 'aircraft', 'surface_obs']

    df_pressure['obs_category'] = df_pressure['instrument'].apply(
        lambda x: 'Satellite' if x in satellite_types else 'Conventional'
    )

    # Aggregate by pressure and category
    agg_data = df_pressure.groupby(['pressure_hpa', 'obs_category'])['sum_impact'].sum().reset_index()
    pivot = agg_data.pivot(index='pressure_hpa', columns='obs_category', values='sum_impact')
    pivot = pivot.sort_index(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Plot 1: Stacked bar chart
    pivot.plot(kind='barh', stacked=False, ax=ax1, color=['#ff6b6b', '#4ecdc4'], alpha=0.8)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Total Impact', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pressure Level (hPa)', fontsize=12, fontweight='bold')
    ax1.set_title('Satellite vs Conventional Observations\nby Pressure Level',
                  fontsize=14, fontweight='bold')
    ax1.legend(title='Observation Category', fontsize=11, title_fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Vertical profiles
    for col in pivot.columns:
        ax2.plot(
            pivot[col].values,
            pivot.index,
            marker='o',
            linewidth=2,
            markersize=8,
            label=col,
            alpha=0.8,
        )

    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Total Impact', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pressure Level (hPa)', fontsize=12, fontweight='bold')
    ax2.set_title('Vertical Profile:\nSatellite vs Conventional',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()

    output_file = OUTPUT_DIR / 'satellite_vs_conventional_by_pressure.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_satellite_channels_by_pressure(df, instruments):
    """Heatmaps of satellite channel impacts by pressure level per instrument."""
    df = _ensure_pressure_column(df)
    df_pressure = df[df['pressure_hpa'].notna()].copy()

    for inst in instruments:
        inst_df = df_pressure[df_pressure['instrument'] == inst]
        if inst_df.empty:
            print(f"✗ No {inst} data with pressure info; skipping {inst} heatmap")
            continue

        # Determine channel count from CHANNEL_NAMES fallback to max channel+1
        if inst in CHANNEL_NAMES:
            ch_map = CHANNEL_NAMES[inst]
            n_ch = max(ch_map.keys()) + 1 if ch_map else inst_df['channel'].max() + 1
        else:
            n_ch = int(inst_df['channel'].max()) + 1
            ch_map = {i: f"{inst} ch{i+1}" for i in range(n_ch)}

        col_keys = [(inst, ch) for ch in range(n_ch)]
        col_labels = [ch_map.get(ch, f"{inst} ch{ch+1}") for ch in range(n_ch)]

        # Build pressure labels, falling back to a single "All-levels" bin when missing
        inst_df = inst_df.copy()
        inst_df['pressure_label'] = inst_df['pressure_hpa'].apply(lambda x: int(x) if pd.notna(x) else 'All-levels')

        agg = inst_df.groupby(['pressure_label', 'instrument', 'channel'])['sum_impact'].sum().reset_index()
        pivot = agg.pivot_table(index='pressure_label', columns=['instrument', 'channel'], values='sum_impact', fill_value=0)
        pivot = pivot.reindex(columns=col_keys, fill_value=0)

        desired_index = [lvl for lvl in STANDARD_PRESSURE_LEVELS if lvl in pivot.index] + (['All-levels'] if 'All-levels' in pivot.index else [])
        pivot = pivot.reindex(desired_index).dropna(how='all')

        data = pivot.values
        if data.size == 0:
            print(f"✗ No data after pivot for {inst}; skipping")
            continue

        vmax = np.abs(data).max()
        vmin = -vmax if vmax != 0 else -1

        fig, ax = plt.subplots(figsize=(max(10, n_ch * 0.6), 8))
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax if vmax != 0 else 1, interpolation='nearest')

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(v) for v in pivot.index], fontsize=10)
        ax.set_xlabel(f'{INSTRUMENT_NAMES.get(inst, inst)} Channels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressure Level (hPa)', fontsize=12, fontweight='bold')
        ax.set_title(f'FSOI Impact by Pressure and Channel: {INSTRUMENT_NAMES.get(inst, inst)}',
                     fontsize=14, fontweight='bold', pad=12)

        ax.set_xticks(np.arange(len(col_labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(pivot.index)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.4, alpha=0.3)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Total Impact (harmful = positive, helpful = negative)', fontsize=11)

        plt.tight_layout()
        outfile = OUTPUT_DIR / f'{inst}_channels_by_pressure_heatmap.png'
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {outfile}")
        plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("RELATIVE CONTRIBUTION OF ALL OBSERVATION TYPES BY PRESSURE LEVEL")
    print("="*80)

    # Load data
    df, has_pressure = load_and_prepare_data()

    if not has_pressure:
        print("\n✗ ERROR: No pressure stratification found in data")
        print("  Make sure you've run FSOI with the updated code that extracts pressure levels")
        return

    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)

    # Create plots
    plot_relative_contribution_by_pressure(df)
    plot_top_contributors_by_level(df)
    plot_satellite_vs_conventional(df)
    plot_satellite_channels_by_pressure(df, instruments=['amsua', 'atms', 'ssmis', 'avhrr', 'ascat', 'seviri_asr', 'seviri_csr'])

    print("\n" + "="*80)
    print("✓ All plots created successfully!")
    print("="*80)
    print(f"\nOutput location: {OUTPUT_DIR}/")
    print("Files created:")
    print("  1. all_instruments_by_pressure_level.png - Heatmap of all observation types")
    print("  2. top_contributors_by_pressure_level.png - Top 15 at key levels")
    print("  3. satellite_vs_conventional_by_pressure.png - Category comparison")
    print("  4+. <instrument>_channels_by_pressure_heatmap.png - Channel impacts by pressure")


if __name__ == '__main__':
    main()
