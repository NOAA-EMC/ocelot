#!/usr/bin/env python
"""
Plot FSOI impact by pressure level for radiosonde and aircraft observations.

This script creates visualizations showing the relative contribution of each
observation type to forecast error at different pressure levels.

Author: Azadeh Gholoubi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Standard pressure levels used in the model
STANDARD_PRESSURE_LEVELS = np.array([
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
])


def load_fsoi_data(data_dir):
    """Load FSOI CSV files."""
    data_dir = Path(data_dir)

    by_channel = pd.read_csv(data_dir / 'csv' / 'fsoi_by_channel.csv')
    by_instrument = pd.read_csv(data_dir / 'csv' / 'fsoi_by_instrument.csv')

    return by_channel, by_instrument


def map_channels_to_pressure(df, instrument='radiosonde'):
    """
    Map radiosonde/aircraft channels to pressure levels and variables.

    For radiosonde:
        Channel 0: temperature
        Channel 1: dewpoint
        Channel 2: u_wind
        Channel 3: v_wind

    For aircraft:
        Channel 0: temperature
        Channel 1: humidity (specific humidity)
        Channel 2: u_wind
        Channel 3: v_wind

    NOTE: Current FSOI doesn't stratify by actual pressure levels,
    only by channel (variable type). This function prepares the data
    structure for when pressure stratification is added.
    """
    df = df[df['instrument'] == instrument].copy()

    if instrument == 'radiosonde':
        channel_map = {
            0: 'temperature',
            1: 'dewpoint',
            2: 'u_wind',
            3: 'v_wind'
        }
    elif instrument == 'aircraft':
        channel_map = {
            0: 'temperature',
            1: 'humidity',
            2: 'u_wind',
            3: 'v_wind'
        }
    else:
        return df

    df['variable'] = df['channel'].map(channel_map)

    return df


def create_pressure_level_plot(by_channel, output_dir):
    """
    Create stacked bar plot showing contribution by observation type and variable.

    Since current FSOI doesn't have true pressure stratification, this shows
    the aggregate impact by variable type as a placeholder.
    """
    # Filter for conventional obs (radiosonde and aircraft)
    conventional = by_channel[by_channel['instrument'].isin(['radiosonde', 'aircraft'])].copy()

    if conventional.empty:
        print("No conventional observation data found")
        return

    # Map channels to variables
    radiosonde_data = map_channels_to_pressure(conventional, 'radiosonde')
    aircraft_data = map_channels_to_pressure(conventional, 'aircraft')

    # Aggregate by pair to get consistent statistics
    radiosonde_agg = radiosonde_data.groupby(['pair_idx', 'variable']).agg({
        'sum_impact': 'sum',
        'total_count': 'sum'
    }).reset_index()

    aircraft_agg = aircraft_data.groupby(['pair_idx', 'variable']).agg({
        'sum_impact': 'sum',
        'total_count': 'sum'
    }).reset_index()

    # Average across pairs
    radiosonde_final = radiosonde_agg.groupby('variable').agg({
        'sum_impact': 'mean',
        'total_count': 'mean'
    }).reset_index()
    radiosonde_final['instrument'] = 'radiosonde'

    aircraft_final = aircraft_agg.groupby('variable').agg({
        'sum_impact': 'mean',
        'total_count': 'mean'
    }).reset_index()
    aircraft_final['instrument'] = 'aircraft'

    # Combine
    combined = pd.concat([radiosonde_final, aircraft_final], ignore_index=True)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Total Impact by Variable
    pivot_impact = combined.pivot(index='variable', columns='instrument', values='sum_impact')
    pivot_impact.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'])
    ax1.set_title('Mean FSOI Impact by Variable\n(Averaged across forecast pairs)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Variable', fontsize=11)
    ax1.set_ylabel('Mean Impact', fontsize=11)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.legend(title='Observation Type', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Observation Count by Variable
    pivot_count = combined.pivot(index='variable', columns='instrument', values='total_count')
    pivot_count.plot(kind='bar', ax=ax2, color=['#2E86AB', '#A23B72'])
    ax2.set_title('Mean Observation Count by Variable\n(Averaged across forecast pairs)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Variable', fontsize=11)
    ax2.set_ylabel('Mean Count', fontsize=11)
    ax2.legend(title='Observation Type', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    output_file = Path(output_dir) / 'figures' / 'fsoi_by_variable.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Print summary
    print("\n" + "="*80)
    print("FSOI Impact Summary by Variable")
    print("="*80)
    print("\nRadiosonde:")
    print(radiosonde_final[['variable', 'sum_impact', 'total_count']].to_string(index=False))
    print("\nAircraft:")
    print(aircraft_final[['variable', 'sum_impact', 'total_count']].to_string(index=False))
    print("\n" + "="*80)


def create_relative_contribution_plot(by_channel, by_instrument, output_dir):
    """
    Create plot showing relative contribution of each obs type to total impact.
    """
    # Get conventional obs
    conventional_inst = by_instrument[by_instrument['instrument'].isin(['radiosonde', 'aircraft'])].copy()

    if conventional_inst.empty:
        print("No conventional observation data for relative contribution")
        return

    # Aggregate by pair first
    paired_agg = conventional_inst.groupby(['pair_idx', 'instrument']).agg({
        'sum_impact': 'sum'
    }).reset_index()

    # Average across pairs
    final_agg = paired_agg.groupby('instrument')['sum_impact'].mean()

    # Create bar chart instead of pie (since we have negative values)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2E86AB' if val < 0 else '#A23B72' for val in final_agg.values]
    bars = ax.barh(final_agg.index, final_agg.values, color=colors)

    ax.set_title('Mean Impact on Forecast Error\n(Radiosonde vs Aircraft)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Mean Impact', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (inst, val) in enumerate(zip(final_agg.index, final_agg.values)):
        ax.text(val, i, f'  {val:.1f}', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_file = Path(output_dir) / 'figures' / 'fsoi_relative_contribution.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_combined_visualization(by_channel, by_instrument, output_dir):
    """
    Create a comprehensive multi-panel visualization.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Filter conventional obs
    conventional_ch = by_channel[by_channel['instrument'].isin(['radiosonde', 'aircraft'])].copy()
    conventional_inst = by_instrument[by_instrument['instrument'].isin(['radiosonde', 'aircraft'])].copy()

    if conventional_ch.empty or conventional_inst.empty:
        print("Insufficient data for combined visualization")
        return

    # Map channels to variables
    radiosonde_data = map_channels_to_pressure(conventional_ch, 'radiosonde')
    aircraft_data = map_channels_to_pressure(conventional_ch, 'aircraft')

    # Panel 1: Impact by variable (radiosonde)
    ax1 = fig.add_subplot(gs[0, 0])
    rad_summary = radiosonde_data.groupby(['pair_idx', 'variable'])['sum_impact'].sum().reset_index()
    rad_mean = rad_summary.groupby('variable')['sum_impact'].mean().sort_values()
    rad_mean.plot(kind='barh', ax=ax1, color='#2E86AB')
    ax1.set_title('Radiosonde: Mean Impact by Variable', fontweight='bold')
    ax1.set_xlabel('Mean Impact')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Impact by variable (aircraft)
    ax2 = fig.add_subplot(gs[0, 1])
    air_summary = aircraft_data.groupby(['pair_idx', 'variable'])['sum_impact'].sum().reset_index()
    air_mean = air_summary.groupby('variable')['sum_impact'].mean().sort_values()
    air_mean.plot(kind='barh', ax=ax2, color='#A23B72')
    ax2.set_title('Aircraft: Mean Impact by Variable', fontweight='bold')
    ax2.set_xlabel('Mean Impact')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)

    # Panel 3: Harmful fraction by variable (radiosonde)
    ax3 = fig.add_subplot(gs[1, 0])
    rad_harmful = radiosonde_data.groupby('variable')['positive_frac'].mean().sort_values() * 100
    rad_harmful.plot(kind='barh', ax=ax3, color='#2E86AB', alpha=0.7)
    ax3.set_title('Radiosonde: % Harmful Observations', fontweight='bold')
    ax3.set_xlabel('% Harmful')
    ax3.set_xlim(0, 100)
    ax3.grid(axis='x', alpha=0.3)

    # Panel 4: Harmful fraction by variable (aircraft)
    ax4 = fig.add_subplot(gs[1, 1])
    air_harmful = aircraft_data.groupby('variable')['positive_frac'].mean().sort_values() * 100
    air_harmful.plot(kind='barh', ax=ax4, color='#A23B72', alpha=0.7)
    ax4.set_title('Aircraft: % Harmful Observations', fontweight='bold')
    ax4.set_xlabel('% Harmful')
    ax4.set_xlim(0, 100)
    ax4.grid(axis='x', alpha=0.3)

    # Panel 5-6: Side-by-side comparison
    ax5 = fig.add_subplot(gs[2, :])

    # Prepare data for comparison
    rad_final = rad_mean.reset_index()
    rad_final['instrument'] = 'Radiosonde'
    air_final = air_mean.reset_index()
    air_final['instrument'] = 'Aircraft'
    combined = pd.concat([rad_final, air_final])
    combined.columns = ['variable', 'impact', 'instrument']

    # Create grouped bar chart
    variables = combined['variable'].unique()
    x = np.arange(len(variables))
    width = 0.35

    rad_vals = [
        combined[(combined['variable'] == v) & (combined['instrument'] == 'Radiosonde')]['impact'].values[0]
        if len(combined[(combined['variable'] == v) & (combined['instrument'] == 'Radiosonde')]) > 0 else 0
        for v in variables
    ]
    air_vals = [
        combined[(combined['variable'] == v) & (combined['instrument'] == 'Aircraft')]['impact'].values[0]
        if len(combined[(combined['variable'] == v) & (combined['instrument'] == 'Aircraft')]) > 0 else 0
        for v in variables
    ]

    ax5.bar(x - width/2, rad_vals, width, label='Radiosonde', color='#2E86AB')
    ax5.bar(x + width/2, air_vals, width, label='Aircraft', color='#A23B72')

    ax5.set_title('Comparative Impact by Variable', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Variable', fontsize=12)
    ax5.set_ylabel('Mean Impact', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels(variables)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    output_file = Path(output_dir) / 'figures' / 'fsoi_comprehensive_analysis.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function to create all plots."""
    # Path to FSOI output directory
    data_dir = Path("FSOI/fsoi_outputs/radiosonde_temp_impact")

    if not (data_dir / 'csv' / 'fsoi_by_channel.csv').exists():
        print(f"Error: FSOI data not found in {data_dir}")
        print("Expected: FSOI/fsoi_outputs/radiosonde_temp_impact/csv/fsoi_by_channel.csv")
        return

    print("Loading FSOI data...")
    by_channel, by_instrument = load_fsoi_data(data_dir)

    print(f"\nFound {len(by_channel)} channel-level records")
    print(f"Found {len(by_instrument)} instrument-level records")
    print(f"Instruments: {by_instrument['instrument'].unique()}")

    print("\nCreating visualizations...")

    # Create plots
    create_pressure_level_plot(by_channel, data_dir)
    create_relative_contribution_plot(by_channel, by_instrument, data_dir)
    create_combined_visualization(by_channel, by_instrument, data_dir)

    print("\n" + "="*80)
    print("✓ All plots created successfully")
    print("="*80)
    print(f"\nOutput directory: {data_dir / 'figures'}")

    # NOTE about pressure levels
    print("\n" + "="*80)
    print("IMPORTANT NOTE: Pressure Level Stratification")
    print("="*80)
    print("""
The current FSOI implementation stratifies by CHANNEL (variable type) but not
by actual PRESSURE LEVEL. To add true pressure-level stratification:

1. Modify fsoi_utils.py to extract pressure_level from batch:
   - In get_fsoi_inputs(): Also extract pressure_level for radiosonde/aircraft
   - Store as metadata dict alongside fsoi_values

2. Modify aggregate_fsoi_by_channel() to include pressure_level:
   - Group by (instrument, channel, pressure_level)
   - Map pressure_level indices to standard levels (1000, 925, 850, ...)

3. The validation CSV files from training already contain this info:
   - Column: pressure_level_idx (0-15)
   - Column: pressure_level_label (e.g., "850hPa")

Would you like me to implement these modifications to the FSOI code?
""")


if __name__ == "__main__":
    main()
