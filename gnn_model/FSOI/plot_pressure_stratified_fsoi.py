#!/usr/bin/env python
"""
Plot FSOI impact by pressure level for radiosonde and aircraft observations.

This script creates visualizations showing the relative contribution of each
observation type to forecast error at different pressure levels, using the
pressure-stratified FSOI data.

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

# Variable name mapping
VARIABLE_NAMES = {
    0: 'temperature',
    1: 'dewpoint',      # radiosonde
    2: 'u_wind',
    3: 'v_wind',
}

AIRCRAFT_VARIABLE_NAMES = {
    0: 'temperature',
    1: 'humidity',  # aircraft
    2: 'u_wind',
    3: 'v_wind',
}


def load_fsoi_data(data_dir):
    """Load FSOI CSV files."""
    data_dir = Path(data_dir)

    by_channel = pd.read_csv(data_dir / 'csv' / 'fsoi_by_channel.csv')
    by_instrument = pd.read_csv(data_dir / 'csv' / 'fsoi_by_instrument.csv')

    return by_channel, by_instrument


def plot_radiosonde_by_pressure_and_variable(df, output_dir):
    """
    Create heatmap showing radiosonde FSOI impact by pressure level and variable.
    """
    # Filter radiosonde data
    rad_data = df[df['instrument'] == 'radiosonde'].copy()

    if rad_data.empty:
        print("No radiosonde data found")
        return

    # Check if we have pressure level data
    if 'pressure_hpa' not in rad_data.columns:
        print("No pressure level stratification in data")
        return

    # Map channel to variable name
    rad_data['variable'] = rad_data['channel'].map(VARIABLE_NAMES)

    # Aggregate by pressure level and variable
    pivot_data = rad_data.groupby(['pressure_hpa', 'variable'])['mean_impact'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='pressure_hpa', columns='variable', values='mean_impact')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Heatmap
    im = ax1.imshow(pivot_table.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_yticks(range(len(pivot_table.index)))
    ax1.set_yticklabels([f"{int(p)}" for p in pivot_table.index])
    ax1.set_xticks(range(len(pivot_table.columns)))
    ax1.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
    ax1.set_ylabel('Pressure (hPa)', fontsize=12)
    ax1.set_xlabel('Variable', fontsize=12)
    ax1.set_title('Radiosonde: Mean FSOI Impact by Pressure Level', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Mean Impact', fontsize=11)

    # Plot 2: Line plot
    for var in pivot_table.columns:
        ax2.plot(pivot_table[var], pivot_table.index, marker='o', label=var, linewidth=2)

    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Mean Impact', fontsize=12)
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)
    ax2.set_title('Radiosonde: Vertical Profile of FSOI Impact', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()  # Higher pressure at bottom
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = Path(output_dir) / 'figures' / 'radiosonde_by_pressure_level.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_aircraft_by_pressure_and_variable(df, output_dir):
    """
    Create heatmap showing aircraft FSOI impact by pressure level and variable.
    """
    # Filter aircraft data
    air_data = df[df['instrument'] == 'aircraft'].copy()

    if air_data.empty:
        print("No aircraft data found")
        return

    # Check if we have pressure level data
    if 'pressure_hpa' not in air_data.columns:
        print("No pressure level stratification in data")
        return

    # Map channel to variable name
    air_data['variable'] = air_data['channel'].map(AIRCRAFT_VARIABLE_NAMES)

    # Aggregate by pressure level and variable
    pivot_data = air_data.groupby(['pressure_hpa', 'variable'])['mean_impact'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='pressure_hpa', columns='variable', values='mean_impact')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Heatmap
    im = ax1.imshow(pivot_table.values, aspect='auto', cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    ax1.set_yticks(range(len(pivot_table.index)))
    ax1.set_yticklabels([f"{int(p)}" for p in pivot_table.index])
    ax1.set_xticks(range(len(pivot_table.columns)))
    ax1.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
    ax1.set_ylabel('Pressure (hPa)', fontsize=12)
    ax1.set_xlabel('Variable', fontsize=12)
    ax1.set_title('Aircraft: Mean FSOI Impact by Pressure Level', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Mean Impact', fontsize=11)

    # Plot 2: Line plot
    for var in pivot_table.columns:
        ax2.plot(pivot_table[var], pivot_table.index, marker='o', label=var, linewidth=2)

    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Mean Impact', fontsize=12)
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)
    ax2.set_title('Aircraft: Vertical Profile of FSOI Impact', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()  # Higher pressure at bottom
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = Path(output_dir) / 'figures' / 'aircraft_by_pressure_level.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_comparative_pressure_profiles(df, output_dir):
    """
    Create side-by-side comparison of radiosonde and aircraft vertical profiles.
    """
    # Filter data
    rad_data = df[df['instrument'] == 'radiosonde'].copy()
    air_data = df[df['instrument'] == 'aircraft'].copy()

    if rad_data.empty or air_data.empty:
        print("Need both radiosonde and aircraft data for comparison")
        return

    if 'pressure_hpa' not in rad_data.columns or 'pressure_hpa' not in air_data.columns:
        print("No pressure level stratification in data")
        return

    # Map channels to variables
    rad_data['variable'] = rad_data['channel'].map(VARIABLE_NAMES)
    air_data['variable'] = air_data['channel'].map(AIRCRAFT_VARIABLE_NAMES)

    # Focus on temperature for comparison
    rad_temp = rad_data[rad_data['variable'] == 'temperature'].groupby('pressure_hpa')['mean_impact'].mean()
    air_temp = air_data[air_data['variable'] == 'temperature'].groupby('pressure_hpa')['mean_impact'].mean()
    air_humid = air_data[air_data['variable'] == 'humidity'].groupby('pressure_hpa')['mean_impact'].mean()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Plot 1: Temperature comparison
    ax1.plot(rad_temp, rad_temp.index, marker='o', label='Radiosonde', linewidth=2, color='#2E86AB')
    ax1.plot(air_temp, air_temp.index, marker='s', label='Aircraft', linewidth=2, color='#A23B72')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Mean Impact', fontsize=12)
    ax1.set_ylabel('Pressure (hPa)', fontsize=12)
    ax1.set_title('Temperature: Radiosonde vs Aircraft', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Humidity variables
    rad_dewpoint = rad_data[rad_data['variable'] == 'dewpoint'].groupby('pressure_hpa')['mean_impact'].mean()
    ax2.plot(rad_dewpoint, rad_dewpoint.index, marker='o', label='Radiosonde Dewpoint', linewidth=2, color='#2E86AB')
    ax2.plot(air_humid, air_humid.index, marker='s', label='Aircraft Humidity', linewidth=2, color='#A23B72')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Mean Impact', fontsize=12)
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)
    ax2.set_title('Moisture Variables: Dewpoint vs Specific Humidity', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = Path(output_dir) / 'figures' / 'comparative_pressure_profiles.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    """Main function to create all pressure-stratified plots."""
    # Path to FSOI output directory
    data_dir = Path("FSOI/fsoi_outputs/radiosonde_temp_impact")

    if not (data_dir / 'csv' / 'fsoi_by_channel.csv').exists():
        print(f"Error: FSOI data not found in {data_dir}")
        print("Expected: FSOI/fsoi_outputs/radiosonde_temp_impact/csv/fsoi_by_channel.csv")
        return

    print("Loading FSOI data...")
    by_channel, by_instrument = load_fsoi_data(data_dir)

    print(f"\nFound {len(by_channel)} channel-level records")
    print(f"Columns: {by_channel.columns.tolist()}")

    # Check if we have pressure stratification
    if 'pressure_hpa' in by_channel.columns:
        print("\n✓ Pressure level stratification FOUND in data!")
        print(f"Pressure levels present: {sorted(by_channel['pressure_hpa'].unique())}")

        print("\nCreating pressure-stratified visualizations...")
        plot_radiosonde_by_pressure_and_variable(by_channel, data_dir)
        plot_aircraft_by_pressure_and_variable(by_channel, data_dir)
        plot_comparative_pressure_profiles(by_channel, data_dir)

        print("\n" + "="*80)
        print("✓ All pressure-stratified plots created successfully")
        print("="*80)
    else:
        print("\n✗ No pressure level stratification found in data")
        print("The FSOI data only contains channel-level aggregation (by variable)")
        print("\nTo get pressure-stratified results, you need to:")
        print("1. Re-run the FSOI analysis with the updated code")
        print("2. The modifications to fsoi_utils.py and fsoi_inference.py will")
        print("   automatically extract and include pressure level information")


if __name__ == "__main__":
    main()
