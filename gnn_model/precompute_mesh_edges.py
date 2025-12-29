"""
Pre-compute mesh prediction edges offline to avoid rtree multiprocessing issues.

This script creates the mesh→grid connections once and saves them to a file.
The training script then loads these pre-computed edges instead of calling
obs_mesh_conn at runtime.

Usage:
    python precompute_mesh_edges.py --config target_config.yaml --output mesh_pred_edges.npz
"""

import argparse
import numpy as np
import torch
import yaml
from create_mesh_graph_global import create_mesh, obs_mesh_conn


def precompute_edges(target_config_path, output_path, mesh_resolution=6):
    """
    Pre-compute mesh prediction edges and save to file.

    Args:
        target_config_path: Path to target_config.yaml
        output_path: Path to output .npz file
        mesh_resolution: Mesh resolution (default: 6)
    """
    # Load target configuration
    print(f"Loading config from {target_config_path}...")
    with open(target_config_path, 'r') as f:
        target_config = yaml.safe_load(f)

    if not target_config.get('enabled', False):
        print("Target variables not enabled in config. Exiting.")
        return

    target_instruments = list(target_config.get('variables', {}).keys())
    print(f"Target instruments: {target_instruments}")

    if not target_instruments:
        print("No target instruments found. Exiting.")
        return

    # Create mesh structure
    print(f"Creating mesh with resolution {mesh_resolution}...")
    mesh_structure = create_mesh(splits=mesh_resolution, levels=4, hierarchical=False, plot=False)

    # Get mesh coordinates (target grid points)
    mesh_latlon = mesh_structure["mesh_lat_lon_list"][-1]
    target_lats = mesh_latlon[:, 0].copy()  # Use .copy() to create standalone arrays
    target_lons = mesh_latlon[:, 1].copy()

    print(f"Mesh grid:")
    print(f"  Lat range: [{target_lats.min():.2f}, {target_lats.max():.2f}]")
    print(f"  Lon range: [{target_lons.min():.2f}, {target_lons.max():.2f}]")
    print(f"  Grid points: {len(target_lats)}")

    # Pre-compute edges for each target instrument
    edges_data = {
        'lats': target_lats,
        'lons': target_lons,
        'num_nodes': len(target_lats)
    }

    for inst_name in target_instruments:
        print(f"\nComputing edges for {inst_name}...")

        edge_index, edge_attr = obs_mesh_conn(
            target_lats,
            target_lons,
            mesh_structure["m2m_graphs"],
            mesh_structure["mesh_lat_lon_list"],
            mesh_structure["mesh_list"],
            o2m=False
        )

        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Edge attr shape: {edge_attr.shape}")

        # Convert to numpy and store
        edges_data[f'{inst_name}_edge_index'] = edge_index.cpu().numpy()
        # Note: We don't save edge_attr because model creates zeros with hidden_dim

    # Save to file
    print(f"\nSaving to {output_path}...")
    np.savez(output_path, **edges_data)

    # Verify saved file
    print(f"\nVerifying saved file...")
    loaded = np.load(output_path)
    print(f"Saved arrays:")
    for key in loaded.files:
        print(f"  {key}: shape {loaded[key].shape}, dtype {loaded[key].dtype}")

    print(f"\n✅ Successfully saved mesh prediction edges to {output_path}")
    print(f"   Grid points: {len(target_lats)}")
    print(f"   Instruments: {target_instruments}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute mesh prediction edges")
    parser.add_argument(
        "--config",
        type=str,
        default="target_config.yaml",
        help="Path to target config YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mesh_pred_edges.npz",
        help="Output path for .npz file"
    )
    parser.add_argument(
        "--mesh_resolution",
        type=int,
        default=6,
        help="Mesh resolution (default: 6)"
    )

    args = parser.parse_args()
    precompute_edges(args.config, args.output, args.mesh_resolution)
