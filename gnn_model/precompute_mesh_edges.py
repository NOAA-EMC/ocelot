"""
Pre-compute mesh prediction edges offline to avoid rtree multiprocessing issues.

This script creates the mesh→grid connections once and saves them to a file.
The training script then loads these pre-computed edges instead of calling
obs_mesh_conn at runtime.

Usage:
    python precompute_mesh_edges.py --config configs/mesh_config.yaml --output mesh_pred_edges.npz
"""

import argparse
import numpy as np
import torch
import yaml
from create_mesh_graph_global import create_mesh, obs_mesh_conn


def precompute_edges(mesh_config_path, output_path, mesh_resolution=6):
    """
    Pre-compute mesh prediction edges and save to file.

    Args:
        mesh_config_path: Path to mesh_config.yaml
        output_path: Path to output .npz file
        mesh_resolution: Mesh resolution (default: 6)
    """
    # Load mesh configuration
    print(f"Loading config from {mesh_config_path}...")
    with open(mesh_config_path, 'r') as f:
        mesh_config = yaml.safe_load(f)

    if not mesh_config.get('enable_mesh_pred', False):
        print("Mesh-grid variables not enabled in config. Exiting.")
        return

    mesh_instruments = list(mesh_config.get('variables', {}).keys())
    print(f"Mesh-grid instruments: {mesh_instruments}")

    if not mesh_instruments:
        print("No mesh-grid instruments found. Exiting.")
        return

    # Create mesh structure
    print(f"Creating mesh with resolution {mesh_resolution}...")
    mesh_structure = create_mesh(splits=mesh_resolution, levels=4, hierarchical=False, plot=False)

    # Get mesh coordinates (mesh grid points)
    mesh_latlon = mesh_structure["mesh_lat_lon_list"][-1]
    mesh_lats = mesh_latlon[:, 0].copy()  # Use .copy() to create standalone arrays
    mesh_lons = mesh_latlon[:, 1].copy()

    print(f"Mesh grid:")
    print(f"  Lat range: [{mesh_lats.min():.2f}, {mesh_lats.max():.2f}]")
    print(f"  Lon range: [{mesh_lons.min():.2f}, {mesh_lons.max():.2f}]")
    print(f"  Grid points: {len(mesh_lats)}")

    # Pre-compute edges for each mesh-grid instrument
    edges_data = {
        'lats': mesh_lats,
        'lons': mesh_lons,
        'num_nodes': len(mesh_lats)
    }

    print("\nComputing mesh→grid edges (shared for all mesh-grid instruments)...")
    edge_index, edge_attr = obs_mesh_conn(
        mesh_lats,
        mesh_lons,
        mesh_structure["m2m_graphs"],
        mesh_structure["mesh_lat_lon_list"],
        mesh_structure["mesh_list"],
        o2m=False
    )

    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Edge attr shape: {edge_attr.shape}")

    # Convert to numpy once and reuse for each instrument
    edge_index_np = edge_index.cpu().numpy()
    edge_attr_np = edge_attr.cpu().numpy()
    for inst_name in mesh_instruments:
        print(f"\nStoring edges for {inst_name}...")
        # Use the same mesh→grid edges for each instrument
        edges_data[f'{inst_name}_edge_index'] = edge_index_np
        edges_data[f'{inst_name}_edge_attr'] = edge_attr_np
        # Note: edge_attr contains 4D GraphCast-style spatial features
        # These are consumed by BipartiteGAT with edge_dim=bipartite_edge_attr_dim

    # Save to file
    print(f"\nSaving to {output_path}...")
    np.savez(output_path, **edges_data)

    # Verify saved file
    print(f"\nVerifying saved file...")
    loaded = np.load(output_path)
    print(f"Saved arrays:")
    for key in loaded.files:
        print(f"  {key}: shape {loaded[key].shape}, dtype {loaded[key].dtype}")

    print(f"\n Successfully saved mesh prediction edges to {output_path}")
    print(f"   Grid points: {len(mesh_lats)}")
    print(f"   Instruments: {mesh_instruments}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute mesh prediction edges")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mesh_config.yaml",
        help="Path to mesh config YAML file"
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
