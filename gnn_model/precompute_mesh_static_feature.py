"""
precompute_static_mesh_features.py
====================================
Standalone, one-time script that interpolates two static datasets onto the
icosahedral mesh nodes and saves the results to disk.

Inputs
------
  gfs_land_mask.nc              – GFS land/sea mask, UNSTRUCTURED (Location dim)
  topography.gmted2010.30s.nc   – GMTED2010 30 arc-second terrain elevation,
                                   REGULAR GRID (jdim x idim)

Outputs
-------
  static_mesh_features.npz
      Contains one entry per mesh level:
        land_mask_level_{i}       shape (N_i,)  float32  0=ocean, 1=land
        elevation_level_{i}       shape (N_i,)  float32  metres, raw
        elevation_norm_level_{i}  shape (N_i,)  float32  tanh-normalised
        mesh_lat_level_{i}        shape (N_i,)  float32  degrees
        mesh_lon_level_{i}        shape (N_i,)  float32  degrees

Normalisation
-------------
  elev_norm = tanh(elev / SCALE)  with SCALE=3000 m  (signed: oceans negative)

Usage
-----
  python precompute_static_mesh_features.py \
      --land_mask  /path/to/gfs_land_mask.nc \
      --topography /path/to/topography.gmted2010.30s.nc \
      --output     static_mesh_features.npz \
      --splits     6 \
      --levels     4
"""

import argparse
import sys
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------
try:
    from create_mesh_graph_global import create_mesh, vertice_cart_to_lat_lon
except ModuleNotFoundError:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gnn_model"))
    from create_mesh_graph_global import create_mesh, vertice_cart_to_lat_lon


# ---------------------------------------------------------------------------
# Land mask  (UNSTRUCTURED – 1-D Location array)
# ---------------------------------------------------------------------------

def load_land_mask_unstructured(path: str):
    """
    Load the GFS land mask from an unstructured (Location-dim) NetCDF file.

    File structure (from ncdump -h):
        dimensions:   Location = 1038240
        variables:
            float latitude(Location)    degrees_north
            float longitude(Location)   degrees_east  [-180, 180]
            float land_mask(Location)   0=ocean, 1=land

    Returns
    -------
    tree        : cKDTree built on unit-sphere XYZ of the 1038240 source points
    mask_values : (N_valid,) float32  corresponding land_mask values
    """
    print(f"[LAND MASK] Loading unstructured file: {path}")
    ds = xr.open_dataset(path)

    lats = ds["latitude"].values.astype(np.float64)
    lons = ds["longitude"].values.astype(np.float64)
    mask = ds["land_mask"].values.astype(np.float32)

    # Mask out fill values (3.402823e+38)
    fill_threshold = 1e+37
    valid = (
        (np.abs(lats) < fill_threshold) &
        (np.abs(lons) < fill_threshold) &
        (mask < fill_threshold) &
        np.isfinite(lats) &
        np.isfinite(lons) &
        np.isfinite(mask)
    )
    lats  = lats[valid]
    lons  = lons[valid]
    mask  = mask[valid]

    print(f"  Valid points  : {valid.sum():,} / {len(valid):,}")
    print(f"  Lat range     : [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"  Lon range     : [{lons.min():.2f}, {lons.max():.2f}]")
    print(f"  Land fraction : {mask.mean()*100:.1f}%")

    # Build KD-tree on unit-sphere XYZ for great-circle-correct nearest neighbour
    xyz  = _latlon_to_xyz(lats, lons)   # (N, 3)
    tree = cKDTree(xyz)
    print(f"  KD-tree built")
    return tree, mask


def query_land_mask(tree: cKDTree, mask_values: np.ndarray,
                    mesh_lat_deg: np.ndarray, mesh_lon_deg: np.ndarray,
                    k: int = 1) -> np.ndarray:
    """
    Nearest-neighbour land mask lookup for mesh nodes.

    k=1  → hard nearest neighbour (recommended for a binary mask)
    k>1  → average of k neighbours (gives soft fractional values)
    """
    query_xyz = _latlon_to_xyz(mesh_lat_deg, mesh_lon_deg)
    if k == 1:
        _, idx = tree.query(query_xyz, k=1, workers=-1)
        return mask_values[idx].astype(np.float32)
    else:
        _, idxs = tree.query(query_xyz, k=k, workers=-1)   # (N, k)
        return mask_values[idxs].mean(axis=1).astype(np.float32)


def _latlon_to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert lat/lon (degrees) to unit-sphere Cartesian (N, 3)."""
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Topography  (REGULAR GRID – jdim x idim)
# ---------------------------------------------------------------------------

def load_topography_regular(path: str):
    """
    Load GMTED2010 from a regular-grid NetCDF file.

    File structure (from ncdump -h):
        dimensions:  jdim=21600, idim=43200
        variables:
            double lat(jdim)           grid cell centre latitude
            double lon(idim)           grid cell centre longitude
            short  topo(jdim, idim)    topography metres

    Returns
    -------
    interp : RegularGridInterpolator
        Call as interp(np.stack([lat_deg, lon_deg], axis=1)) -> elev_m (N,)
    """
    print(f"\n[TOPOGRAPHY] Loading regular-grid file: {path}")
    ds = xr.open_dataset(path)

    lats = ds["lat"].values.astype(np.float64)    # (21600,)
    lons = ds["lon"].values.astype(np.float64)    # (43200,)

    print(f"  Reading topo array ({lats.shape[0]:,} × {lons.shape[0]:,}) ...")
    topo = ds["topo"].values.astype(np.float64)   # (21600, 43200)  int16 → float64

    print(f"  Lat range  : [{lats.min():.4f}, {lats.max():.4f}]")
    print(f"  Lon range  : [{lons.min():.4f}, {lons.max():.4f}]")
    print(f"  Elev range : [{topo.min():.0f}, {topo.max():.0f}] m")

    # GMTED2010 lat is typically descending (90 → -90); RegularGridInterpolator
    # requires strictly ascending axes.
    if lats[-1] < lats[0]:
        lats = lats[::-1]
        topo = topo[::-1, :]
        print(f"  Flipped lat to ascending order")

    # GMTED2010 lon is already [-180, 180] based on ncdump; guard anyway.
    if lons.max() > 180.1:
        lons = ((lons + 180.0) % 360.0) - 180.0
        sort_idx = np.argsort(lons)
        lons     = lons[sort_idx]
        topo     = topo[:, sort_idx]
        print(f"  Converted lon to [-180, 180]")

    interp = RegularGridInterpolator(
        (np.ascontiguousarray(lats), np.ascontiguousarray(lons)),
        np.ascontiguousarray(topo),
        method="linear",
        bounds_error=False,
        fill_value=None,    # extrapolate at poles rather than return NaN
    )
    print(f"  RegularGridInterpolator ready")
    return interp


def query_topography(interp: RegularGridInterpolator,
                     mesh_lat_deg: np.ndarray,
                     mesh_lon_deg: np.ndarray) -> np.ndarray:
    """Sample topography at mesh node positions. Returns (N,) float32 metres."""
    lon_clamped = np.clip(mesh_lon_deg, -180.0, 180.0)
    query = np.stack([mesh_lat_deg, lon_clamped], axis=1)   # (N, 2)
    return interp(query).astype(np.float32)


# ---------------------------------------------------------------------------
# Elevation normalisation
# ---------------------------------------------------------------------------

def normalize_elevation(elev: np.ndarray, scale: float = 3000.0) -> np.ndarray:
    """
    Signed tanh normalisation centred on 0 m:
        elev_norm = tanh(elev / scale)

    With scale=3000 m:
      +8848 m (Everest)  → +0.9997
          0 m (sea level)→  0.000
       -200 m            → -0.066
      -8376 m (Mariana)  → -0.9994
    """
    return np.tanh(elev / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute static mesh features (land mask + elevation)."
    )
    parser.add_argument("--land_mask",   default="/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gcgfs/data/gfs_land_mask.nc",
                        help="Path to gfs_land_mask.nc  (unstructured, Location dim)")
    parser.add_argument("--topography",  default="/scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gcgfs/data/topography.gmted2010.30s.nc",
                        help="Path to topography.gmted2010.30s.nc  (regular grid, jdim x idim)")
    parser.add_argument("--output",      default="static_mesh_features.npz",
                        help="Output .npz path  (default: static_mesh_features.npz)")
    parser.add_argument("--splits",      type=int, default=6,
                        help="Mesh splits – must match training config  (default: 6)")
    parser.add_argument("--levels",      type=int, default=4,
                        help="Mesh levels to keep – must match training config  (default: 4)")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Use hierarchical mesh (multi-level). "
                             "Match the --mesh_type used in training: "
                             "omit for fixed/GraphCast mesh (1 merged level), "
                             "set for hierarchical mesh (levels 0..N-1).")
    parser.add_argument("--elev_scale",  type=float, default=3000.0,
                        help="tanh normalisation scale in metres  (default: 3000)")
    parser.add_argument("--land_mask_k", type=int, default=1,
                        help="Neighbours to average for land mask KD-tree lookup  (default: 1)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Build mesh (CPU-only, no GPU required)
    #
    #  hierarchical=False  →  create_mesh merges all levels into ONE mesh
    #                          m2m_graphs has 1 entry  →  level_0 only
    #                          Use this when training with --mesh_type fixed
    #
    #  hierarchical=True   →  create_mesh keeps each level separate
    #                          m2m_graphs has `levels` entries  →  level_0 … level_N
    #                          Use this when training with --mesh_type hierarchical
    # ------------------------------------------------------------------
    print(f"\n[MESH] Building mesh: splits={args.splits}, levels={args.levels}, "
          f"hierarchical={args.hierarchical} ...")
    mesh_structure = create_mesh(
        splits=args.splits,
        levels=args.levels,
        hierarchical=args.hierarchical,
        plot=False,
    )
    mesh_lat_lon_list = mesh_structure["mesh_lat_lon_list"]  # list of (N_i, 2) degrees
    num_levels = len(mesh_lat_lon_list)
    print(f"  Mesh levels  : {num_levels}")
    for i, ll in enumerate(mesh_lat_lon_list):
        print(f"    level {i}: {ll.shape[0]:,} nodes")

    # ------------------------------------------------------------------
    # 2. Load static datasets
    # ------------------------------------------------------------------
    land_tree, land_values = load_land_mask_unstructured(args.land_mask)
    topo_interp            = load_topography_regular(args.topography)

    # ------------------------------------------------------------------
    # 3. Interpolate onto each mesh level
    # ------------------------------------------------------------------
    save_dict = {}

    for level_idx, mesh_lat_lon_deg in enumerate(mesh_lat_lon_list):
        n = mesh_lat_lon_deg.shape[0]
        print(f"\n[LEVEL {level_idx}]  {n:,} nodes ...")

        lat = mesh_lat_lon_deg[:, 0]   # degrees
        lon = mesh_lat_lon_deg[:, 1]   # degrees

        # Coordinates (for verification / debugging)
        save_dict[f"mesh_lat_level_{level_idx}"] = lat.astype(np.float32)
        save_dict[f"mesh_lon_level_{level_idx}"] = lon.astype(np.float32)

        # ---- Land mask  (nearest-neighbour on unit sphere) ----
        land = query_land_mask(land_tree, land_values, lat, lon, k=args.land_mask_k)
        land = np.clip(land, 0.0, 1.0)
        save_dict[f"land_mask_level_{level_idx}"] = land
        print(f"  land_mask  : min={land.min():.3f}  max={land.max():.3f}  "
              f"mean={land.mean():.3f}  land%={land.mean()*100:.1f}")

        # ---- Elevation raw (metres) ----
        elev = query_topography(topo_interp, lat, lon)
        save_dict[f"elevation_level_{level_idx}"] = elev
        print(f"  elevation  : min={elev.min():.1f}  max={elev.max():.1f}  "
              f"mean={elev.mean():.1f} m")

        # ---- Elevation normalised ----
        elev_norm = normalize_elevation(elev, scale=args.elev_scale)
        save_dict[f"elevation_norm_level_{level_idx}"] = elev_norm
        print(f"  elev_norm  : min={elev_norm.min():.4f}  max={elev_norm.max():.4f}  "
              f"mean={elev_norm.mean():.4f}")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print(f"\n[SAVE] Writing to {args.output} ...")
    np.savez(args.output, **save_dict)

    loaded = np.load(args.output)
    print(f"\n[VERIFY] Contents of {args.output}:")
    for key in sorted(loaded.files):
        arr = loaded[key]
        print(f"  {key:44s}  shape={str(arr.shape):12s}  dtype={arr.dtype}")

    print("\nDone.")


if __name__ == "__main__":
    main()
