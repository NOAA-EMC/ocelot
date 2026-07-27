"""
mesh_resolution.py — print mesh resolution table for a given domain.

Usage examples:

  # From projected x/y extents (metres):
  python mesh_resolution.py --xmin -3000 --xmax 3000 --ymin -1500 --ymax 1800

  # From lat/lon bounding box (uses RRFS 15 km LCC projection):
  python mesh_resolution.py --lat_min 24.5 --lat_max 49.5 --lon_min -125 --lon_max -66.5

  # Try different branching factors:
  python mesh_resolution.py --lat_min 24.5 --lat_max 49.5 --lon_min -125 --lon_max -66.5 --nx 2 4

All x/y extents are in kilometres unless --metres is passed.
"""

import argparse
import math
import sys


def _project_latlon(lat_min, lat_max, lon_min, lon_max):
    try:
        import numpy as np
        from pyproj import CRS, Transformer
    except ImportError:
        sys.exit("pyproj is required for lat/lon input:  pip install pyproj")

    lcc = CRS.from_proj4(
        "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5"
        " +lon_0=-97.5 +ellps=WGS84 +x_0=0 +y_0=0"
    )
    tr = Transformer.from_crs("EPSG:4326", lcc, always_xy=True)

    lons = np.array([lon_min, lon_min, lon_max, lon_max])
    lats = np.array([lat_min, lat_max, lat_min, lat_max])
    xs, ys = tr.transform(lons, lats)
    return xs.min() / 1e3, xs.max() / 1e3, ys.min() / 1e3, ys.max() / 1e3


def resolution_table(W_km, H_km, nx_values, max_levels=8):
    """
    Return list of rows: (NX, levels, n_finest, dx_km, dy_km).
    n_finest = NX^levels nodes per side at the finest mesh level.
    dx = W / n_finest,  dy = H / n_finest.
    """
    rows = []
    for NX in nx_values:
        for levels in range(1, max_levels + 1):
            n = NX ** levels
            dx = W_km / n
            dy = H_km / n
            rows.append((NX, levels, n, dx, dy))
    return rows


def print_table(rows, W_km, H_km):
    header = f"{'NX':>4}  {'levels':>6}  {'n_finest':>9}  {'dx (km)':>9}  {'dy (km)':>9}"
    print(header)
    print("-" * len(header))
    prev_nx = None
    for NX, levels, n, dx, dy in rows:
        if prev_nx is not None and NX != prev_nx:
            print()
        print(f"{NX:>4}  {levels:>6}  {n:>9}  {dx:>9.1f}  {dy:>9.1f}")
        prev_nx = NX


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    domain = p.add_argument_group("domain (choose one input mode)")
    domain.add_argument("--xmin", type=float, help="x min")
    domain.add_argument("--xmax", type=float, help="x max")
    domain.add_argument("--ymin", type=float, help="y min")
    domain.add_argument("--ymax", type=float, help="y max")
    domain.add_argument("--lat_min", type=float, help="south latitude (deg)")
    domain.add_argument("--lat_max", type=float, help="north latitude (deg)")
    domain.add_argument("--lon_min", type=float, help="west longitude (deg)")
    domain.add_argument("--lon_max", type=float, help="east longitude (deg)")
    domain.add_argument(
        "--metres",
        action="store_true",
        help="x/y inputs are in metres (default: km)")

    p.add_argument(
        "--nx",
        type=int,
        nargs="+",
        default=[
            2,
            3,
            4],
        metavar="NX",
        help="branching factor(s) to tabulate (default: 2 3 4)")
    p.add_argument("--max_levels", type=int, default=8,
                   help="max mesh levels to show (default: 8)")

    args = p.parse_args()

    # Resolve domain extent in km
    if args.lat_min is not None:
        for attr in ("lat_min", "lat_max", "lon_min", "lon_max"):
            if getattr(args, attr) is None:
                p.error(f"--{attr} is required when using lat/lon input")
        xmin, xmax, ymin, ymax = _project_latlon(
            args.lat_min, args.lat_max, args.lon_min, args.lon_max
        )
        print(f"Projected domain (RRFS 15 km LCC):")
        print(f"  x: [{xmin:.0f}, {xmax:.0f}] km   W = {xmax - xmin:.0f} km")
        print(f"  y: [{ymin:.0f}, {ymax:.0f}] km   H = {ymax - ymin:.0f} km")
        print()
    elif args.xmin is not None:
        for attr in ("xmin", "xmax", "ymin", "ymax"):
            if getattr(args, attr) is None:
                p.error(f"--{attr} is required when using x/y input")
        scale = 1e-3 if args.metres else 1.0
        xmin, xmax = args.xmin * scale, args.xmax * scale
        ymin, ymax = args.ymin * scale, args.ymax * scale
    else:
        p.error("Provide either lat/lon bounds or x/y bounds.")

    W_km = xmax - xmin
    H_km = ymax - ymin

    print(f"Domain: W = {W_km:.0f} km,  H = {H_km:.0f} km")
    print(f"Formula: dx = W / NX^levels,  dy = H / NX^levels\n")

    rows = resolution_table(W_km, H_km, args.nx, args.max_levels)
    print_table(rows, W_km, H_km)


if __name__ == "__main__":
    main()
