import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
import multiprocessing as mp
from glob import glob

# Global variables for multiprocessing
ds = None
halfday_bins = None
out_file = None

# -------------------------------
# Function to process a single 12-hour bin
# -------------------------------


def process_bin(bin_time):
    global ds, halfday_bins, out_file
    try:
        print(f"🔄 Processing bin starting at {bin_time}")
        # Select data for this 12-hour bin
        mask = halfday_bins == bin_time
        ds_part = ds.isel(time=np.where(mask)[0])

        if ds_part.sizes["time"] == 0:
            return False  # skip empty partitions

        # Identify non-time coordinates
        non_time_coords = [c for c in ds_part.coords if c != "time"]
        if non_time_coords:
            # Drop variables depending on non-time coords
            vars_to_drop = [
                v for v, da in ds_part.data_vars.items()
                if any(d in non_time_coords for d in da.dims)
            ]
            ds_part = ds_part.drop_vars(vars_to_drop + non_time_coords)

        # Convert to Dask DataFrame (lazy)
        ddf_part = ds_part.to_dask_dataframe()

        # Add partition column for Hive-style
        bin_label = pd.Timestamp(bin_time).strftime("%Y-%m-%d_%H")
        ddf_part["day_half"] = bin_label

        # Output path
        out_path = os.path.join(out_file, f"day_half={bin_label}")
        ddf_part.to_parquet(out_path, engine="pyarrow", write_index=False)

        print(f"✅ Wrote partition {bin_label} -> {out_path}")
        return True

    except Exception as e:
        print(f"❌ Error processing bin {bin_time}: {e}")
        return False

# -------------------------------
# Function to process a single zarr file
# -------------------------------


def process_zarr_file(zarr_file, output_parquet_dir, num_cores=8):
    global ds, halfday_bins, out_file

    FILL_VALUE = 3.4028235e+38

    # Extract base name from zarr file (e.g., "atms_20240401_20240407")
    base_name = os.path.basename(zarr_file).replace('.zarr', '')

    # Use the shared output directory (all files write to same parquet)
    out_file = output_parquet_dir

    print(f"\n{'='*60}")
    print(f"🔍 Processing {base_name}")
    print(f"{'='*60}")

    # Open dataset lazily
    ds = xr.open_zarr(
        zarr_file,
        chunks={
            "time": 5_000_000},
        consolidated=False)

    print("📥 Dataset opened.")
    ds = ds.where(ds != FILL_VALUE, np.nan)

    # Convert numeric time to datetime if needed
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        print("⏳ Converting numeric timestamps to datetime...")
        ds["time"] = pd.to_datetime(
            ds["time"].values, unit="s", errors="coerce")

    print(f"🗄️  Dataset opened with {ds.sizes['time']} time entries.")

    # Floor to 12-hour bins
    times = pd.to_datetime(ds["time"].values)
    halfday_bins = times.floor("12h")
    unique_bins = pd.unique(halfday_bins)
    print(f"📦 Found {len(unique_bins)} unique 12-hour bins")

    # Parallel execution using multiprocessing
    n_cores = min(mp.cpu_count(), num_cores)
    print(f"🧠 Using {n_cores} parallel processes")

    try:
        with mp.Pool(processes=n_cores) as pool:
            results = pool.map(process_bin, unique_bins)
    finally:
        ds.close()

    success = sum(results)
    print(
        f"🏁 Finished {success}/{len(unique_bins)} partitions successfully for {base_name}")

    return success, len(unique_bins)


# -------------------------------
# Main script
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python script.py <data_path> <obs_type_pattern> [num_cores]")
        print("Example: python script.py /path/to/data 'atms_*.zarr' 8")
        print("         python script.py /path/to/data 'radiosonde_*.zarr' 8")
        print("         python script.py /path/to/data 'surface_obs_*.zarr'")
        sys.exit(1)

    data_path = sys.argv[1]
    obs_type_pattern = sys.argv[2]
    num_cores = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    # Extract output name from pattern (e.g., 'atms_*.zarr' -> 'atms')
    # Remove wildcards and .zarr extension
    output_name = obs_type_pattern.replace(
        '*.zarr',
        '').replace(
        '_*.zarr',
        '').rstrip('_')
    if not output_name:
        # Fallback: use first file's prefix
        search_pattern = os.path.join(data_path, obs_type_pattern)
        sample_files = glob(search_pattern)
        if sample_files:
            output_name = os.path.basename(sample_files[0]).split('_')[0]
        else:
            output_name = 'output'

    # Create single output directory for all files of this observation type
    base_parquet_dir = '/scratch3/NCEPDEV/da/Xin.C.Jin/my_data/ocelot/data_v4/global'
    output_parquet_dir = os.path.join(
        base_parquet_dir, f"{output_name}.parquet")
    os.makedirs(output_parquet_dir, exist_ok=True)

    # Find all matching zarr files
    search_pattern = os.path.join(data_path, obs_type_pattern)
    zarr_files = sorted(glob(search_pattern))

    if not zarr_files:
        print(f"❌ No files found matching pattern: {search_pattern}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"📂 Found {len(zarr_files)} zarr files to process:")
    print(f"📁 Output directory: {output_parquet_dir}")
    print(f"{'='*60}")
    for zf in zarr_files:
        print(f"  - {os.path.basename(zf)}")
    print(f"{'='*60}\n")

    # Process each zarr file (all write to same output directory)
    total_success = 0
    total_bins = 0

    for i, zarr_file in enumerate(zarr_files, 1):
        print(
            f"\n🔄 Processing file {i}/{len(zarr_files)}: {os.path.basename(zarr_file)}")
        try:
            success, bins = process_zarr_file(
                zarr_file, output_parquet_dir, num_cores)
            total_success += success
            total_bins += bins
        except Exception as e:
            print(f"❌ Error processing {zarr_file}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"🎉 ALL DONE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_parquet_dir}")
    print(f"Total files processed: {len(zarr_files)}")
    print(f"Total partitions: {total_success}/{total_bins}")
    print(f"{'='*60}\n")
