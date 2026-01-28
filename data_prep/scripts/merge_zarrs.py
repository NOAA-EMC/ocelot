import os
import argparse
from glob import glob
import zarr
import numpy as np


def find_zarr_files(input_dir, data_type) -> list[str]:
    # The zarr files are named like <data_type>_YYYY.zarr. We need to return
    # a the list of zarr files sorted by year so that we can merge them in that
    # order.
    pattern = os.path.join(input_dir, f"{data_type}_*.zarr")
    zarr_files = glob(pattern)
    zarr_files.sort()  # Sort by filename which includes year
    return zarr_files

def merge_zarr_datasets(zarr_files: list[str], output_dir: str, data_type: str):
    if not zarr_files:
        raise ValueError("No zarr files provided")
    
    ChunkSize = 1024  # Define a reasonable chunk size for appending

    # Open the first zarr file to get the structure
    first_zarr = zarr.open(zarr_files[0], mode='r')

    # Create the output zarr store (truncate if exists)
    output_path = os.path.join(output_dir, f"{data_type}.zarr")
    output_zarr = zarr.open(output_path, mode='w')

    # Copy the structure from the first zarr file, preserving compressor and filters
    keys = list(first_zarr.array_keys())
    for key in keys:
        src = first_zarr[key]
        shape = list(src.shape)
        shape[0] = 0  # Set the first dimension to 0 for appending
        chunks = src.chunks
        dtype = src.dtype
        compressor = getattr(src, 'compressor', None)
        filters = getattr(src, 'filters', None)
        fill_value = getattr(src, 'fill_value', None)
        output_zarr.create_dataset(
            key,
            shape=tuple(shape),
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            filters=filters,
            fill_value=fill_value,
        )

    # Append data from each zarr file using chunked reads to limit memory use
    for zarr_file in zarr_files:
        print(f"Processing {zarr_file}")
        current_zarr = zarr.open(zarr_file, mode='r')

        for key in keys:
            print(f"  Adding {key}")
            src = current_zarr[key]
            total = src.shape[0]

            # Determine chunk size for axis 0. Fall back to a reasonable default.
            if src.chunks and len(src.chunks) > 0 and src.chunks[0] is not None:
                chunk0 = src.chunks[0]
            else:
                # Avoid tiny or enormous defaults; choose up to ChunkSize or total
                chunk0 = min(max(1, total // 8), ChunkSize)

            # Read and append in slices to avoid loading entire arrays into memory
            for start in range(0, total, chunk0):
                stop = min(start + chunk0, total)
                data_to_append = src[start:stop]
                output_zarr[key].append(data_to_append, axis=0)

    print(f"Merged {len(zarr_files)} Zarr datasets into {output_path}")
    return output_path



if __name__ == "__main__":
    # Parse input dir output dir and data type (ex: atms, amsua, aircraft etc...)
    parser = argparse.ArgumentParser(description="Merge multiple Zarr datasets into a single Zarr dataset.")
    parser.add_argument("input_dir", type=str, help="Directory containing Zarr datasets to merge.")
    parser.add_argument("output_dir", type=str, help="Directory to save the merged Zarr dataset.")
    parser.add_argument("data_type", type=str, help="Type of data (e.g., atms, amsua, aircraft).")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    data_type = args.data_type

    zarr_files = find_zarr_files(input_dir, data_type)
    merge_zarr_datasets(zarr_files, output_dir, data_type)
