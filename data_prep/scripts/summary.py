#!/usr/bin/env python3
"""Print summary statistics for variables in a Zarr file."""
import argparse
from typing import List

import numpy as np
import zarr
import bufr


def _format_table(rows: List[List[str]]) -> str:
    """Return a fixed width table string from rows."""
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    formatted_rows = []
    for row in rows:
        formatted_row = " ".join(val.ljust(col_widths[i]) for i, val in enumerate(row))
        formatted_rows.append(formatted_row)
    return "\n".join(formatted_rows)


def summarize_zarr(path: str, max_obs: int) -> str:
    root = zarr.open(path, mode="r")
    rows: List[List[str]] = [[
        "Variable",
        "Units",
        "Total",
        "Good",
        "Good %",
        "Bad",
        "Min",
        "Mean",
        "Max",
        "StDev",
    ]]

    for var_name in root.array_keys():
        if max_obs == 0:
            arr = root[var_name][:]
        else:
            arr = root[var_name][:max_obs]

        #print('NICKE ', var_name, root[var_name].attrs['units'])

        #print(var_name)
        #print(root[var_name].attrs.keys())
        units = str(root[var_name].attrs['units'])
        #units = str("1")
        #print(units)

        if arr.dtype.kind in {"U", "S", "O"}:
            missing_strings = {"", None}
            arr = (~np.isin(arr, list(missing_strings))).astype(int)

        total = arr.size
        missing = bufr.get_missing_value(arr.dtype)
        good_mask = arr != missing
        if arr.dtype.kind == "f":
            good_mask &= ~np.isnan(arr)
        good = arr[good_mask]
        good_count = int(good.size)
        bad_count = total - good_count 
        if good_count:
            mn = np.min(good)
            mean = np.mean(good)
            mx = np.max(good)
            stdev = np.std(good)
        else:
            mn = mean = mx = stdev = float("nan")
        pct = 100.0 * good_count / total if total else 0.0
        rows.append([
            var_name,
            units,
            f"{total}",
            f"{good_count}",
            f"{pct:.1f}%",
            f"{bad_count}",
            f"{mn:.3f}",
            f"{mean:.3f}",
            f"{mx:.3f}",
            f"{stdev:.3f}",
        ])

    table = _format_table(rows)
    print(table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize variables in a Zarr file")
    parser.add_argument("zarr_path", help="Path to Zarr dataset")
    parser.add_argument("--max_obs", default=0, help="Limit the number of data obs for large files.")
    args = parser.parse_args()
    summarize_zarr(args.zarr_path, int(args.max_obs))


if __name__ == "__main__":
    main()
