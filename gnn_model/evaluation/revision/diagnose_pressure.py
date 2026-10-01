#!/usr/bin/env python
"""How many radiosonde and aircraft reports carry no usable pressure?

Radiosonde and aircraft observations are placed in the vertical only by their pressure: it sets
the log-pressure height in the encoder input and the pressure-level embedding in both the encoder
and the decoder. The configured airPressure QC (range and quality flag) is keyed on airPressure,
which is neither a model feature nor a listed metadata key (the metadata key is the derived
log_pressure_height), so that QC is never applied. What happens instead:

  * NaN pressure: targets are dropped (log_pressure_height becomes NaN); inputs are kept, with the
    height imputed from the column mean and a level index of -1.
  * fill-value or out-of-range pressure: the height is computed from the value clipped to
    [1, 1100] hPa and the level index from the nearest standard level, so a fill value is placed
    at about 1000-1100 hPa. These rows are kept as inputs and as targets.
  * a bad quality flag: kept, because the flag test is not applied either.

This script measures how often each case occurs in the archive, separately for the training
years and for 2025, by sampling evenly spaced chunks of each array.

Usage (from gnn_model/):
    python evaluation/revision/diagnose_pressure.py --data_path $DATA_PATH
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FILL = 3.402823e38
# lower bound from the configured QC; the upper bound is 1100 hPa for both
LOWER_HPA = {"radiosonde": 1.0, "aircraft": 100.0}
GOOD_FLAGS = (0, 1, 2)


def classify(pressure: np.ndarray, flags: np.ndarray | None, lower: float) -> dict[str, np.ndarray]:
    """Boolean masks for each failure mode; `intended_qc` is what the configured QC meant to remove."""
    p = np.asarray(pressure, dtype="float64")
    nan = ~np.isfinite(p)
    fill = np.isfinite(p) & (p >= FILL)
    finite = np.isfinite(p) & ~fill
    low = finite & (p < lower)
    high = finite & (p > 1100.0)
    out = {"nan": nan, "fill": fill, "below_range": low, "above_range": high}
    if flags is not None:
        f = np.asarray(flags, dtype="float64")
        known = np.isfinite(f) & (f >= 0) & (f < FILL)
        out["bad_flag"] = known & ~np.isin(f, GOOD_FLAGS)
        out["flag_missing"] = ~known
    else:
        out["bad_flag"] = np.zeros(p.shape, bool)
        out["flag_missing"] = np.ones(p.shape, bool)
    out["intended_qc"] = nan | fill | low | high | out["bad_flag"]
    out["silently_kept"] = fill | low | high | out["bad_flag"]  # reach the model as if valid
    return out


def _years(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[Y]").astype(int) + 1970
    secs = t.astype("float64")
    secs = np.where(np.isfinite(secs) & (secs < FILL), secs, np.nan)
    yrs = np.full(secs.shape, -1, dtype=int)
    ok = np.isfinite(secs)
    yrs[ok] = (secs[ok].astype("int64").astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970)
    return yrs


def sample(arr, n_blocks: int) -> tuple[np.ndarray, list[slice]]:
    n = arr.shape[0]
    chunk = int(arr.chunks[0]) if getattr(arr, "chunks", None) else min(n, 1_000_000)
    starts = np.unique(np.linspace(0, max(n - chunk, 0), num=min(n_blocks, max(1, n // max(chunk, 1))), dtype=np.int64))
    sl = [slice(int(s), int(min(s + chunk, n))) for s in starts]
    return np.concatenate([np.asarray(arr[s]) for s in sl]), sl


def report(name: str, masks: dict[str, np.ndarray], years: np.ndarray) -> None:
    groups = {"2015-2023 (training)": (years >= 2015) & (years <= 2023), "2025 (test)": years == 2025,
              "all sampled": np.ones(years.shape, bool)}
    print(f"\n### {name}")
    head = f"{'case':16s}" + "".join(f"{g:>22s}" for g in groups)
    print(head)
    print(f"{'rows sampled':16s}" + "".join(f"{int(m.sum()):>22,}" for m in groups.values()))
    for case in ("nan", "fill", "below_range", "above_range", "bad_flag", "flag_missing", "intended_qc", "silently_kept"):
        cells = []
        for m in groups.values():
            n = int(m.sum())
            cells.append(f"{(100.0 * masks[case][m].sum() / n if n else float('nan')):>21.4f}%")
        print(f"{case:16s}" + "".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--n_blocks", type=int, default=200, help="evenly spaced chunks sampled per array")
    args = ap.parse_args()

    import zarr
    from gnn_datamodule import _resolve_zarr_path

    for inst, zname in (("radiosonde", "raw_radiosonde"), ("aircraft", "aircraft")):
        try:
            zpath, _ = _resolve_zarr_path(args.data_path, zname, "2015-01-01")
        except FileNotFoundError as e:
            print(f"\n### {inst}: {e}")
            continue
        z = zarr.open(zpath, mode="r")
        if "airPressure" not in z:
            print(f"\n### {inst}: no airPressure array in {zpath}; every report lacks a vertical coordinate")
            continue
        p, slices = sample(z["airPressure"], args.n_blocks)
        flags = np.concatenate([np.asarray(z["airPressureQuality"][s]) for s in slices]) if "airPressureQuality" in z else None
        years = _years(np.concatenate([np.asarray(z["time"][s]) for s in slices])) if "time" in z else np.full(p.shape, -1)
        report(f"{inst}  ({zpath}, QC range [{LOWER_HPA[inst]:g}, 1100] hPa, flags {list(GOOD_FLAGS)})",
               classify(p, flags, LOWER_HPA[inst]), years)

    print("\nintended_qc   = share the configured airPressure QC was meant to remove (and does not)")
    print("silently_kept = share that reaches the model as if its pressure were valid")
    print("NaN-pressure targets are already dropped; NaN-pressure inputs keep an imputed height.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
