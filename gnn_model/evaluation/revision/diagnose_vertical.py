#!/usr/bin/env python
"""Is part of the radiosonde/aircraft error caused by how vertical position reaches the decoder?

The encoder receives each report's continuous log-pressure height, but the decoder is conditioned
only on the nearest of 16 standard pressure levels (plus valid time). Reports at 230 and 270 hPa
are therefore decoded as the same "250 hPa" target, while the observations differ by the lapse
rate across that interval. If that matters, the error contains a component set by each report's
position within its level bin: spatially random (low C_LS), not predictable from the latent state
(low ACC), and not representativeness noise in the usual sense.

The test, per target variable and level bin, uses the within-bin offset x = ln(p / p_standard).
Observations, forecasts and errors are demeaned within groups of (initialization, lead, level,
5-degree box), so large-scale weather and geography drop out, and the remaining variation is
regressed on x:

  obs slope   how the observations change with position inside the bin (about the lapse rate)
  fc slope    how the forecast changes with it (near zero if the decoder cannot see x)
  err share   share of the total squared error explained by x through the error slope
              (what a decoder that saw the continuous height could at most recover, linearly)

Usage (from gnn_model/):
    python evaluation/revision/diagnose_vertical.py \
        --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space --n_files 40
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

LEVELS = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10], float)
VARS = {"aircraft": ["airTemperature", "windU", "windV"],
        "radiosonde": ["airTemperature", "dewPointTemperature", "wind_u", "wind_v"]}
BOX_DEG = 5.0


def _mask(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).to_numpy()


def load(pred_dir: str, inst: str, n_files: int, recursive: bool) -> pd.DataFrame:
    pat = os.path.join(pred_dir, "**" if recursive else "", f"pred_{inst}_target_init_*.csv")
    files = [f for f in sorted(glob.glob(pat, recursive=recursive)) if "_vs_" not in f]
    files = files[:: max(1, len(files) // n_files)][:n_files]
    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        m = re.search(r"_init_(\d+)", os.path.basename(f))
        d["init"] = m.group(1) if m else os.path.basename(f)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def analyse(d: pd.DataFrame, var: str) -> pd.DataFrame:
    need = {f"pred_{var}", f"true_{var}", f"mask_{var}", "pressure_hPa", "pressure_level_idx",
            "lead_hours_nominal", "lat", "lon"}
    if not need.issubset(d.columns):
        return pd.DataFrame()
    p = pd.to_numeric(d["pressure_hPa"], errors="coerce").to_numpy(float)
    li = pd.to_numeric(d["pressure_level_idx"], errors="coerce").to_numpy(float)
    f = pd.to_numeric(d[f"pred_{var}"], errors="coerce").to_numpy(float)
    o = pd.to_numeric(d[f"true_{var}"], errors="coerce").to_numpy(float)
    ok = (_mask(d[f"mask_{var}"]) & np.isfinite(p) & (p > 0) & np.isfinite(li) & (li >= 0) & (li < len(LEVELS))
          & np.isfinite(f) & np.isfinite(o))
    if ok.sum() < 1000:
        return pd.DataFrame()
    lev = LEVELS[li[ok].astype(int)]
    g = pd.DataFrame({
        "level": lev,
        "x": np.log(p[ok] / lev),
        "f": f[ok], "o": o[ok],
        "grp": (d["init"].to_numpy()[ok].astype(str) + "|"
                + pd.to_numeric(d["lead_hours_nominal"], errors="coerce").to_numpy()[ok].astype(str) + "|"
                + lev.astype(int).astype(str) + "|"
                + np.floor((pd.to_numeric(d["lat"], errors="coerce").to_numpy()[ok] + 90) / BOX_DEG).astype(int).astype(str) + "|"
                + np.floor(np.mod(pd.to_numeric(d["lon"], errors="coerce").to_numpy()[ok] + 180, 360) / BOX_DEG).astype(int).astype(str)),
    })
    g["e"] = g["f"] - g["o"]
    for c in ("x", "f", "o", "e"):
        g[f"d{c}"] = g[c] - g.groupby("grp")[c].transform("mean")

    rows = []
    for lvl, h in list(g.groupby("level")) + [("all", g)]:
        sxx = float((h["dx"] ** 2).sum())
        if len(h) < 500 or sxx <= 0:
            continue
        b_o = float((h["dx"] * h["do"]).sum() / sxx)
        b_f = float((h["dx"] * h["df"]).sum() / sxx)
        b_e = float((h["dx"] * h["de"]).sum() / sxx)
        sse = float((h["e"] ** 2).sum())
        rows.append(dict(
            level=lvl if lvl == "all" else int(lvl), n=len(h),
            off_standard_pct=100.0 * float((np.abs(h["x"]) > 0.01).mean()),
            median_abs_offset_hPa=float(np.median(np.abs(np.exp(h["x"]) - 1.0) * (h["level"].astype(float)))),
            obs_slope=b_o, fc_slope=b_f, err_slope=b_e,
            err_share_pct=100.0 * b_e ** 2 * sxx / sse if sse > 0 else np.nan,
            rmse=float(np.sqrt(sse / len(h))),
        ))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--n_files", type=int, default=40)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--instruments", default="aircraft,radiosonde")
    ap.add_argument("--out", default=None, help="optional CSV of every row printed")
    args = ap.parse_args()

    out = []
    for inst in args.instruments.split(","):
        d = load(args.pred_dir, inst, args.n_files, args.recursive)
        if d.empty:
            print(f"\n### {inst}: no prediction files")
            continue
        if "pressure_hPa" not in d.columns:
            print(f"\n### {inst}: no pressure_hPa column in the CSVs")
            continue
        print(f"\n### {inst}: {d['init'].nunique()} initializations, {len(d):,} rows")
        for var in VARS.get(inst, []):
            r = analyse(d, var)
            if r.empty:
                print(f"  {var}: not enough valid rows")
                continue
            r.insert(0, "variable", var)
            r.insert(0, "instrument", inst)
            out.append(r)
            print(f"\n  {var}  (slopes in units per unit ln p; +x = higher pressure, lower in the atmosphere)")
            print(f"  {'level':>6s} {'n':>10s} {'off-std%':>9s} {'|dp| hPa':>9s} {'obs slope':>10s} "
                  f"{'fc slope':>9s} {'err share%':>11s} {'rmse':>7s}")
            for _, x in r.iterrows():
                print(f"  {str(x['level']):>6s} {int(x['n']):>10,} {x['off_standard_pct']:>9.1f} "
                      f"{x['median_abs_offset_hPa']:>9.1f} {x['obs_slope']:>10.2f} {x['fc_slope']:>9.2f} "
                      f"{x['err_share_pct']:>11.2f} {x['rmse']:>7.2f}")
    if args.out and out:
        pd.concat(out, ignore_index=True).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    print("\nfc slope near 0 while obs slope is large: the decoder does not see position within the level bin.")
    print("err share: the part of the squared error that a decoder given the continuous height could")
    print("remove linearly. A few percent means the error is not an ingestion artifact; tens of")
    print("percent means a large part of it is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
