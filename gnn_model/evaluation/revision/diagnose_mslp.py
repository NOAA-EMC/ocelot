#!/usr/bin/env python
"""Why is surface MSLP verification so poor? Separate observation error from forecast error.

Both OCELOT (3.2 hPa) and GFS (2.3 hPa) score poorly at +3 h against the same reports. An
operational analysis should be well under 1 hPa against good sea-level pressure observations, so
the verifying observations are the prime suspect. Unlike every other surface variable, MSLP passes
only a range check of 800-1100 hPa in the observation config, with no quality-mark filter, and that
range is wide enough to admit station pressure (a station at 1500 m reports about 845 hPa).

The report covers:
  1. distribution of the observed values (a low tail indicates station pressure, not MSLP);
  2. how concentrated the squared error is (outliers versus a broadly elevated error);
  3. GFS minus observation, which is the forecast-independent-ish view of observation quality;
  4. what each method would score if obviously bad reports were screened out;
  5. residual against local solar time, which would reveal a timing or tidal error.

Screening on GFS minus observation favours GFS and must NOT be used for published scores; it is
here only to show whether observation error dominates. Published numbers need a screen that is
independent of the compared forecasts (the quality mark, or station elevation).

Usage (from gnn_model/):
    python evaluation/revision/diagnose_mslp.py \
        --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space --n_files 20
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

V = "pressureMeanSeaLevel_prepbufr"


def rmse(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float))
    d = d[np.isfinite(d)]
    return float(np.sqrt((d ** 2).mean())) if d.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--n_files", type=int, default=20)
    ap.add_argument("--lead", type=float, default=3.0)
    ap.add_argument("--screen_hpa", type=float, default=5.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_dir, f"pred_surface_obs_target_init_*_vs_gfs.csv")))
    if not files:
        raise SystemExit(f"No *_vs_gfs.csv for surface_obs under {args.pred_dir}")
    files = files[:: max(1, len(files) // args.n_files)][: args.n_files]
    cols = ["lat", "lon", "lead_hours_nominal", "obs_time_unix", f"pred_{V}", f"true_{V}", f"persist_{V}",
            f"mask_{V}", "gfs_mslp_hPa"]
    frames = []
    for f in files:
        have = pd.read_csv(f, nrows=0).columns
        frames.append(pd.read_csv(f, usecols=[c for c in cols if c in have]))
    d = pd.concat(frames, ignore_index=True)
    m = d[f"mask_{V}"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]) if f"mask_{V}" in d else True
    d = d[m & (d.lead_hours_nominal == args.lead)].copy()
    o, p, g = d[f"true_{V}"].to_numpy(float), d[f"pred_{V}"].to_numpy(float), d["gfs_mslp_hPa"].to_numpy(float)
    ok = np.isfinite(o) & np.isfinite(p) & np.isfinite(g)
    d, o, p, g = d[ok], o[ok], p[ok], g[ok]
    print(f"{len(files)} files, {len(d):,} matched observations at +{args.lead:.0f} h\n")

    print("1. OBSERVED MSLP DISTRIBUTION (hPa)   [sea-level pressure should sit near 1013 +/- ~10]")
    q = np.percentile(o, [0.01, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    print("   pct  " + "  ".join(f"{x:>6}" for x in ["0.01", "0.1", "1", "5", "25", "50", "75", "95", "99", "99.9"]))
    print("   val  " + "  ".join(f"{v:6.1f}" for v in q))
    for lim in (950, 970, 990):
        n = int((o < lim).sum())
        print(f"   below {lim} hPa: {n:,} ({100.0 * n / len(o):.3f}%)"
              + (f"   <-- plausible station pressure, not MSLP" if lim == 950 and n else ""))
    print()

    print("2. HOW CONCENTRATED IS THE ERROR?")
    for name, f_ in (("OCELOT", p), ("GFS", g)):
        e = np.abs(f_ - o)
        s = np.sort(e ** 2)[::-1]
        frac1 = 100.0 * s[: max(1, len(s) // 100)].sum() / s.sum()
        print(f"   {name:7s} RMSE {rmse(f_, o):6.3f}   median|err| {np.median(e):5.3f}   "
              f"p99|err| {np.percentile(e, 99):6.2f}   worst 1% of obs carry {frac1:4.1f}% of the squared error")
    print("   (RMSE >> median error and a large 'worst 1%' share means a contaminated subset, not a broadly poor forecast)\n")

    print("3. GFS MINUS OBSERVATION (hPa)   [a good MSLP report should agree with GFS within ~1 hPa]")
    dg = g - o
    print("   mean %+.3f  median %+.3f  robust sd %.3f  |dg|>2: %.2f%%  |dg|>5: %.2f%%  |dg|>20: %.3f%%"
          % (dg.mean(), np.median(dg), 1.4826 * np.median(np.abs(dg - np.median(dg))),
             100.0 * (np.abs(dg) > 2).mean(), 100.0 * (np.abs(dg) > 5).mean(), 100.0 * (np.abs(dg) > 20).mean()))
    print()

    print(f"4. IF REPORTS WITH |GFS - obs| > {args.screen_hpa:g} hPa WERE SCREENED OUT")
    keep = np.abs(dg) <= args.screen_hpa
    print(f"   keeps {keep.sum():,} of {len(o):,} ({100.0 * keep.mean():.2f}%)")
    for name, f_ in (("OCELOT", p), ("GFS", g), ("Persistence", d[f"persist_{V}"].to_numpy(float) if f"persist_{V}" in d else None)):
        if f_ is None:
            continue
        print(f"   {name:12s} all {rmse(f_, o):6.3f}  ->  screened {rmse(f_[keep], o[keep]):6.3f} hPa")
    print("   (this screen favours GFS and is diagnostic only; it shows how much of the error is observation error)\n")

    if "obs_time_unix" in d:
        print("5. OCELOT MINUS OBSERVATION BY LOCAL SOLAR TIME   [a clean semidiurnal signal = tide/timing issue]")
        t = pd.to_datetime(d.obs_time_unix.to_numpy(np.int64), unit="s", utc=True)
        lst = np.mod(t.hour.to_numpy() + t.minute.to_numpy() / 60.0 + d.lat.to_numpy() * 0 + d.lon.to_numpy() / 15.0, 24)
        res = p - o
        good = keep  # exclude obviously bad reports so the signal is visible
        b = pd.DataFrame({"h": (lst[good] // 3).astype(int) * 3, "r": res[good], "gr": dg[good]}).groupby("h").mean()
        print("   local hour :  " + "  ".join(f"{int(h):>5}" for h in b.index))
        print("   OCELOT bias:  " + "  ".join(f"{v:5.2f}" for v in b.r))
        print("   GFS bias   :  " + "  ".join(f"{v:5.2f}" for v in b.gr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
