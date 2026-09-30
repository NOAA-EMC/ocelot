#!/usr/bin/env python
"""Per-channel health check on the verifying observations and the forecasts.

Written to test whether the problems in NOAA-EMC/ocelot#119 reach the manuscript results:
fill values that were not removed and were reported as valid by the channel mask, and an encoder
that cannot tell a missing satellite channel from an average one.

Two signatures are looked for, per channel:

  * a fill value that survived QC: a single exact value repeated far more often than a continuous
    geophysical field ever would. The most frequent value and its share are reported, together with
    how close it sits to the channel's normalization mean, because a missing value imputed as
    normalized zero comes back as exactly that mean.

  * a channel that carries no information: the forecast has far less spread than the observations
    (spread ratio well below one) while the correlation is near zero. That is what a channel looks
    like when the encoder saw mean-filled values, or when the target was mostly fill.

Nothing here changes any result; it only reads the prediction CSVs.

Usage (from gnn_model/):
    python evaluation/revision/diagnose_channels.py --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
        --instruments ssmis,amsua,atms,avhrr --n_files 12
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml


def _stat(d, ch):
    """(n, correlation, forecast spread / observed spread) for one channel."""
    m = d[f"mask_{ch}"]
    m = m.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]) if m.dtype != bool else m
    o, p = d[f"true_{ch}"].to_numpy(float), d[f"pred_{ch}"].to_numpy(float)
    ok = m.to_numpy(bool) & np.isfinite(o) & np.isfinite(p)
    if ok.sum() < 200:
        return int(ok.sum()), np.nan, np.nan
    oo, pp = o[ok], p[ok]
    sd = oo.std()
    corr = float(np.corrcoef(pp, oo)[0, 1]) if sd > 0 and pp.std() > 0 else np.nan
    return int(ok.sum()), corr, (float(pp.std() / sd) if sd > 0 else np.nan)


def by_year(args, stats) -> int:
    """Per-period correlation and forecast spread, which dates a change in an observing system.

    The model is identical across all of these years, so a correlation that drops in one year and
    stays low is a change in the observations rather than in the forecast.
    """
    for inst in args.instruments.split(","):
        pattern = (os.path.join(args.pred_dir, "**", f"pred_{inst}_target_init_*.csv") if args.recursive
                   else os.path.join(args.pred_dir, f"pred_{inst}_target_init_*.csv"))
        files = [f for f in sorted(glob.glob(pattern, recursive=args.recursive)) if "_vs_" not in f]
        if not files:
            print(f"\n### {inst}: no prediction files")
            continue
        by_y = {}
        for f in files:
            m = re.search(r"_init_(\d{4,6})", os.path.basename(f))
            if m:
                by_y.setdefault(m.group(1)[: 6 if args.period == "month" else 4], []).append(f)
        years = sorted(by_y)
        frames = {}
        for y in years:
            sel = by_y[y][:: max(1, len(by_y[y]) // args.n_files)][: args.n_files]
            frames[y] = pd.concat([pd.read_csv(f, low_memory=False) for f in sel], ignore_index=True)
        first = frames[years[0]]
        chans = args.channels.split(",") if args.channels else [
            c[5:] for c in first.columns if c.startswith("true_") and f"pred_{c[5:]}" in first.columns]
        span = f"{min(len(v) for v in by_y.values())}-{max(len(v) for v in by_y.values())}"
        print(f"\n### {inst}: correlation by {args.period}, forecast/observed spread in brackets "
              f"({span} files sampled per {args.period})")
        print(f"{'channel':22s} " + " ".join(f"{y:>13s}" for y in years))
        print(f"{'files':22s} " + " ".join(f"{len(by_y[y]):>13d}" for y in years))
        for ch in chans:
            cells = []
            for y in years:
                if f"true_{ch}" not in frames[y].columns:
                    cells.append(f"{'-':>13s}")
                    continue
                _, r, sr = _stat(frames[y], ch)
                cells.append(f"{'n/a':>13s}" if not np.isfinite(r) else f"{r:7.2f} [{sr:4.2f}]")
            print(f"{ch:22s} " + " ".join(cells))
    print("\nThe model is the same in every period shown, so a correlation that falls and stays low")
    print("indicates a change in the observing system rather than in the forecast. A fall in the")
    print("file count as well as the correlation means observations were withdrawn, not degraded.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--instruments", default="ssmis,amsua,atms,avhrr,seviri_asr,ascat")
    ap.add_argument("--n_files", type=int, default=12)
    ap.add_argument("--recursive", action="store_true",
                    help="Search sub-directories too (the climatology dump stores one folder per date)")
    ap.add_argument("--cfg_path", default="configs/observation_config.yaml")
    ap.add_argument("--spike_pct", type=float, default=0.5, help="flag a repeated value above this share, in percent")
    ap.add_argument("--by_year", action="store_true",
                    help="report correlation and forecast spread per calendar year, to date when a channel changed")
    ap.add_argument("--channels", default=None, help="comma-separated subset of channels (by_year mode)")
    ap.add_argument("--period", choices=["year", "month"], default="year",
                    help="granularity of --by_year: calendar year, or month to date a change inside a year")
    args = ap.parse_args()

    stats = (yaml.safe_load(open(args.cfg_path, encoding="utf-8")) or {}).get("feature_stats", {})

    if args.by_year:
        return by_year(args, stats)

    for inst in args.instruments.split(","):
        pattern = (os.path.join(args.pred_dir, "**", f"pred_{inst}_target_init_*.csv") if args.recursive
                   else os.path.join(args.pred_dir, f"pred_{inst}_target_init_*.csv"))
        files = sorted(glob.glob(pattern, recursive=args.recursive))
        files = [f for f in files if "_vs_" not in f]
        if not files:
            print(f"\n### {inst}: no prediction files\n")
            continue
        files = files[:: max(1, len(files) // args.n_files)][: args.n_files]
        d = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
        chans = [c[5:] for c in d.columns if c.startswith("true_") and f"pred_{c[5:]}" in d.columns]
        print(f"\n### {inst}   {len(files)} files, {len(d):,} rows")
        print(f"{'channel':22s} {'n_valid':>10s} {'obs_sd':>7s} {'fc_sd':>7s} {'sd_ratio':>8s} {'corr':>6s} "
              f"{'top value':>10s} {'share%':>7s} {'=mean?':>7s}  flags")
        for ch in chans:
            m = d[f"mask_{ch}"]
            m = m.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]) if m.dtype != bool else m
            o = d[f"true_{ch}"].to_numpy(float)
            p = d[f"pred_{ch}"].to_numpy(float)
            ok = m.to_numpy(bool) & np.isfinite(o) & np.isfinite(p)
            n = int(ok.sum())
            if n < 100:
                print(f"{ch:22s} {n:>10,}   (too few valid observations)")
                continue
            oo, pp = o[ok], p[ok]
            vals, cnts = np.unique(np.round(oo, 4), return_counts=True)
            top_v, top_c = vals[cnts.argmax()], cnts.max()
            share = 100.0 * top_c / n
            mu = stats.get(inst, {}).get(ch, [np.nan])[0]
            near_mean = "yes" if np.isfinite(mu) and abs(top_v - mu) < 0.5 else ""
            osd, fsd = float(oo.std()), float(pp.std())
            ratio = fsd / osd if osd > 0 else np.nan
            corr = float(np.corrcoef(pp, oo)[0, 1]) if osd > 0 and fsd > 0 else np.nan
            flags = []
            if share > args.spike_pct:
                flags.append(f"repeated value {share:.1f}%")
            if np.isfinite(ratio) and ratio < 0.3:
                flags.append("forecast nearly flat")
            if np.isfinite(corr) and corr < 0.3:
                flags.append("low correlation")
            print(f"{ch:22s} {n:>10,} {osd:7.2f} {fsd:7.2f} {ratio:8.2f} {corr:6.2f} "
                  f"{top_v:10.2f} {share:7.3f} {near_mean:>7s}  {'; '.join(flags)}")
    print("\nA share far above a fraction of a percent at one exact value is a fill value that passed QC.")
    print("'=mean? yes' means that value equals the channel's normalization mean, i.e. a missing value")
    print("imputed as normalized zero that was then reported as a valid observation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
