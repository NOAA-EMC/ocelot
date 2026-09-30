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
import sys

import numpy as np
import pandas as pd
import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--instruments", default="ssmis,amsua,atms,avhrr,seviri_asr,ascat")
    ap.add_argument("--n_files", type=int, default=12)
    ap.add_argument("--cfg_path", default="configs/observation_config.yaml")
    ap.add_argument("--spike_pct", type=float, default=0.5, help="flag a repeated value above this share, in percent")
    args = ap.parse_args()

    stats = (yaml.safe_load(open(args.cfg_path, encoding="utf-8")) or {}).get("feature_stats", {})

    for inst in args.instruments.split(","):
        files = sorted(glob.glob(os.path.join(args.pred_dir, f"pred_{inst}_target_init_*.csv")))
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
