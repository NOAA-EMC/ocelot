#!/usr/bin/env python3
import argparse
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ---- feature -> (value_col, qm_col) mapping for NC000101 ----
FEATURES = {
    "airPressure": ("PRSSQ1.PRES", "PRSSQ1.QMPR"),
    "airTemperature": ("TMPSQ1.TMDB", "TMPSQ1.QMAT"),
    "dewPointTemperature": ("TMPSQ1.TMDP", "TMPSQ1.QMDD"),
    "relativeHumidity": ("TMPSQ1.TMPSQ2.REHU", None),  # no dedicated QM in this feed
    "wind_speed": ("WNDSQ1.WSPD", "WNDSQ1.QMWN"),
    "wind_direction": ("WNDSQ1.WDIR", "WNDSQ1.QMWN"),
}

KEEP_STRICT = {14}
KEEP_LENIENT = {-1, 0, 1, 2, 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", help=".../data/v1/conv/adpsfc/NC000101")
    ap.add_argument("--start", help="OBS_DATE >= (YYYY-MM-DD)")
    ap.add_argument("--end", help="OBS_DATE <= (YYYY-MM-DD)")
    ap.add_argument("--out", default="qc_summary.csv", help="CSV to write")
    args = ap.parse_args()

    ds_root = args.dataset_root
    dataset = ds.dataset(ds_root, format="parquet", partitioning="hive")

    # Optional date filter via partition
    filt = None
    part_schema = getattr(dataset.partitioning, "schema", None)
    if part_schema and "OBS_DATE" in [f.name for f in part_schema]:
        fld = ds.field("OBS_DATE")
        if args.start and args.end:
            filt = (fld >= args.start) & (fld <= args.end)
        elif args.start:
            filt = fld >= args.start
        elif args.end:
            filt = fld <= args.end

    # collect columns to scan
    value_cols = [c for c, _ in FEATURES.values()]
    qm_cols = sorted({q for _, q in FEATURES.values() if q})
    cols = sorted(set(value_cols + qm_cols))

    # scan in batches
    totals = defaultdict(int)  # column -> total non-null seen
    qc_counts = {q: defaultdict(int) for q in qm_cols}
    feat_stats = defaultdict(lambda: {"rows": 0, "nonnull": 0, "qm_nonnull": 0, "keep_strict": 0, "keep_lenient": 0})

    scanner = ds.Scanner.from_dataset(dataset, columns=cols, filter=filt, batch_size=1_000_000)
    total_rows = 0
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        total_rows += len(df)
        # per-QM distributions
        for q in qm_cols:
            s = df[q]
            vals, cnts = np.unique(s.fillna(-1).to_numpy(), return_counts=True)
            for v, n in zip(vals, cnts):
                v_int = int(v) if pd.notna(v) else -1
                qc_counts[q][v_int] += int(n)
            totals[q] += s.notna().sum()
        # per-feature keep stats
        for feat, (vcol, qcol) in FEATURES.items():
            s = df[vcol]
            feat_stats[feat]["rows"] += len(s)
            nn = s.notna()
            feat_stats[feat]["nonnull"] += int(nn.sum())
            if qcol:
                q = df[qcol].fillna(-1).astype(int)
                feat_stats[feat]["qm_nonnull"] += int((df[qcol].notna()).sum())
                feat_stats[feat]["keep_strict"] += int(((q.isin(list(KEEP_STRICT))) & nn).sum())
                feat_stats[feat]["keep_lenient"] += int(((q.isin(list(KEEP_LENIENT))) & nn).sum())
            else:
                # no QM: consider all non-null as "kept" for both policies
                feat_stats[feat]["keep_strict"] += int(nn.sum())
                feat_stats[feat]["keep_lenient"] += int(nn.sum())

    # tidy QC summary
    rows = []
    for q, d in qc_counts.items():
        tot = sum(d.values())
        for val, cnt in sorted(d.items(), key=lambda kv: kv[0]):
            rows.append({"qc_column": q, "qm_value": val, "count": cnt, "fraction": (cnt / tot) if tot else np.nan, "total": tot})
    qc_summary = pd.DataFrame(rows).sort_values(["qc_column", "qm_value"]).reset_index(drop=True)

    # per-feature summary
    fr = []
    for feat, st in FEATURES.items():
        s = feat_stats[feat]
        fr.append(
            {
                "feature": feat,
                "value_col": st and st,  # unused, placeholder
                "rows_scanned": s["rows"],
                "values_nonnull": s["nonnull"],
                "qm_nonnull": s["qm_nonnull"],
                "kept_strict_14": s["keep_strict"],
                "kept_lenient_-1_0_1_2_4": s["keep_lenient"],
            }
        )
    feat_summary = pd.DataFrame(fr)
    feat_summary["strict_fraction"] = feat_summary["kept_strict_14"] / feat_summary["values_nonnull"].replace(0, np.nan)
    feat_summary["lenient_fraction"] = feat_summary["kept_lenient_-1_0_1_2_4"] / feat_summary["values_nonnull"].replace(0, np.nan)

    # write & print
    qc_summary.to_csv(args.out, index=False)
    feat_summary.to_csv(args.out.replace(".csv", "_features.csv"), index=False)

    print(f"\nScanned rows: {total_rows:,}")
    print("\nQC value distribution (head):")
    print(qc_summary.head(20))
    print("\nPer-feature keep summary:")
    print(feat_summary)


if __name__ == "__main__":
    main()
