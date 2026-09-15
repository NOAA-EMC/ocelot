"""Compare channel rankings under scalings saved for the same sampled rows.

New runs compare HT, N/n, raw totals, and valid-value sample means. Archived
N/n-only results are supported explicitly, without inventing missing HT weights.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SATELLITES = ("atms", "amsua", "ssmis", "ascat", "avhrr", "seviri_asr")
KEYS = ["target", "month", "instrument", "channel", "pair_idx"]
BASE_COLS = ["instrument", "channel", "pair_idx", "sum_impact", "sum_impact_scaled",
             "total_count", "raw_total_count", "sample_scale"]
HT_COLS = {"sum_impact_ht", "sum_impact_scaled_uniform", "population_scaling_method"}


def read_cycle_scalings(path, target, month, require_ht=False, chunksize=100000):
    """Validate the saved convention and average target groups within cycles."""
    columns = set(pd.read_csv(path, nrows=0).columns)
    has_ht = HT_COLS.issubset(columns)
    if (columns & HT_COLS and not has_ht) or (require_ht and not has_ht):
        raise ValueError(f"{path}: complete HT columns are required; found {columns & HT_COLS}")
    usecols = BASE_COLS + (sorted(HT_COLS) if has_ht else [])
    metrics = ["uniform_total", "raw_total", "valid_value_mean"]
    if has_ht:
        metrics.insert(0, "ht_total")
    totals, counts, missing = [], [], []
    with pd.read_csv(path, usecols=usecols, chunksize=chunksize) as reader:
        for df in reader:
            df = df[df.instrument.isin(SATELLITES)].copy()
            if df.empty:
                continue
            if has_ht:
                if not df.population_scaling_method.eq("horvitz_thompson_row_inclusion").all():
                    raise ValueError(f"{path}: mixed or unrecognized population scaling methods")
                np.testing.assert_allclose(df.sum_impact_scaled, df.sum_impact_ht)
                df["ht_total"] = df.sum_impact_ht
                df["uniform_total"] = df.sum_impact_scaled_uniform
            else:
                df["uniform_total"] = df.sum_impact_scaled
            np.testing.assert_allclose(df.uniform_total, df.sum_impact * df.sample_scale,
                                       rtol=1e-5, atol=1e-10)
            if not np.isfinite(df[["sum_impact", "sum_impact_scaled", "sample_scale"]]).all().all():
                raise ValueError(f"{path}: non-finite totals or sample scales")
            if ((df.total_count < 0) | (df.total_count > df.raw_total_count)).any():
                raise ValueError(f"{path}: invalid sampled channel counts")
            df["target"], df["month"] = target, month
            df["raw_total"] = df.sum_impact
            df["valid_value_mean"] = df.sum_impact / df.total_count.replace(0, np.nan)
            grouped = df.groupby(KEYS)[metrics]
            totals.append(grouped.sum())
            counts.append(grouped.count())
            df["valid_fraction"] = df.total_count / df.raw_total_count.replace(0, np.nan)
            missing.append(df[["instrument", "channel", "pair_idx", "total_count",
                               "raw_total_count", "valid_fraction"]].drop_duplicates())
    if not totals:
        raise ValueError(f"{path}: no satellite channel records")
    sums = pd.concat(totals).groupby(KEYS).sum()
    count = pd.concat(counts).groupby(KEYS).sum()
    # A channel without valid values has an undefined mean, not an observed zero.
    values = sums.div(count.replace(0, np.nan)).reset_index()
    return values, pd.concat(missing).drop_duplicates(), metrics


def rank_comparisons(result, metrics):
    baseline = metrics[0]
    comparisons = []
    for (target, month, inst), group in result.groupby(["target", "month", "instrument"]):
        for alternative in metrics[1:]:
            g = group[np.isfinite(group[baseline]) & np.isfinite(group[alternative])]
            a, b = g[baseline], g[alternative]
            ar, br = a.rank(ascending=False), b.rank(ascending=False)
            rho = ar.corr(br) if len(g) > 1 and ar.nunique() > 1 and br.nunique() > 1 else np.nan
            aa, ba = a.abs().rank(ascending=False), b.abs().rank(ascending=False)
            rho_abs = aa.corr(ba) if len(g) > 1 and aa.nunique() > 1 and ba.nunique() > 1 else np.nan
            top_a = set(g.loc[a.nlargest(min(3, len(a))).index, "channel"])
            top_b = set(g.loc[b.nlargest(min(3, len(b))).index, "channel"])
            comparisons.append(dict(target=target, month=month, instrument=inst,
                baseline=baseline, alternative=alternative,
                n_channels=len(g), n_channels_excluded=len(group)-len(g),
                spearman_signed=rho, spearman_absolute=rho_abs,
                n_rank_changes=int((ar != br).sum()), max_rank_change=float((ar-br).abs().max()),
                n_sign_changes=int((np.sign(a) != np.sign(b)).sum()),
                top3_overlap=len(top_a & top_b),
                baseline_top3=";".join(map(str, sorted(top_a))),
                alternative_top3=";".join(map(str, sorted(top_b)))))
    return pd.DataFrame(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "fsoi_outputs" / "seasonal_sentinel_fixed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-ht", action="store_true", help="Fail on unweighted or partially updated runs")
    parser.add_argument("--expected-runs", type=int, help="Require this many target-month CSVs")
    args = parser.parse_args()
    out = args.output_dir or args.root / "sampling_rank_sensitivity"
    cycles, missingness = [], []
    metrics = None
    paths = sorted(args.root.glob("*/csv/fsoi_by_channel.csv"))
    if args.expected_runs is not None and len(paths) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} completed run CSVs, found {len(paths)}")
    for path in paths:
        target, month = path.parents[1].name.rsplit("_", 1)
        values, mask, file_metrics = read_cycle_scalings(path, target, month, args.require_ht)
        if metrics is not None and metrics != file_metrics:
            raise ValueError("Do not combine HT-weighted and N/n-only runs in one comparison")
        metrics = file_metrics
        cycles.append(values)
        for (inst, channel), g in mask.groupby(["instrument", "channel"]):
            missingness.append(dict(target=target, month=month, instrument=inst, channel=channel,
                n_cycle_count_records=len(g), minimum_valid_fraction=g.valid_fraction.min(),
                mean_valid_fraction=g.valid_fraction.mean(), n_records_missing=int(g.valid_fraction.lt(1).sum())))
        print(f"Read {path.parents[1].name}", flush=True)
    if not cycles:
        raise ValueError(f"No */csv/fsoi_by_channel.csv files found under {args.root}")
    cycle = pd.concat(cycles, ignore_index=True)
    monthly = cycle.groupby(["target", "month", "instrument", "channel"])[metrics].mean().reset_index()
    seasonal = monthly.groupby(["target", "instrument", "channel"])[metrics].mean().reset_index()
    month_count = monthly.groupby(["target", "instrument", "channel"]).month.nunique()
    seasonal["n_months"] = month_count.to_numpy()
    seasonal["month"] = np.where(seasonal.n_months.eq(4), "four_month_equal_mean", "available_month_equal_mean")
    result = pd.concat([monthly, seasonal], ignore_index=True)
    comparisons = rank_comparisons(result, metrics)
    out.mkdir(parents=True, exist_ok=True)
    cycle.to_csv(out / "channel_cycle_scaling.csv", index=False)
    result.to_csv(out / "channel_scaling_summary.csv", index=False)
    comparisons.to_csv(out / "channel_rank_comparisons.csv", index=False)
    pd.DataFrame(missingness).to_csv(out / "sample_channel_missingness.csv", index=False)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
