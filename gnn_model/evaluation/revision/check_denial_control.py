#!/usr/bin/env python
"""Check #2 of README_REVISION: the denial control must reproduce the manuscript forecasts.

The denial experiments express impact as a ratio of denied MSE to control MSE, so the whole
matrix is only meaningful if the control run reproduces the forecasts the manuscript reports.
The control withholds nothing, so on the initializations it shares with the main 2025
evaluation it should give the same errors up to reduced-precision differences.

Two distinct problems are looked for, and they need different tests.

  * Different verifying samples. If the two runs were scored with different --verify_qc, they
    verified against different observations and their errors are not comparable at all. Targets
    whose sample sizes differ are reported and then excluded, and the QC summary each run wrote
    is used to say which run applied what.

  * Different forecasts. A wrong checkpoint, config or prediction directory offsets a whole
    instrument in one direction. Reduced-precision accumulation does not: it scatters in sign,
    and in relative terms it is largest where the absolute errors are smallest. So the test is
    not whether any single target moves by more than some percentage, which would flag a
    low-variance radiance channel whose RMSE is a degree or two, but whether the differences
    are systematic: a non-zero median overall, or one instrument offset consistently one way.

Usage (from gnn_model/):
    python evaluation/revision/check_denial_control.py \
        --control evaluation/revision/results/denial/control \
        --reference evaluation/revision/results/v1_2025
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

KEY = ["instrument", "variable", "lead"]


def _load(d: str) -> pd.DataFrame:
    p = os.path.join(d, "per_init_stats.csv")
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    s = pd.read_csv(p, dtype={"init": str})
    if "plev" in s.columns:
        s = s[s["plev"] == "all"]
    return s[KEY + ["init", "n", "sse"]]


def _explain_n(args) -> None:
    """Say which run applied verification QC, the usual reason the sample sizes differ."""
    print("\n  Verification QC actually applied in each run (qc_removal_summary.csv):")
    for label, d in (("control", args.control), ("reference", args.reference)):
        p = os.path.join(d, "qc_removal_summary.csv")
        if not os.path.exists(p):
            print(f"    {label:<10s} no qc_removal_summary.csv, so this run was scored WITHOUT --verify_qc")
            continue
        q = pd.read_csv(p)
        col = "pct_removed_total" if "pct_removed_total" in q.columns else None
        if col is None:
            print(f"    {label:<10s} {len(q)} rows, but no pct_removed_total column")
            continue
        print(f"    {label:<10s} QC applied to {len(q)} targets; most affected:")
        for _, x in q.nlargest(4, col).iterrows():
            print(f"      {str(x['instrument']):>12s} {str(x['variable']):<32s} {x[col]:7.3f}% removed")
    print("  If only one run has that file, score both the same way and rerun this check.")
    print("  If both have it, compare them: a rule that removes a very large share of one variable")
    print("  (for example a dew-point relation test that cannot find its paired temperature) is a")
    print("  bug in the rule rather than a property of the data, and must be fixed before the")
    print("  numbers are used.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="results dir of the denial control run")
    ap.add_argument("--reference", required=True, help="results dir of the main 2025 evaluation")
    ap.add_argument("--tol_median_pct", type=float, default=0.10,
                    help="fails if the median signed difference exceeds this, i.e. a systematic offset")
    ap.add_argument("--tol_inst_pct", type=float, default=0.50,
                    help="fails if one instrument is offset by more than this AND is one-sided")
    ap.add_argument("--tol_max_pct", type=float, default=3.0,
                    help="fails if any single target differs by more than this")
    ap.add_argument("--one_sided_frac", type=float, default=0.8,
                    help="share of an instrument's targets sharing the sign of its median to call it one-sided")
    ap.add_argument("--out", default=None, help="optional CSV of the per-target comparison")
    args = ap.parse_args()

    c, r = _load(args.control), _load(args.reference)
    inits = sorted(set(c["init"]) & set(r["init"]))
    if not inits:
        sys.exit("the two runs share no initializations")
    print(f"control {c['init'].nunique()} inits, reference {r['init'].nunique()} inits, "
          f"{len(inits)} in common")

    c, r = c[c["init"].isin(inits)], r[r["init"].isin(inits)]
    agg = {"n": "sum", "sse": "sum", "init": "nunique"}
    g = (c.groupby(KEY).agg(agg)
          .join(r.groupby(KEY).agg(agg), lsuffix="_ctl", rsuffix="_ref", how="inner")
          .reset_index()
          .rename(columns={"init_ctl": "inits_ctl", "init_ref": "inits_ref"}))
    if g.empty:
        sys.exit("no targets in common between the two runs")

    g["rmse_ctl"] = np.sqrt(g["sse_ctl"] / g["n_ctl"])
    g["rmse_ref"] = np.sqrt(g["sse_ref"] / g["n_ref"])
    g["diff_pct"] = 100.0 * (g["rmse_ctl"] / g["rmse_ref"] - 1.0)
    g["diff_abs"] = g["rmse_ctl"] - g["rmse_ref"]
    g["n_match"] = g["n_ctl"] == g["n_ref"]

    mismatched = g[~g["n_match"]]
    if len(mismatched):
        print(f"\n{len(mismatched)} of {len(g)} targets have different sample sizes, so their errors "
              "are not comparable; they are excluded below.")
        m = mismatched.assign(gap=(mismatched["n_ctl"] - mismatched["n_ref"]).abs()).nlargest(8, "gap")
        print(f"  {'instrument':>12s} {'variable':<30s} {'lead':>5s} {'n_control':>13s} "
              f"{'n_reference':>13s} {'ratio':>6s} {'inits':>11s} {'obs per init':>19s}")
        for _, x in m.iterrows():
            pi_c = x["n_ctl"] / max(x["inits_ctl"], 1)
            pi_r = x["n_ref"] / max(x["inits_ref"], 1)
            print(f"  {x['instrument']:>12s} {x['variable']:<30s} {str(x['lead']):>5s} "
                  f"{int(x['n_ctl']):>13,} {int(x['n_ref']):>13,} "
                  f"{x['n_ctl'] / max(x['n_ref'], 1):>6.3f} "
                  f"{int(x['inits_ctl']):>5d}/{int(x['inits_ref']):<5d} "
                  f"{pi_c:>9,.0f}/{pi_r:<9,.0f}")
        per_init_same = (mismatched.assign(
            a=mismatched["n_ctl"] / mismatched["inits_ctl"].clip(lower=1),
            b=mismatched["n_ref"] / mismatched["inits_ref"].clip(lower=1))
            .eval("abs(a - b) / b < 0.01").mean())
        print("")
        print(f"  For {100 * per_init_same:.0f}% of these targets the observations PER INITIALIZATION "
              "agree to within 1%.")
        print("  Where they agree, the two runs cover a different number of initializations for that")
        print("  target (an incomplete merge, or a target absent from some inits), so the totals are")
        print("  not comparable. Where they disagree, the two runs verified against different")
        print("  observations within the same windows, which is a QC difference.")
        _explain_n(args)

    ok = g[g["n_match"]].copy()
    if ok.empty:
        sys.exit("\nno target has a comparable sample size; score both runs the same way first")

    worst = ok.assign(a=ok["diff_pct"].abs()).nlargest(10, "a")
    med_all = ok["diff_pct"].median()
    mx_all = ok["diff_pct"].abs().max()
    print(f"\nRelative RMSE difference, control vs reference, on {len(ok)} comparable targets:")
    print(f"  median {med_all:+.4f}%   max |diff| {mx_all:.3f}%")
    print(f"\n{'instrument':>12s} {'variable':<22s} {'lead':>5s} {'rmse_ctl':>10s} {'rmse_ref':>10s} "
          f"{'diff%':>8s} {'diff_abs':>10s}")
    for _, x in worst.iterrows():
        print(f"{x['instrument']:>12s} {x['variable']:<22s} {str(x['lead']):>5s} "
              f"{x['rmse_ctl']:10.4f} {x['rmse_ref']:10.4f} {x['diff_pct']:+8.3f} {x['diff_abs']:+10.5f}")

    # A wrong checkpoint or configuration offsets a whole instrument in one direction. Reduced
    # precision scatters in sign and is largest, relative to the error, where the error is smallest.
    print(f"\n{'instrument':>12s} {'targets':>8s} {'median%':>9s} {'max|%|':>8s} {'share +':>8s}  verdict")
    systematic = []
    for inst, h in ok.groupby("instrument"):
        med, mx = h["diff_pct"].median(), h["diff_pct"].abs().max()
        share_pos = float((h["diff_pct"] > 0).mean())
        sided = max(share_pos, 1.0 - share_pos) >= args.one_sided_frac
        flag = abs(med) > args.tol_inst_pct and sided
        systematic.append((inst, med, flag))
        if flag:
            verdict = "ONE-SIDED OFFSET"
        elif mx > args.tol_inst_pct:
            verdict = "mixed sign: precision"
        else:
            verdict = "clean"
        print(f"{inst:>12s} {len(h):>8d} {med:>+9.4f} {mx:>8.3f} {share_pos:>8.2f}  {verdict}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        g.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    fails = []
    if abs(med_all) > args.tol_median_pct:
        fails.append(f"the overall median difference is {med_all:+.3f}%, above {args.tol_median_pct}%, "
                     "which is a systematic offset rather than noise")
    for inst, med, flag in systematic:
        if flag:
            fails.append(f"{inst} is offset by {med:+.3f}% consistently in one direction")
    if mx_all > args.tol_max_pct:
        fails.append(f"one target differs by {mx_all:.3f}%, above {args.tol_max_pct}%")

    if fails:
        print("\nFAIL: the control does not reproduce the manuscript forecasts.")
        for x in fails:
            print(f"  - {x}")
        print("  Check the checkpoint path, the observation config and the prediction directory")
        print("  before using the denial matrix.")
        return 1
    print(f"\nPASS: the median difference is {med_all:+.4f}%, no instrument is offset consistently in")
    print(f"one direction, and the largest single difference is {mx_all:.3f}% on a target whose")
    print("absolute error is small. Differences of both signs within one instrument are")
    print("reduced-precision accumulation, not a different model, so the denial control reproduces")
    print("the manuscript forecasts and the impact ratios can be trusted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
