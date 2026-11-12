#!/usr/bin/env python
"""Test whether per-instrument quadrature residuals cancel along the combined path.

Reads the run produced by scripts/submit_combined_path.sh, in which every source is
denied together so the denied path is the combined-replacement path. For each cycle and
each instrument it compares

    two-endpoint contribution   I2  (matched_fsoi_by_instrument)
    five-point contribution     I5  = Simpson quadrature of that instrument's
                                     directional derivatives along the same path

and reports the residual r = I2 - I5. Cancellation is measured by |sum r| / sum |r|:
near zero means the per-instrument residuals offset one another, near one means they
accumulate in the same direction.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FSOI = Path(__file__).resolve().parent
SIMPSON = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0


def parse_map(text, cast):
    out = {}
    for item in str(text).split(";"):
        if ":" not in item:
            continue
        name, value = item.split(":", 1)
        out[name.strip()] = cast(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=FSOI / "fsoi_outputs" / "combined_path" / "radiosonde_jul2025")
    args = parser.parse_args()

    d = pd.read_csv(args.run / "evaluation" / "ose_results.csv")
    d = d[d.path_integrated_fsoi.notna()]
    if d.empty:
        raise SystemExit("No path-integration rows; check OSE_PATH_INTEGRATION_PAIR_INDICES")

    rows, cyc = [], []
    for r in d.itertuples():
        t_values = [float(x) for x in str(r.path_integration_t_values).split(",")]
        assert np.allclose(t_values, [0, 0.25, 0.5, 0.75, 1]), t_values
        two = parse_map(r.matched_fsoi_by_instrument, float)
        deriv = parse_map(r.path_directional_derivatives_by_instrument,
                          lambda s: np.array([float(x) for x in s.split(",")]))
        assert set(two) == set(deriv), (sorted(two), sorted(deriv))
        five = {k: float(SIMPSON @ v) for k, v in deriv.items()}
        for inst in sorted(two):
            rows.append(dict(cycle=str(r.curr_bin)[3:], instrument=inst, I2=two[inst],
                             I5=five[inst], residual=two[inst] - five[inst]))
        res = np.array([two[i] - five[i] for i in two])
        cyc.append(dict(cycle=str(r.curr_bin)[3:], dJ=r.delta_j_actual,
                        I2_total=sum(two.values()), I5_total=sum(five.values()),
                        rho2=r.matched_closure_ratio, rho5=r.path_closure_ratio,
                        sum_residual=res.sum(), sum_abs_residual=np.abs(res).sum(),
                        cancellation=abs(res.sum()) / np.abs(res).sum(),
                        n_same_sign_as_total=int((np.sign(res) == np.sign(res.sum())).sum())))

    per = pd.DataFrame(rows)
    tot = pd.DataFrame(cyc)
    pd.set_option("display.width", 220)
    print("Per-instrument contributions along the combined path\n")
    print(per.to_string(index=False, float_format=lambda v: f"{v: .3e}"))
    print("\nPer cycle\n")
    print(tot.to_string(index=False, float_format=lambda v: f"{v: .4g}"))

    print("\nHow to read the cancellation column:")
    print("  |sum r| / sum |r| near 0  -> per-instrument residuals offset one another")
    print("     (the combined total is better than its parts, i.e. errors cancel)")
    print("  near 1                    -> residuals share a sign and add up")
    print("     (no cancellation; combined and conditional differ for other reasons)")
    print(f"\nmedian cancellation = {tot.cancellation.median():.3f}; "
          f"median |sum r| = {tot.sum_residual.abs().median():.3e}; "
          f"median sum |r| = {tot.sum_abs_residual.median():.3e}")
    out = args.run / "combined_path_residuals.csv"
    per.to_csv(out, index=False)
    tot.to_csv(out.with_name("combined_path_cycles.csv"), index=False)
    print(f"\nwrote {out} and {out.with_name('combined_path_cycles.csv')}")


if __name__ == "__main__":
    main()
