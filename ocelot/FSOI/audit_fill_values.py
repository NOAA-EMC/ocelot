#!/usr/bin/env python
"""Audit configured observation columns for fill values and range violations.

Run on the HPC with --cfg_path and --data_path for the experiment being checked.
Samples are spread across each store; repeated values are alerts, not proof of a
sentinel. Physical bounds still need confirmation against the observation data.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


BASE = "/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7"
MODE_ALERT = 0.01
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "observation_config.yaml"


def build_spec(observation_config):
    """Resolve store columns and bounds from the same configuration as training."""
    dependencies = {
        "wind_u": ("windSpeed", "windDirection"),
        "wind_v": ("windSpeed", "windDirection"),
        "log_pressure_height": ("airPressure",),
    }
    spec = {}
    for instruments in observation_config.values():
        for inst, cfg in instruments.items():
            features = cfg.get("features", [])
            columns = []
            for name in features + cfg.get("metadata", []):
                columns.extend(dependencies.get(name, (name,)))
            columns = list(dict.fromkeys(columns))
            qc = cfg.get("qc_filters") or cfg.get("qc") or {}
            ranges = {}
            for col in columns:
                rule = qc.get(col)
                bounds = rule.get("range") if isinstance(rule, dict) else rule
                if bounds is None and col in features:
                    bounds = cfg.get("feature_range")
                if bounds is not None:
                    ranges[col] = tuple(bounds)
            spec[inst] = dict(zarr=cfg.get("zarr_name", inst), cols=columns, ranges=ranges)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data_path", type=Path, default=Path(BASE))
    parser.add_argument("--sample_rows", type=int, default=300_000)
    args = parser.parse_args()
    if args.sample_rows <= 0:
        parser.error("--sample_rows must be positive")
    with args.cfg_path.open(encoding="utf-8") as stream:
        spec = build_spec(yaml.safe_load(stream)["observation_config"])

    import zarr

    print(f"{'instrument':12s} {'column':34s} {'nonfin':>7} {'>=1e6':>7} "
          f"{'out_rng':>7} {'modal value':>16} {'share':>7} {'min':>11} {'max':>11}  flag")
    for inst, entry in spec.items():
        zname = entry["zarr"]
        path = args.data_path / (zname if zname.endswith(".zarr") else f"{zname}.zarr")
        try:
            store = zarr.open(str(path), mode="r")
        except Exception as exc:
            print(f"{inst:12s} !! cannot open {path}: {type(exc).__name__}: {exc}")
            continue
        for col in entry["cols"]:
            if col not in store:
                print(f"{inst:12s} {col:34s} !! not present in store")
                continue
            array = store[col]
            if len(array) == 0:
                print(f"{inst:12s} {col:34s} EMPTY")
                continue
            indices = np.linspace(0, len(array) - 1, min(args.sample_rows, len(array)), dtype=np.int64)
            values = np.asarray(array.oindex[indices], dtype=np.float64)
            nonfin = float((~np.isfinite(values)).mean())
            huge = float((np.isfinite(values) & (np.abs(values) >= 1e6)).mean())
            finite = values[np.isfinite(values)]
            modal, share, lo, hi = float("nan"), 0.0, float("nan"), float("nan")
            if finite.size:
                unique, counts = np.unique(finite, return_counts=True)
                k = int(counts.argmax())
                modal, share = float(unique[k]), float(counts[k] / finite.size)
                lo, hi = float(finite.min()), float(finite.max())
            flags = []
            if nonfin > 0:
                flags.append("NONFINITE")
            if huge > 0:
                flags.append("HUGE")
            if share >= MODE_ALERT:
                flags.append(f"REPEATED({share:.1%})")
            out_range = float("nan")
            if col in entry["ranges"]:
                lower, upper = entry["ranges"][col]
                out_range = float(((values < lower) | (values > upper)).mean())
                flags.append(f"range=[{lower},{upper}]")
            print(f"{inst:12s} {col:34s} {nonfin:7.3f} {huge:7.3f} {out_range:7.3f} "
                  f"{modal:16.4g} {share:7.3f} {lo:11.4g} {hi:11.4g}  {' '.join(flags)}")


if __name__ == "__main__":
    main()
