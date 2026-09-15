#!/usr/bin/env python3
"""Join externally supplied channel physics metadata to FSOI results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["instrument", "channel"]
ANNOTATIONS = [
    "channel_frequency_ghz",
    "weighting_function_peak_hpa",
    "sensitivity_class",
    "temperature_sensitivity",
    "water_vapor_sensitivity",
    "cloud_contamination_risk",
    "surface_sensitivity",
    "geometry_notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    metadata = pd.read_csv(args.metadata)
    for name, frame in (("results", results), ("metadata", metadata)):
        missing = [key for key in KEYS if key not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required key columns: {missing}")
    if metadata.duplicated(KEYS).any():
        raise ValueError("metadata must contain one row per instrument/channel")

    annotations = [column for column in ANNOTATIONS if column in metadata.columns]
    if not annotations:
        raise ValueError("metadata contains no recognized physics annotation columns")
    overlap = set(annotations) & set(results.columns)
    if overlap:
        raise ValueError(f"refusing to overwrite result columns: {sorted(overlap)}")

    enriched = results.merge(
        metadata[KEYS + annotations], on=KEYS, how="left",
        validate="many_to_one", indicator="physics_metadata_match",
    )
    enriched["physics_metadata_match"] = enriched["physics_metadata_match"].eq("both")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    matched = int(enriched["physics_metadata_match"].sum())
    print(f"Wrote {len(enriched):,} rows to {args.output}")
    print(f"Physics metadata matched: {matched:,}/{len(enriched):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
