"""Summarize the metadata-only target audit without using FSOI signs or losses."""

import argparse
from pathlib import Path
import pandas as pd


def summarize(root):
    frames = []
    for path in sorted(Path(root).rglob('target_metric/pair*.csv')):
        df = pd.read_csv(path)
        df['run'] = str(path.parents[2])
        frames.append(df)
    if not frames:
        raise ValueError(f"No target-metric coverage files beneath {root}")
    raw = pd.concat(frames, ignore_index=True)
    keys = ['run', 'target_instrument', 'variable', 'pressure_hpa']
    grouped = raw.groupby(keys, dropna=False)
    summary = grouped.agg(
        cycles=('eligible', 'size'), eligible_cycles=('eligible', 'sum'),
        min_cells=('n_cells', 'min'), median_cells=('n_cells', 'median'),
        min_targets=('n_scored_targets', 'min'), median_targets=('n_scored_targets', 'median'),
        mean_invalid_coordinates=('n_invalid_coordinates', 'mean'),
    ).reset_index()
    summary['eligible_fraction'] = summary['eligible_cycles'] / summary['cycles']
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print("Select supported groups from coverage, then freeze the definition before examining impact results.")


if __name__ == '__main__':
    main()
