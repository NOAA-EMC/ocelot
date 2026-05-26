#!/usr/bin/env python
"""
Compute FSOI-derived observation weights for OCELOT loss function.

Reads one or more fsoi_by_instrument.csv files (from multiple periods/seasons),
aggregates them, computes per-instrument and per-(instrument, variable) weights,
and writes a new observation_config_fsoi_weighted.yaml ready for fine-tuning.

Two weight variants are computed (see Methods section of paper):

  Variant A — Magnitude:
    w_inst = |mean_fsoi_inst| / mean(|mean_fsoi|)
    Proportional to information content (how much this instrument changes error).

  Variant B — Magnitude × Reliability:
    w_inst = |mean_fsoi_inst| × (1 - positive_frac_inst)² / normalizer
    Weights by both impact magnitude and consistency of benefit.

Both variants are normalized so mean(w) = 1.0, preserving total loss scale.

Usage:
    python FSOI/compute_fsoi_weights.py \
        --fsoi_dirs  FSOI/fsoi_outputs/*/csv \
        --obs_config configs/observation_config.yaml \
        --output_dir FSOI/fsoi_weights/ \
        [--min_pairs 20]

Outputs:
    fsoi_weights/
        fsoi_weight_summary.csv          — human-readable weight table
        observation_config_variant_a.yaml — Variant A weights
        observation_config_variant_b.yaml — Variant B weights
        observation_config_channel_weighted.yaml — per-variable channel weights
"""

import argparse
import copy
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required.  conda activate gnn-env")
    sys.exit(1)


# ── Weight computation ────────────────────────────────────────────────────────

def load_and_aggregate_fsoi(fsoi_dirs: list[str], min_pairs: int = 20) -> pd.DataFrame:
    """Load all fsoi_by_instrument.csv files and aggregate per instrument."""
    frames = []
    for pattern in fsoi_dirs:
        paths = sorted(glob.glob(str(pattern)))
        for p in paths:
            path = Path(p)
            if path.is_dir():
                path = path / "fsoi_by_instrument.csv"
            if not path.is_file():
                continue
            try:
                df = pd.read_csv(path)
                df['_source'] = str(path)
                frames.append(df)
                print(f"  Loaded: {path} ({len(df)} rows, "
                      f"{df['pair_idx'].nunique()} pairs)")
            except Exception as e:
                print(f"  SKIP {path}: {e}")

    if not frames:
        raise FileNotFoundError("No fsoi_by_instrument.csv files found in provided paths")

    df = pd.concat(frames, ignore_index=True)
    total_pairs = df['pair_idx'].nunique()
    print(f"\nTotal: {len(df)} rows across {total_pairs} pairs from {len(frames)} files")

    if total_pairs < min_pairs:
        print(f"WARNING: Only {total_pairs} pairs — results may not be robust. "
              f"Recommend >= {min_pairs}.")

    return df


def compute_instrument_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-instrument weight variants from aggregated FSOI."""
    impact_col = 'sum_impact_scaled' if 'sum_impact_scaled' in df.columns else 'sum_impact'

    # Per-instrument aggregates across all pairs
    agg = df.groupby('instrument').agg(
        n_pairs=('pair_idx', 'nunique'),
        n_obs_total=(impact_col, 'count'),
        mean_impact=('mean_impact', 'mean'),
        sum_impact=(impact_col, 'sum'),
        positive_frac_mean=('positive_frac', 'mean'),
        positive_frac_std=('positive_frac', 'std'),
    ).reset_index()

    # Per-observation mean impact (consistent with positive_frac scale)
    agg['abs_mean_impact'] = agg['mean_impact'].abs()

    # ── Variant A: proportional to |mean_impact| ──────────────────────────
    denom_a = agg['abs_mean_impact'].mean()
    if denom_a > 0:
        agg['weight_variant_a'] = (agg['abs_mean_impact'] / denom_a).clip(lower=0.1)
    else:
        agg['weight_variant_a'] = 1.0

    # Renormalize so mean = 1.0
    agg['weight_variant_a'] /= agg['weight_variant_a'].mean()

    # ── Variant B: magnitude × reliability ────────────────────────────────
    # reliability = (1 - positive_frac)^2   [0=all detrimental, 1=all helpful]
    agg['reliability'] = (1.0 - agg['positive_frac_mean']).clip(0, 1) ** 2
    raw_b = agg['abs_mean_impact'] * agg['reliability']
    denom_b = raw_b.mean()
    if denom_b > 0:
        agg['weight_variant_b'] = (raw_b / denom_b).clip(lower=0.1)
    else:
        agg['weight_variant_b'] = 1.0

    agg['weight_variant_b'] /= agg['weight_variant_b'].mean()

    return agg


def compute_channel_weights(df: pd.DataFrame) -> dict:
    """Compute per-(instrument, variable/channel) weights.

    Returns {instrument_name: [w_ch0, w_ch1, ...]} normalized per instrument
    so mean channel weight within each instrument = 1.0.
    """
    if 'target_variable' not in df.columns or 'target_channel' not in df.columns:
        print("  NOTE: No per-variable stratification in CSV — channel weights = uniform")
        return {}

    impact_col = 'sum_impact_scaled' if 'sum_impact_scaled' in df.columns else 'sum_impact'

    ch_weights = {}
    for inst, grp in df.groupby('instrument'):
        ch_agg = grp.groupby(['target_channel', 'target_variable'])['mean_impact'].mean().reset_index()
        ch_agg['abs_mean_impact'] = ch_agg['mean_impact'].abs()
        ch_agg = ch_agg.sort_values('target_channel')

        if ch_agg.empty:
            continue

        raw = ch_agg['abs_mean_impact'].values
        denom = raw.mean()
        if denom > 0:
            w = (raw / denom).clip(0.1).tolist()
        else:
            w = [1.0] * len(raw)

        # Renormalize within instrument
        w_arr = np.array(w)
        w_arr = (w_arr / w_arr.mean()).tolist()

        ch_weights[inst] = {
            'weights': w_arr,
            'channel_labels': ch_agg['target_variable'].tolist(),
        }

    return ch_weights


def patch_observation_config(
    base_config: dict,
    instrument_weights: dict,
    channel_weights: dict = None,
) -> dict:
    """Return a deep copy of base_config with updated instrument/channel weights."""
    cfg = copy.deepcopy(base_config)

    # Update instrument_weights section
    if 'instrument_weights' not in cfg:
        cfg['instrument_weights'] = {}
    for inst, w in instrument_weights.items():
        cfg['instrument_weights'][inst] = round(float(w), 6)

    # Update channel_weights section
    if channel_weights:
        if 'channel_weights' not in cfg:
            cfg['channel_weights'] = {}
        for inst, info in channel_weights.items():
            w_list = [round(float(v), 6) for v in info['weights']]
            cfg['channel_weights'][inst] = w_list

    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FSOI-derived weights for OCELOT loss function")
    parser.add_argument(
        "--fsoi_dirs", nargs="+", required=True,
        help="Paths to directories containing fsoi_by_instrument.csv, or glob patterns. "
             "Multiple periods can be combined, e.g. FSOI/fsoi_outputs/*/csv")
    parser.add_argument(
        "--obs_config",
        default="configs/observation_config.yaml",
        help="Path to base observation_config.yaml (default: configs/observation_config.yaml)")
    parser.add_argument(
        "--output_dir",
        default="FSOI/fsoi_weights",
        help="Directory to write weight outputs (default: FSOI/fsoi_weights)")
    parser.add_argument(
        "--min_pairs", type=int, default=20,
        help="Warn if fewer than this many pairs are available (default: 20)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load and aggregate FSOI data ──────────────────────────────────
    print("Loading FSOI data...")
    df = load_and_aggregate_fsoi(args.fsoi_dirs, min_pairs=args.min_pairs)

    # ── 2. Compute weights ────────────────────────────────────────────────
    print("\nComputing instrument weights...")
    inst_df = compute_instrument_weights(df)

    print("\nInstrument weight table:")
    display = inst_df[[
        'instrument', 'n_pairs', 'mean_impact', 'positive_frac_mean',
        'reliability', 'weight_variant_a', 'weight_variant_b',
    ]].sort_values('weight_variant_a', ascending=False)
    print(display.to_string(index=False))

    weight_csv = out_dir / "fsoi_weight_summary.csv"
    inst_df.to_csv(weight_csv, index=False)
    print(f"\nSaved: {weight_csv}")

    print("\nComputing per-variable channel weights...")
    ch_weights = compute_channel_weights(df)
    if ch_weights:
        print("  Per-instrument channel weights:")
        for inst, info in ch_weights.items():
            labels = info['channel_labels']
            weights = info['weights']
            print(f"    {inst}: " +
                  "  ".join(f"{l}={w:.3f}" for l, w in zip(labels, weights)))

    # ── 3. Load base config ───────────────────────────────────────────────
    obs_config_path = Path(args.obs_config)
    if not obs_config_path.is_file():
        print(f"ERROR: {obs_config_path} not found — writing weight CSV only")
        return

    with open(obs_config_path) as f:
        base_config = yaml.safe_load(f)

    # ── 4. Write Variant A config ─────────────────────────────────────────
    weights_a = dict(zip(inst_df['instrument'], inst_df['weight_variant_a']))
    cfg_a = patch_observation_config(base_config, weights_a)
    path_a = out_dir / "observation_config_variant_a.yaml"
    with open(path_a, 'w') as f:
        yaml.dump(cfg_a, f, default_flow_style=False, sort_keys=False)
    print(f"Saved Variant A config: {path_a}")
    print("  Variant A instrument weights:")
    for inst, w in sorted(weights_a.items(), key=lambda kv: -kv[1]):
        print(f"    {inst}: {w:.3f}")

    # ── 5. Write Variant B config ─────────────────────────────────────────
    weights_b = dict(zip(inst_df['instrument'], inst_df['weight_variant_b']))
    cfg_b = patch_observation_config(base_config, weights_b)
    path_b = out_dir / "observation_config_variant_b.yaml"
    with open(path_b, 'w') as f:
        yaml.dump(cfg_b, f, default_flow_style=False, sort_keys=False)
    print(f"Saved Variant B config: {path_b}")

    # ── 6. Write channel-weighted config ─────────────────────────────────
    if ch_weights:
        cfg_ch = patch_observation_config(base_config, weights_a, ch_weights)
        path_ch = out_dir / "observation_config_channel_weighted.yaml"
        with open(path_ch, 'w') as f:
            yaml.dump(cfg_ch, f, default_flow_style=False, sort_keys=False)
        print(f"Saved channel-weighted config: {path_ch}")

    # ── 7. Print usage instructions ──────────────────────────────────────
    print("\n" + "="*60)
    print("Next steps — fine-tune with FSOI weights:")
    print("="*60)
    print(f"""
# Variant A (magnitude only):
python train_gnn.py \\
    --obs_config {path_a} \\
    --checkpoint /path/to/Epoch3079_fixedval.ckpt \\
    --max_epochs 500 --lr 1e-5  # 10x smaller LR for fine-tuning

# Variant B (magnitude × reliability):
python train_gnn.py \\
    --obs_config {path_b} \\
    --checkpoint /path/to/Epoch3079_fixedval.ckpt \\
    --max_epochs 500 --lr 1e-5

# Evaluate RMSE comparison:
python evaluation/run_pred_eval_gfs.py --checkpoint <finetuned.ckpt>
""")
    print("="*60)


if __name__ == "__main__":
    main()
