# Channel Statistical Strength Analysis

This document explains the strengthened statistical treatment for OCELOT FSOI channel results. The goal is to avoid treating target variable-pressure rows as independent replicates. Rows share cycles, model states, locations, and source observations, so the statistical unit should be the 12-hour cycle.

## What The Script Does

Script:

```text
gnn_model/FSOI/analyze_channel_statistical_strength.py
```

Current output folder:

```text
gnn_model/FSOI/fsoi_outputs/channel_statistical_strength_current/
```

The script reads the current seasonal channel outputs:

```text
gnn_model/FSOI/fsoi_outputs/seasonal_sentinel_fixed/{target}_{month}2025/csv/fsoi_by_channel.csv
```

It currently analyzes satellite source channels by default:

```text
atms, amsua, ssmis, avhrr, ascat, seviri_asr
```

## Statistical Method

For each target, month, instrument, channel, and physical target group, the script first aggregates row-level FSOI into one value per 12-hour cycle:

```text
cycle_channel_impact =
    sum of sum_impact_scaled over the selected target rows
```

Positive impact is detrimental. Negative impact is beneficial.

Physical target groups include:

- `all`
- individual variables such as `temperature`, `dewpoint_temperature`, `u_wind`, `v_wind`, `surface_pressure`
- combined `wind`
- vertical layers for pressure-level targets:
  - `lower_troposphere`: pressure >= 700 hPa
  - `middle_troposphere`: 300 <= pressure < 700 hPa
  - `upper_troposphere`: pressure < 300 hPa

For each channel/month/group, the script estimates:

- mean cycle impact
- median cycle impact
- median relative impact as percent of target-group control error
- detrimental-cycle fraction
- beneficial-cycle fraction
- practical detrimental-cycle fraction
- practical beneficial-cycle fraction
- one-sided bootstrap p-values for beneficial and detrimental impact

Uncertainty is estimated with a 5-day block bootstrap:

```text
block_cycles = 10
n_boot = 5000
```

The script then applies Benjamini-Hochberg false-discovery-rate correction within each:

```text
target x month x target_group x sign_direction
```

A monthly channel result is marked robust only if it passes both:

```text
q <= 0.10
abs(median_relative_impact_pct) >= 0.001%
```

A replicated result requires the same robust sign in at least two months.

A strict cross-target candidate requires clean replicated evidence in at least two verification targets using the `all` target group. If a target has both replicated beneficial months and replicated detrimental months for the same channel, that target is labeled mixed and is not counted as clean cross-target evidence.

## Command Used

From the repository root:

```bash
uv run --offline --with pandas --with numpy python gnn_model/FSOI/analyze_channel_statistical_strength.py \
  --seasonal-root gnn_model/FSOI/fsoi_outputs/seasonal_sentinel_fixed \
  --out-dir gnn_model/FSOI/fsoi_outputs/channel_statistical_strength_current \
  --targets radiosonde aircraft surface_obs \
  --months jan apr jul oct \
  --block-cycles 10 \
  --n-boot 5000 \
  --fdr-alpha 0.10 \
  --min-relative-effect-pct 0.001
```

## Output Files

| File | Meaning |
|---|---|
| `channel_cycle_impacts.csv` | Cycle-level impacts after aggregating row-level channel impacts within physical target groups. |
| `channel_monthly_bootstrap_stats.csv` | Bootstrap confidence intervals, p-values, q-values, practical effect flags, and robust monthly sign classification. |
| `channel_fdr_results.csv` | FDR-adjusted q-values for each beneficial and detrimental hypothesis. |
| `channel_replication_summary.csv` | Month-replication summary for each target/group/instrument/channel. |
| `channel_cross_target_candidates.csv` | Strict clean cross-target candidates using the `all` target group. |
| `CHANNEL_STATISTICAL_STRENGTH.md` | Auto-generated summary report for the current run. |

## Current Run Size

The current run produced:

| Output | Rows |
|---|---:|
| `channel_cycle_impacts.csv` | 402,768 |
| `channel_monthly_bootstrap_stats.csv` | 7,200 |
| `channel_fdr_results.csv` | 14,400 |
| `channel_replication_summary.csv` | 1,800 |
| `channel_cross_target_candidates.csv` | 75 |

Of the 7,200 monthly channel/group tests, 4,983 pass both the FDR and practical-effect filters:

| Robust direction | Count |
|---|---:|
| Beneficial | 2,740 |
| Detrimental | 2,243 |
| Not robust | 2,217 |

## Current Cross-Target Detrimental Candidates

Under the strict clean-target rule, the following channels have replicated detrimental evidence in at least two target definitions:

| Channel | Clean replicated detrimental targets |
|---|---|
| AMSU-A ch9 | radiosonde; surface_obs |
| AMSU-A ch10 | radiosonde; surface_obs |
| AMSU-A ch14 | radiosonde; surface_obs |
| ATMS ch15 | radiosonde; surface_obs |
| ATMS ch16 | radiosonde; surface_obs |
| ATMS ch20 | aircraft; radiosonde |
| SSMIS ch3 | aircraft; radiosonde |
| SSMIS ch4 | aircraft; surface_obs |
| SSMIS ch7 | radiosonde; surface_obs |
| SSMIS ch9 | radiosonde; surface_obs |
| SSMIS ch21 | radiosonde; surface_obs |
| SSMIS ch22 | radiosonde; surface_obs |

These are stronger candidates than row-fraction-only diagnostics because they pass cycle-level replication, practical effect, FDR correction, and cross-target consistency.

## SSMIS Channel 21 Result

SSMIS ch21 is substantially more defensible after this analysis.

For the `all` target group:

| Target | Robust detrimental months | Robust beneficial months | Mean detrimental-cycle fraction | Median monthly relative impact |
|---|---:|---:|---:|---:|
| Radiosonde | 4/4 | 0/4 | 94.7% | 0.0269% |
| Surface obs | 4/4 | 0/4 | 85.1% | 0.0106% |
| Aircraft | 1/4 | 0/4 | 61.8% | 0.0087% |

Therefore SSMIS ch21 is a strict cross-target detrimental candidate for radiosonde and surface-observation verification, but not for aircraft using the all-group metric.

This strengthens the original channel-level result:

- The old row-fraction result said SSMIS ch21 was often detrimental across variable-pressure rows.
- The new result says SSMIS ch21 is detrimental in independent 12-hour cycles, survives 5-day block-bootstrap uncertainty, passes FDR correction, exceeds a practical effect-size threshold, and replicates in all four months for radiosonde and surface targets.

However, this still does not mean SSMIS ch21 should be removed. The dedicated OSE result showed:

- SSMIS ch21 background replacement reduced radiosonde error in 51/53 July cycles.
- SSMIS ch21 full masking increased radiosonde error in 53/53 July cycles.

So the strongest interpretation is:

```text
SSMIS ch21 has a robust detrimental increment signal for selected targets,
but the channel itself carries useful information.
```

The appropriate next step is targeted correction or downweighting, not blind removal.

## How To Use These Results

Use this analysis to decide which channel interventions deserve OSE or retraining tests.

Good candidates should satisfy most of the following:

- robust monthly sign after FDR correction
- practical effect size above the configured threshold
- replicated in at least two months
- clean replicated sign in at least two targets
- no strong contradiction from full-mask OSE

Do not use these statistics alone as an operational observation rejection rule. They are a stronger screening method for selecting controlled experiments.

## Suggested Next Experiments

Highest-priority follow-up channels from the current strict detrimental list:

1. SSMIS ch21: already tested for July radiosonde with background replacement and full mask; repeat for other months and targets.
2. SSMIS ch22 and ch7: strong radiosonde/surface replicated detrimental candidates.
3. AMSU-A ch14: replicated detrimental for radiosonde and surface targets.
4. ATMS ch15, ch16, and ch20: replicated detrimental candidates that should be compared with the ATMS full-mask result.

For each candidate, run:

- matched background replacement,
- full mask,
- optionally partial downweighting or retraining.

The paired interpretation is important: background replacement tests the increment; full masking tests whether the full channel input is useful.
