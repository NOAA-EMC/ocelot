# FSOI Project — Current State (updated 2026-05-26)

## Environment Setup

```bash
CONDA_BASE="/scratch3/NCEPDEV/da/Azadeh.Gholoubi/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh" && conda activate gnn-env

cd /scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/FSOI_test/ocelot/gnn_model
git checkout feature/FSOI_evaluation
```

Key paths:
```
Repo root    : /scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/FSOI_test/ocelot/gnn_model
Checkpoint   : /scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt
Data         : /scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7
GFS analysis : /scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25
FSOI outputs : FSOI/fsoi_outputs/
FSOI weights : FSOI/fsoi_weights/
```

---

## ⚠️ Column-offset bug fixed (2026-05-23, commit 7f04963)

All output directories produced before commit `7f04963` were deleted. See
`CHANGES_SESSION_2026_05_22.md` Section 0 for full details. Every result below
comes from the corrected code.

---

## Key Notation

| Symbol | Meaning |
|---|---|
| **xa** | The real observation values at the current time — the verified/analysis state (truth proxy) |
| **xb** | The GNN's background forecast at the current time, produced without seeing current observations |
| **xa − xb** | Analysis increment: how much reality differed from the GNN's prior forecast |
| **ga, gb** | Gradients of forecast error with respect to xa and xb (computed via autograd) |
| **FSOI_i** | `0.5 × (xa_i − xb_i) · (ga_i + gb_i)` — estimated impact of observation i on forecast error |
| **e(xa)** | Forecast error evaluated at the locations/variables of xa |

Note: xa is **not** a GNN output. It is the observed/analysis state used as ground truth. xb is what the GNN predicted. Their difference `xa − xb` is the information content that observations add over the model's prior.

---

## All 17 Fixed Jobs Complete (2026-05-24)

| Category | Subdirectory | Seasons/Levels | Status |
|---|---|---|---|
| FD check | `fd_check_skip/` | — | COMPLETE |
| Obs-space (Jul 2025) | `full_eval_fixed_20250701_20250731/radiosonde/` | Jul 2025 | COMPLETE |
| Obs-space seasonal | `seasonal_fixed/radiosonde_jan2025/` | Jan 2025 | COMPLETE |
| Obs-space seasonal | `seasonal_fixed/radiosonde_apr2025/` | Apr 2025 | COMPLETE |
| Obs-space seasonal | `seasonal_fixed/radiosonde_oct2025/` | Oct 2025 | COMPLETE |
| Mesh-space 250 hPa | `mesh_seasonal/radiosonde_250hPa_{jan,apr,jul,oct}2025/` | 4 seasons | COMPLETE |
| Mesh-space 500 hPa | `mesh_seasonal/radiosonde_500hPa_{jan,apr,jul,oct}2025/` | 4 seasons | COMPLETE |
| Mesh-space 850 hPa | `mesh_seasonal/radiosonde_850hPa_{jan,apr,jul,oct}2025/` | 4 seasons | COMPLETE |
| FD check Tier 2 (Rademacher) | `fd_check_enhanced/` | Jul 2025 (3 days) | COMPLETE — all 6 instruments PASS |
| OSE ATMS Jul 2025 | `ose_atms_jul2025_fixed/` | Jul 2025 | **COMPLETE** — 59 pairs, 93% sign agree |
| OSE ATMS Jan 2025 | `ose_atms_jan2025_fixed/` | Jan 2025 | **COMPLETE** |

**All FSOI runs complete as of 2026-05-26.** Weights computed, OSE validated, Tier 2 gradient check passed. →
`FSOI/fsoi_weights/` and `FSOI/fsoi_weights_mesh/` (see section below).

---

## FSOI Output Directory Map

```
FSOI/fsoi_outputs/
├── fd_check_skip/
│   └── evaluation/fd_validation.csv      ← Tier 1: 1 WARN, 29 SKIP (radiosonde WARN, satellites SKIP)
│
├── fd_check_enhanced/
│   ├── evaluation/fd_validation.csv          ← Tier 1 (same pattern)
│   ├── evaluation/fd_directional_validation.csv  ← Tier 2 ALL PASS (6 instruments, Pearson r>0.9999)
│   └── csv/fsoi_summary.csv                  ← FSOI rankings for 3-day test run
│
├── full_eval_fixed_20250701_20250731/
│   └── radiosonde/
│       ├── csv/             ← fsoi_by_instrument.csv (60 pairs)
│       ├── evaluation/      ← closure_ratio 1.524 (FAIL), helpful_fraction 82.2%
│       └── figures/
│
├── seasonal_fixed/
│   ├── radiosonde_jan2025/csv/            ← 58 pairs
│   ├── radiosonde_apr2025/csv/            ← 56 pairs
│   └── radiosonde_oct2025/csv/            ← 60 pairs
│
├── mesh_seasonal/
│   ├── radiosonde_250hPa_{jan,apr,jul,oct}2025/
│   ├── radiosonde_500hPa_{jan,apr,jul,oct}2025/
│   └── radiosonde_850hPa_{jan,apr,jul,oct}2025/
│
├── ose_atms_jul2025_fixed/
│   └── evaluation/
│       ├── ose_results.csv                ← 59 pairs, 55 detrimental (93%), tiny impact (0.001–0.08%)
│       └── ose_vs_fsoi_comparison.csv     ← closure ratio 77–2451 (median ~150), sign agree 93%
└── ose_atms_jan2025_fixed/
    └── evaluation/
        ├── ose_results.csv
        └── ose_vs_fsoi_comparison.csv     ← closure ratio 34–92 (better than July)
```

---

## FSOI Weights

Two weight sets were computed. The mesh-space set is the primary recommendation.

### ⚠️ Obs-space weights (radiosonde-target only) — biased, do not use for training

`FSOI/fsoi_weights/` — **input was radiosonde-only verification targets.**
All 4 seasonal obs-space runs used `target_instruments: ["radiosonde"]`, meaning
e(xa) was computed only at ~3,000 radiosonde sites per pair (NH land-concentrated).
This systematically inflates radiosonde's own weight and suppresses satellites that
improve the ocean/tropics/SH atmosphere. These weights are retained for comparison
but should not be used for fine-tuning.

| Instrument | Weight A | Weight B | pos_frac |
|---|---|---|---|
| radiosonde | 8.178 | 8.179 | 0.458 |
| aircraft | 0.153 | 0.146 | 0.470 |
| surface_obs | 0.106 | 0.111 | 0.445 |
| all satellites | 0.094 | 0.094 | 0.21–0.28 |

---

### ✓ Mesh-space weights (GFS global analysis) — primary, recommended for training

`FSOI/fsoi_weights_mesh/` — **input: 12 mesh jobs, 3 pressure levels × 4 seasons.**  
Verification against GFS analysis at 40,962 global icosahedral mesh nodes.
No geographic concentration bias. Architecturally natural for a GNN that predicts
onto the mesh. 6,168 rows × 60 pairs from 12 files.

| Instrument | n_pairs | mean_impact | positive_frac | reliability | **Weight A** | **Weight B** |
|---|---|---|---|---|---|---|
| **radiosonde** | 60 | −3.71e−05 | 0.460 | 0.291 | **8.166** | **8.180** |
| aircraft | 60 | −7.01e−07 | 0.466 | 0.285 | 0.154 | 0.151 |
| surface_obs | 60 | +5.30e−07 | 0.489 | 0.261 | 0.117 | 0.105 |
| amsua | 59 | +1.83e−08 | 0.399 | 0.362 | 0.094 | 0.094 |
| ascat | 59 | −2.30e−08 | 0.394 | 0.367 | 0.094 | 0.094 |
| avhrr | 60 | −5.00e−10 | 0.351 | 0.421 | 0.094 | 0.094 |
| atms | 59 | +9.61e−10 | 0.403 | 0.356 | 0.094 | 0.094 |
| seviri_asr | 59 | −3.13e−09 | 0.369 | 0.398 | 0.094 | 0.094 |
| ssmis | 59 | +7.27e−09 | 0.415 | 0.342 | 0.094 | 0.094 |

**Key findings:**
- Weights are nearly identical to the biased obs-space set. Radiosonde dominance
  is real, not an artifact of geographic concentration bias. The verification
  network does not change the ranking.
- Satellite positive_frac is markedly higher in mesh-space (0.35–0.42) than in
  obs-space (0.21–0.28) — satellites appear more detrimental when verified globally
  than near radiosonde sites. This confirms the bias existed in the obs-space
  weights, even if it didn't shift the floor instruments off the minimum.
- Radiosonde 50× dominant regardless of verification approach: its 3–4σ innovation
  amplitude is genuinely large relative to all other instruments in this GNN.
- Channel weights: mesh jobs ran with `stratify_by_variable: false` → no
  per-variable channel weights available from this set. Use the obs-space channel
  weights (`FSOI/fsoi_weights/observation_config_channel_weighted.yaml`) if needed.

**Output files (primary, use these for fine-tuning):**
```
FSOI/fsoi_weights_mesh/
├── fsoi_weight_summary.csv          ← full weight table
├── observation_config_variant_a.yaml  ← Variant A: w ∝ |mean_impact|
└── observation_config_variant_b.yaml  ← Variant B: w ∝ |mean_impact| × reliability²
```

---

## Key Scientific Findings (Jul 2025 obs-space, area-weighted, corrected)

From `full_eval_fixed_20250701_20250731/radiosonde/` (60 pairs, Jul 2025):

| Instrument | sum_impact_scaled | Rank | Interpretation |
|---|---|---|---|
| radiosonde | −1667 | 1 | Strongly beneficial |
| aircraft | −84 | 2 | Beneficial |
| ascat | −7.2 | 3 | Beneficial |
| avhrr | −6.8 | 4 | Slightly beneficial |
| seviri_asr | −4.2 | 5 | Slightly beneficial |
| surface_obs | +115 | — | Net detrimental |
| atms | +27.7 | — | Net detrimental |
| amsua | +17.5 | — | Net detrimental |
| ssmis | +16.6 | — | Net detrimental |

Diagnostics:
- helpful_fraction = 82.2% (above 80% target — radiosonde dominance drives this)
- closure_ratio = 1.524 (FAIL) — large δx (3–4σ) violates linearization. Aggregate sign is correct; magnitudes should be treated as relative, not absolute.
- radiosonde innovation_rms = 3.21σ; surface_obs = 4.01σ
- FD Tier 1: 1 WARNING (radiosonde ch2, 1% error), 29 SKIP (all satellites — float32 noise floor)
- FD Tier 2 (Rademacher): ALL PASS — 6 instruments, max rel_error 2.2%, Pearson r > 0.9999. **Gradient computation confirmed correct.**

OSE ATMS validation (Jul 2025, 59 pairs):
- Sign: 55/59 (93%) agreement — ATMS is genuinely net detrimental (confirmed)
- Magnitude: closure ratio 77–2451 (median ~150) — FSOI overpredicts by 100–2400×
- Jan 2025 OSE: closure ratio 34–92 (better, less summer nonlinearity)

---

## Next Steps — FSOI-Weighted Fine-Tuning

All FSOI runs and weight computations are complete. Use mesh-space weights (primary).

```bash
CHECKPOINT="/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/main_PR/ocelot/gnn_model/checkpoints/PR_Test/Epoch3079_fixedval.ckpt"

# Variant A: weight ∝ |mean_impact|  (mesh-space, unbiased)
python train_gnn.py \
    --obs_config FSOI/fsoi_weights_mesh/observation_config_variant_a.yaml \
    --checkpoint "$CHECKPOINT" \
    --max_epochs 500 --lr 1e-5

# Variant B: weight ∝ |mean_impact| × reliability²  (mesh-space, unbiased)
python train_gnn.py \
    --obs_config FSOI/fsoi_weights_mesh/observation_config_variant_b.yaml \
    --checkpoint "$CHECKPOINT" \
    --max_epochs 500 --lr 1e-5

# Evaluate: compare M0 (baseline) vs M1 (Variant A) vs M2 (Variant B)
python evaluation/run_pred_eval_gfs.py --checkpoint <finetuned.ckpt>
```

---

## Paper Status

| Section | Status |
|---|---|
| Methods: FSOI formulation | Ready to write |
| Methods: OSE design | Ready to write |
| Results: Jul 2025 obs-space rankings | COMPLETE — numbers available |
| Results: 4-season obs-space (seasonal) | COMPLETE — Jan/Apr/Jul/Oct 2025 |
| Results: Mesh-space vs GFS rankings | COMPLETE — 3 levels × 4 seasons |
| Results: OSE validation of ATMS | **COMPLETE** — 93% sign agree, closure ratio 77–2451 |
| Results: FSOI-weighted training | NOT STARTED — weights ready, train next |
| Results: FSOI-weighted RMSE vs baseline | NOT STARTED |

---

## Git State

```
Branch: feature/FSOI_evaluation
Recent commits: 939a0a3  bug fix
                22bbe00  Add enhanced gradient validation: doc, config, and run script
                3fe3e2d  Add directional-derivative and float64 FD gradient tests for satellite instruments
                21e6d63  Fix OSE scale/sign bugs, improve per-level closure, add weights and output guide
                bee60d7  Add OSE scripts, fix mesh config epsilon, document first valid FSOI results
```
