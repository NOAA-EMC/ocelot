# Forecast Sensitivity to Observations for an AI-Based Numerical Weather Prediction System

**FSOI Implementation Guide — OCELOT AI Weather Model**

*May 2026 • gnn_model/FSOI/*

> For a reference of the **output files** a run produces (directory layout and per-CSV/plot
> column definitions), see [docs/FSOI_OUTPUTS_GUIDE.md](docs/FSOI_OUTPUTS_GUIDE.md).
> For the **scientific findings and rankings**, see [docs/FSOI_FINAL_FINDINGS.md](docs/FSOI_FINAL_FINDINGS.md).

---

## 1  What is FSOI?

> **FSOI answers: "How much did each observation actually help (or hurt) the forecast?"**
> It uses adjoint gradients to attribute forecast error changes back to individual
> observations — without re-running the model for every possible denial experiment.

In classical Numerical Weather Prediction (NWP), observations from radiosondes, aircraft, satellites, and surface stations are assimilated via variational methods each cycle. In OCELOT's AI-based approach, no explicit assimilation step is performed: the true observations at analysis time serve directly as the analysis state (x_a), while the model's prior forecast from the previous window provides the background (x_b). Not all observations are equally useful — some reduce forecast error, some increase it. Running a separate Observing System Experiment (OSE) for each instrument type is prohibitively expensive. FSOI provides a single-pass, gradient-based shortcut.

---

## 2  The Core Formula

```
FSOI(k)  =  0.5 × δx(k)  ⊙  ( ga(k)  +  gb(k) )
```

| Symbol | Name | Meaning |
|--------|------|---------|
| `k` | Observation index | A single measurement from any sensor |
| `δx(k) = xa(k) − xb(k)` | Innovation | How much the analysis differed from background |
| `ga(k)` | Analysis adjoint | ∂J/∂xa — gradient of forecast error w.r.t. analysis obs |
| `gb(k)` | Background adjoint | ∂J/∂xb — gradient of forecast error w.r.t. background obs |
| `⊙` | Element-wise product | Multiply innovation × average gradient component-wise |

**Sign convention:**

- **Negative:** `FSOI(k) < 0`  →  observation k REDUCED forecast error  (beneficial ✓)
- **Positive:** `FSOI(k) > 0`  →  observation k INCREASED forecast error (detrimental ✗)
- **Sum property:** `Σ FSOI(k) ≈ ΔJ = J(xa) − J(xb)`  (closure property)

---

## 3  Are the Gradients Always Negative?

> **No.** ga(k) and gb(k) can be positive OR negative — their sign is not fixed.
> The sign of FSOI is determined by the PRODUCT of the innovation δx and the gradient sum (ga + gb).
> These two quantities are independent, and either combination is physically possible.

**What does the gradient actually mean?**

`ga(k) = ∂J/∂xa(k)` answers a simple question: "If I nudge observation k upward by a tiny amount ε, does the forecast error J go up or down?"

| Gradient value | Physical meaning | What the model 'wants' |
|----------------|-----------------|------------------------|
| `ga(k) < 0` | Increasing obs k → error DECREASES  (∂J/∂x < 0) | Obs k should be pushed HIGHER |
| `ga(k) > 0` | Increasing obs k → error INCREASES  (∂J/∂x > 0) | Obs k should be pushed LOWER |

The gradient is computed purely by backpropagation through the GNN. It reflects the model's learned sensitivity — how much each observation channel influences the downstream forecast error at the verification targets. Satellite channels often have gradients near zero because the GNN has learned to rely on them weakly; radiosonde channels have large gradients because the model is highly sensitive to them.

**What determines the sign of FSOI?**

`FSOI = 0.5 × δx × (ga + gb)` is the product of two independent quantities. FSOI is beneficial (negative) only when the innovation and the gradient point in OPPOSITE directions — meaning the true observation deviates from the prior forecast in the direction that reduces forecast error.

```
  FSOI sign logic:
  ─────────────────────────────────────────────────────────────────────────
  δx < 0  AND  (ga+gb) > 0  →  FSOI = (−)(+) = NEGATIVE  ✓  BENEFICIAL
    Analysis pulled obs DOWN, gradient says going DOWN reduces error

  δx > 0  AND  (ga+gb) < 0  →  FSOI = (+)(−) = NEGATIVE  ✓  BENEFICIAL
    Analysis pulled obs UP,   gradient says going UP   reduces error

  δx < 0  AND  (ga+gb) < 0  →  FSOI = (−)(−) = POSITIVE  ✗  DETRIMENTAL
    Analysis pulled obs DOWN, gradient says going DOWN makes error WORSE

  δx > 0  AND  (ga+gb) > 0  →  FSOI = (+)(+) = POSITIVE  ✗  DETRIMENTAL
    Analysis pulled obs UP,   gradient says going UP   makes error WORSE
  ─────────────────────────────────────────────────────────────────────────
  Rule: FSOI < 0 ⟺ the true obs deviates from the prior forecast (δx)
        in the direction the gradient says reduces forecast error.
```

**Direction-agreement table:**

|  | δx > 0  (analysis went UP) | δx < 0  (analysis went DOWN) |
|--|----------------------------|-------------------------------|
| **ga+gb > 0** (model wants obs LOWER) | DETRIMENTAL  + | BENEFICIAL  − |
| **ga+gb < 0** (model wants obs HIGHER) | BENEFICIAL  − | DETRIMENTAL  + |

**What if ga = gb? (the equal-gradient case)**

Even when ga and gb are exactly equal (call them both g), FSOI is still fully determined by the product of δx and g:

```
FSOI = 0.5 × δx × (g + g)  =  δx × g
```

Equal gradients simply remove the 0.5 averaging factor. The sign is unchanged — FSOI is still positive or negative depending on whether δx and g agree or disagree in direction. So yes, FSOI can be strongly positive even when ga = gb.

**Numerical illustration — three scenarios with same δx but different gradient signs:**

```
  Shared setup:  xb = 15.0°C,  xa = 12.0°C,  δx = −3.0
  ───────────────────────────────────────────────────────────────────
  Scenario A — both gradients negative (model wants obs HIGHER):
    ga = −0.8,  gb = −0.6,  ga+gb = −1.4
    FSOI = 0.5 × (−3.0) × (−1.4) = +2.1   ✗  DETRIMENTAL
    Why: analysis pulled obs DOWN but gradient says going DOWN is WRONG

  Scenario B — both gradients positive (model wants obs LOWER):
    ga = +0.8,  gb = +0.6,  ga+gb = +1.4
    FSOI = 0.5 × (−3.0) × (+1.4) = −2.1   ✓  BENEFICIAL
    Why: analysis pulled obs DOWN and gradient says going DOWN is RIGHT

  Scenario C — ga = gb (equal gradients, still positive):
    ga = −0.7,  gb = −0.7,  ga+gb = −1.4
    FSOI = 0.5 × (−3.0) × (−1.4) = +2.1   ✗  DETRIMENTAL
    Why: equal gradients change nothing — mismatch direction still detriment
  ───────────────────────────────────────────────────────────────────
  The innovation δx is the SAME in all three cases.
  The gradient SIGN alone flips FSOI from detrimental to beneficial.
```

> **Key takeaway:** The gradient is NOT a quality score — it is a direction vector.
> A large negative gradient on an obs means the model is highly sensitive to it AND
> increasing it would help. Whether the true observation actually deviates from the
> prior forecast in that direction is what FSOI measures.

---

## 4  Simple Worked Example

*Imagine one radiosonde temperature observation entering the system at analysis time.*

```
  Background (prior ML forecast for this window):   xb  =  15.0°C
  True observed value (analysis state):             xa  =  12.0°C
  ─────────────────────────────────────────────────────────────────
  Innovation:   δx  =  xa − xb  =  12.0 − 15.0  =  −3.0

  Adjoint (gradient of forecast error w.r.t. this obs):
    ga  =  −0.8   (in analysis state)
    gb  =  −0.6   (in background state)

  FSOI  =  0.5 × (−3.0) × (−0.8 + −0.6)
        =  0.5 × (−3.0) × (−1.4)
        =  0.5 × 4.2
        =  +2.1   ←  DETRIMENTAL
```

> Even though the true obs is 3°C below the prior forecast (δx = −3), the gradients say the forecast error went up. This means the innovation pointed in the wrong direction for this particular state — the background was closer to what the model needed.

If FSOI had come out −2.1 instead, that observation would have reduced forecast error by 2.1 units — beneficial.

---

## 5  Full Pipeline — Step-by-Step

The pipeline processes pairs of consecutive 12-hour forecast windows (prev_batch, curr_batch) in strict chronological order. For a month of data at 12-hour intervals that gives 60 pairs.

```
═══════════════════════════════════════════════════════════════════
                     FSOI PIPELINE OVERVIEW
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │  INPUT: Sequential data bins  (12-hr windows, no shuffle)  │
  │         e.g. 2025-07-01 00Z → 2025-07-31 12Z  =  60 pairs  │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 0: Setup & Validation                                 │
  │  • Load model checkpoint (frozen weights, no grad updates)  │
  │  • Verify 12-hr sequential spacing between windows          │
  │  • Set requires_grad=True on INPUT observation tensors      │
  │  • Optional FD gradient check (10 samples to verify ∂J/∂x) │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 1: Extract xa and xb                                  │
  │                                                             │
  │  prev_batch ──► model.forward() ──► predicted mesh state    │
  │                                          │                  │
  │                               decode at curr obs locations  │
  │                                          │                  │
  │                                          ▼                  │
  │  xb = background prediction at current obs locations        │
  │  xa = actual current observations  (from curr_batch)        │
  │                                                             │
  │  CRITICAL column offset in input tensor:                    │
  │    obs_features = x_input[:, 7+n_meta : 7+n_meta+n_ch]     │
  │    (cols 0:7 are geo/time features, NOT observations)       │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 2: Compute Forecast Error  J(xa) and J(xb)            │
  │                                                             │
  │  Run model forward from xa → 12-h forecast → compare truth  │
  │  Run model forward from xb → 12-h forecast → compare truth  │
  │                                                             │
  │  J(xa) = MSE(forecast_from_xa, verifying_analysis)          │
  │  J(xb) = MSE(forecast_from_xb, verifying_analysis)          │
  │  ΔJ    = J(xa) − J(xb)                                      │
  │         (negative = observations collectively helped)       │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 3: Compute Adjoints  ga and gb                        │
  │                                                             │
  │  ga = ∂J/∂xa  via autograd.backward()  on J(xa)             │
  │  gb = ∂J/∂xb  via autograd.backward()  on J(xb)             │
  │                                                             │
  │  These are vectors — one gradient value per obs channel k   │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 4: Apply FSOI Formula per observation                 │
  │                                                             │
  │  δx(k)   = xa(k) − xb(k)                                   │
  │  FSOI(k) = 0.5 × δx(k) × ( ga(k) + gb(k) )                │
  │                                                             │
  │  ∑_k FSOI(k) ≈ ΔJ   (closure property)                     │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 5: Aggregate Results                                  │
  │                                                             │
  │  By instrument : sum FSOI over all obs from same sensor     │
  │  By channel    : sum FSOI per spectral / variable channel   │
  │  By pressure   : sum FSOI per 50 hPa pressure bin           │
  │  By region     : sum FSOI per lat/lon box                   │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 6: Evaluation & Diagnostics                           │
  │                                                             │
  │  Closure check    : ∑FSOI / ΔJ  (ideal = 1.0)              │
  │  Helpful fraction : % obs with FSOI < 0                     │
  │  Innovation RMS   : ‖δx‖ in σ units (linearity check)      │
  │  FD validation    : finite-difference check of gradients    │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  STEP 7 (Optional): OSE Validation                         │
  │                                                             │
  │  Deny one instrument: replace xa[inst] → xb[inst]           │
  │  Re-run forecast → measure actual ΔJ_ose                    │
  │  Compare sign: does FSOI predict same direction as OSE?     │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  OUTPUTS                                                    │
  │  fsoi_by_instrument.csv  ← headline rankings                │
  │  fsoi_by_channel.csv     ← per-channel detail               │
  │  fsoi_closure_summary.csv← linear approx quality           │
  │  fsoi_system_health.csv  ← helpful_fraction, flags          │
  │  figures/*.png           ← impact bars, maps, profiles      │
  └─────────────────────────────────────────────────────────────┘
```

---

## 6  Code Module Map

Each step in the pipeline is handled by a dedicated Python module:

| Module | Pipeline Step | Responsibility |
|--------|--------------|----------------|
| `fsoi_dataset.py` | Step 0 | Build chronological (prev, curr) pairs; verify 12-hr spacing |
| `fsoi_model_extensions.py` | Step 1 | Decode background xb at observation locations; freeze model |
| `fsoi_utils.py` | Steps 1–5 | Extract obs columns, compute δx, ga, gb, FSOI, aggregate |
| `fsoi_validation.py` | Step 0+6 | FD gradient check, alignment verification |
| `fsoi_inference.py` | Orchestrator | Main entry point — runs all steps, writes CSVs |
| `evaluate_fsoi_results.py` | Step 6 | Closure, helpful fraction, innovation diagnostics |
| `fsoi_ose.py` | Step 7 | OSE denial experiments, compare vs FSOI |
| `visualize_fsoi.py` | Post-run | Impact bar charts, time series, scatter plots |
| `compute_fsoi_weights.py` | Post-run | Convert FSOI rankings → training loss weights |

---

## 7  Critical Bug Fixed: Column Offset

> ⚠️ **All FSOI outputs before commit 7f04963 are INVALID due to this bug. Re-run results from that commit onwards.**

The input tensor to the model has this layout:

```
  Input tensor  x_input  layout:
  ──────────────────────────────────────────────────────────────
  Columns  0 : 7          geographic / time features
                          (latitude, longitude, timestamp, ...)
  Columns  7 : 7+M        instrument metadata
                          (scan angles, satellite geometry, ...)
  Columns  7+M : 7+M+C    ACTUAL OBSERVATION VALUES  ← want this
  ──────────────────────────────────────────────────────────────

  BEFORE FIX:  obs = x_input[:, 0 : C]        # reading lat/lon as obs!
  AFTER FIX:   obs = x_input[:, 7+M : 7+M+C]  # correct offset
```

Because `FSOI(k) = 0.5 × δx(k) × (ga + gb)`, and `δx = xa − xb`, extracting the wrong columns meant δx was computed on geographic features rather than actual measurement values. The innovation was meaningless, making all computed FSOI values invalid.

---

## 8  Innovation RMS — What It Means and Why It Matters

The innovation for one observation is `δx = xa − xb` (observed value minus background prediction). Innovation RMS is the root-mean-square of all δx values for an instrument, expressed in units of that instrument's observational standard deviation σ.

```
  σ  =  the 'typical' spread of that observation type
         (standard deviation of the feature, from feature_stats in observation_config.yaml)

  Innovation RMS in σ units  =  RMS(δx) / σ

  Interpretation:
    1.0σ  →  typical innovation equals the natural variability  (borderline OK)
    0.5σ  →  small, well-behaved: model background is close to obs  ✓
    3.0σ  →  very large: background is 3× a typical obs spread away  ✗
    4.0σ  →  extreme: comparable to the tail of a normal distribution
             (4σ event has probability 0.003% under Gaussian assumption)
```

**Concrete example — radiosonde temperature:**

```
  From observation_config.yaml → feature_stats → radiosonde:
    airTemperature:  mean = −32.49°C,  σ = 28.82°C

  One radiosonde sounding at a given location and time:
    xb (background prediction) =  −10.0°C
    xa (analysed / observed)   =  −27.5°C
    δx = xa − xb               =  −17.5°C

  Innovation in σ units:
    |δx| / σ  =  17.5 / 28.82  =  0.61σ  ← this single ob is fine

  But averaged over thousands of radiosondes in Jul 2025:
    RMS(δx) / σ  =  3.21σ  ← the background is systematically far from obs

  That means: on average, the model's background is 3.21 × 28.82 ≈ 93°C
  of RMS error away from radiosonde measurements.  (Very large.)
```

**Why Innovation RMS matters for FSOI:**

FSOI is derived from a linear (first-order) approximation. It assumes the innovation δx is small enough that the forecast error surface is approximately flat between xb and xa. When δx is large, the surface curves — second-order terms become significant and the linear formula gives wrong magnitudes.

| Instrument | Innov. RMS | Status | Implication for FSOI |
|-----------|-----------|--------|----------------------|
| satellites | 0.3–0.6σ | ✓ Well-behaved | Linear approx valid, magnitudes reliable |
| aircraft | 1.26σ | ~ Moderate | Some nonlinearity, signs reliable |
| radiosonde | 3.21σ | ✗ Very large | Magnitudes approximate, signs still valid |
| surface_obs | 4.01σ | ✗ Extreme | Nonlinear — ranking direction valid only |

> **Why the closure ratio is not exactly 1.0**
> FSOI is a linear approximation. When innovations are 3–4σ, nonlinear terms matter and the linear formula over-estimates the true error change.
> The SIGN direction is still trustworthy. MAGNITUDES are approximate — treat them as relative rankings, not absolute values.

---

## 9  Validation Framework

FSOI relies on correct gradient computation through the model. Three tiers of validation are implemented to catch broken computation graphs:

```
  Tier 1 — Scalar FD check (float32)
  ─────────────────────────────────────────────────────────────
  For 10 sampled observations:
    Perturb obs channel by ε = 1e-3
    FD gradient = (J(x+ε) − J(x−ε)) / (2ε)
    Compare vs autograd gradient
    PASS if relative error < 1%  |  WARNING if < 5%  |  FAIL if > 5%

  Tier 2 — Directional derivative (Rademacher vector)
  ─────────────────────────────────────────────────────────────
  Random direction v ∈ {±1}^n (one sample covers all channels)
    FD: (J(x + εv) − J(x − εv)) / (2ε)
    Adjoint: gᵀv
  Efficient for high-dimensional satellite instruments

  Tier 3 — Float64 FD check
  ─────────────────────────────────────────────────────────────
  Same as Tier 1 but promotes to float64 for precision
  Resolves WARNING→PASS cases caused by float32 rounding
```

### 9.1  Deep Dive: Tier 2 Directional Derivative

> **Core question:** Did autograd compute the correct gradient — across ALL channels simultaneously — in just 2 forward passes?

**The Problem with Tier 1 at Scale**

Tier 1 perturbs one channel at a time, so it needs 2 × n_channels forward passes per instrument. For ATMS (22 channels × thousands of obs nodes) that is 44 separate model runs. Tier 2 collapses this to exactly 2 runs regardless of dimension.

| Instrument | Channels | Tier 1 runs vs Tier 2 runs |
|-----------|---------|--------------------------|
| atms | 22 | 44 runs  vs  2 runs |
| amsua | 15 | 30 runs  vs  2 runs |
| radiosonde | 4 | 8 runs  vs  2 runs |

**What is a Rademacher Vector?**

`v ∈ {±1}ⁿ` means every entry is randomly either +1 or −1. It is a random direction in n-dimensional space where every coordinate has equal magnitude — no channel is louder than another.

```
  Example — ATMS (22 channels, simplified to 5 here):

    x = [ 0.5,  1.2, −0.3,  0.8,  0.1 ]   ← current obs values
    v = [  +1,   −1,   +1,   +1,   −1 ]   ← random Rademacher draw
    ε = 0.01

    x + εv = [ 0.51, 1.19, −0.29, 0.81,  0.09 ]
    x − εv = [ 0.49, 1.21, −0.31, 0.79,  0.11 ]

    Run model twice to get:
      J(x + εv)  and  J(x − εv)

    FD  =  ( J(x + εv) − J(x − εv) ) / (2ε)
        =  directional derivative of J along v  (numerical)
```

**What Does the Gradient Predict?**

The adjoint `g = ∂J/∂x` (computed by `autograd.backward()`) predicts the same directional derivative via a simple dot product:

```
Adjoint prediction  =  gᵀv  =  Σᵢ  g[i] × v[i]
```

If autograd is correct, the chain rule guarantees `FD ≈ gᵀv`. If they disagree, autograd lost the gradient somewhere in the computation graph.

**Why ±1 Specifically — Not Random Gaussians?**

| Direction type | Property | Effect on test |
|---------------|---------|---------------|
| Gaussian v ~ N(0,1) | Large entries dominate dot product | Channels with small gradient are tested weakly — bugs can hide |
| Rademacher v ∈ {±1}ⁿ | Every entry has magnitude exactly 1 | All channels tested with equal weight — minimum estimator variance |

**Why Tier 1 Fails for Satellites (and Tier 2 Saves It)**

Satellite brightness temperature innovations are 0.3–0.6σ — very small values. When you perturb a single channel by ε = 1e-3 in float32, the resulting change J(x+ε) − J(x−ε) can be smaller than float32 rounding noise (≈ 1e-7). The FD estimate is just noise — hence the 29 SKIP results.

Tier 2 perturbs ALL N×C elements simultaneously with the Rademacher vector. The total signal is the sum of N×C contributions — orders of magnitude larger — so it clears the float32 noise floor even for satellites.

> - **Tier 1** = check one gradient component carefully (precision, but expensive + skips for small obs)
> - **Tier 2** = check all components together cheaply (catches broken autograd graphs reliably)
> - **Tier 3** = repeat Tier 1 in float64 to resolve float32 rounding (turns SKIP → PASS or FAIL)

---

## 10  OSE vs FSOI — Validating Rankings

> **OSE is the ground-truth experiment. FSOI is the cheap approximation.**
> Comparing them tells you whether the FSOI rankings can be trusted.

**What is an OSE?**

An Observing System Experiment denies one instrument entirely — replacing its true observations (xa) with the model background (xb) — and re-runs the forecast. The change in forecast error compared to the full-obs run directly measures that instrument's real-world impact — no approximations involved.

```
  ┌───────────────────────────────────────────────────────────────┐
  │  FULL RUN (baseline — used by both FSOI and OSE)             │
  │  All true observations used → forecast → error J_full         │
  └───────────────────────────────────────────────────────────────┘
                    │                         │
                    ▼ (FSOI path)             ▼ (OSE path)
  ┌─────────────────────────────┐  ┌──────────────────────────────┐
  │  Backpropagate gradient     │  │  Deny instrument i           │
  │  through the model          │  │  xa[i] → xb[i]  (replace     │
  │                             │  │  true obs with background)   │
  │  Compute:                   │  │                              │
  │  FSOI(k) = ½δx·(ga+gb)      │  │  Re-run forecast             │
  │  Sum over instrument i      │  │  → error J_denied            │
  │                             │  │                              │
  │  ΔJ_fsoi = Σ_k FSOI(k)      │  │  ΔJ_ose = J_denied − J_full  │
  │  Cost: 1 fwd + 1 bwd        │  │  Cost: 1 extra forward pass  │
  └─────────────────────────────┘  └──────────────────────────────┘
                    │                         │
                    └──────────┬──────────────┘
                               ▼
              Compare: sign(ΔJ_fsoi)  vs  sign(ΔJ_ose)
              Ratio:   ΔJ_fsoi / ΔJ_ose  (ideal = 1.0)
```

| Question | OSE answer | FSOI answer |
|----------|-----------|------------|
| Is this instrument beneficial or harmful? | ✓ Definitive ground truth | ✓ Reliable (sign) |
| How much does it help/hurt (magnitude)? | ✓ Exact measurement | ~ Approximate (linear approx) |
| Which individual obs within instrument matter? | ✗ Cannot — only tests whole instrument | ✓ Per-observation FSOI(k) available |

---

## 11  FSOI Weights for Fine-Tuning

The FSOI impact scores are converted into loss weights for the next round of model training. Two variants are computed:

- **Variant A:** `w ∝ |mean_impact|` — weight proportional to absolute average FSOI magnitude
- **Variant B:** `w ∝ |mean_impact| × reliability²` — down-weights instruments with noisy innovations

---

## 12  How to Run FSOI

Three-step canonical usage:

```bash
# Step 1 — Quick sanity test (6 unit tests, ~2 min)
python FSOI/test_fsoi.py --checkpoint /path/to/model.ckpt

# Step 2 — Run FSOI computation
python FSOI/fsoi_inference.py \
    --checkpoint  /path/to/model.ckpt \
    --config       FSOI/configs/fsoi_config.yaml \
    --start_date  2025-06-01 \
    --end_date    2025-07-01 \
    --output_dir  ./FSOI/fsoi_outputs/my_run

# Step 3 — Visualize
python FSOI/visualize_fsoi.py \
    --input  ./FSOI/fsoi_outputs/my_run/csv \
    --output ./FSOI/fsoi_outputs/my_run/figures
```

**Output directory structure per run:**

```
  <output_dir>/
    csv/
      fsoi_by_instrument.csv   ← main results (per pair, per instrument)
      fsoi_by_channel.csv      ← per-channel breakdown
      fsoi_summary.csv         ← aggregated across all pairs
      scatter_samples.csv      ← innovation vs FSOI samples (for maps)
    evaluation/
      fsoi_closure_summary.csv         ← linear approximation quality
      fsoi_system_health.csv           ← helpful_fraction, system_flag
      innovation_diagnostics.csv       ← δx statistics per instrument
      fd_validation.csv                ← FD gradient check results
      fsoi_beneficial_fraction.csv     ← helpful vs harmful obs breakdown
      fsoi_regional_summary.csv        ← geographic breakdown
      fsoi_closure_per_level_summary.csv ← vertical structure
    figures/
      instrument_impacts.png
      innovation/   ← histograms, time series, heatmaps
      maps/         ← global FSOI spatial maps
    logs/
      fsoi_config_used.yaml   ← config snapshot for reproducibility
```
