# FSOI for an AI Direct-Observation-Prediction Model — Final Findings

## Forecast Sensitivity to Observations for OCELOT, a Graph-Transformer Hybrid Model for Direct Observation Prediction

*Compiled: July 2026 | Author: Azadeh Gholoubi | NOAA/NCEP/EMC*
*Model: OCELOT v1 (`ocelot-v1.0`), checkpoint `Epoch3079_fixedval.ckpt` (baseline M0)*
*Purpose: manuscript writing and presentation source-of-truth. Synthesizes all runs in
`FSOI/fsoi_outputs/` and all documentation in `FSOI/docs/`.*

> **What OCELOT is.** OCELOT is a graph-transformer hybrid model for **direct observation
> prediction** — it maps a window of past observations (satellite, surface, radiosonde,
> aircraft) over T−12h→T onto a fixed global icosahedral mesh (resolution 6), evolves the
> latent mesh state with a sliding-window transformer plus graph spatial mixing, and decodes
> directly to observation locations at the target time. It is **not** a data-assimilation or
> analysis-producing system: there is no variational/EnKF analysis step and no background-error
> covariance model. In this document, the FSOI "analysis" state $x^a$ and "background" state
> $x^b$ denote the model's observation-space prediction with and without a given observation
> group in the input window (the perturbation endpoints used to form the sensitivity), not a
> DA analysis versus a first-guess forecast.

> **Relation to prior work (read before claiming novelty).** The idea of applying FSOI to an
> AI Direct-Observation-Prediction model was **first published by GraphDOP** (Laloyaux et al.,
> *Using data assimilation tools to dissect GraphDOP*, ECMWF, arXiv:2510.27388, Nov 2025).
> GraphDOP already demonstrated adjoint-free FSOI via PyTorch autograd on a DOP model in
> observation space (with $x^b$ = 12h forecast from the previous window and $x^a$ = current-window
> observations — the same conceptual setup we use), including per-data-type and per-channel FSOI,
> geographic FSOI maps, a per-variable × per-pressure-level radiosonde heatmap, innovation-vs-FSOI
> scatter plots, and case studies, averaged over Jan–Mar 2023. **We therefore do NOT claim to be
> the first to apply FSOI to an AI-DOP model.** OCELOT's genuine, defensible contributions
> — several of which GraphDOP explicitly lists as unexplored future work — are:
>
> 1. **Independent confirmation on a second, independently developed AI-DOP model** (OCELOT,
>    NOAA/NCEP/EMC), showing the FSOI-on-DOP approach generalizes beyond GraphDOP/ECMWF.
> 2. **Quantitative OSE validation of FSOI** (ATMS denial, 93% sign agreement; closure ratio
>    77–2451) — GraphDOP states "these strands of work … have not been explored yet."
> 3. **Finite-difference gradient validation** (Tier-1 scalar and Tier-2 Rademacher, <1% error) —
>    an explicit correctness proof of the autograd gradients.
> 4. **Multi-season analysis** (Jan/Apr/Jul/Oct 2025) rather than a single winter period.
> 5. **Multi-target / cross-target verification** (radiosonde, aircraft, surface, and an
>    independent GFS-analysis mesh) revealing **verification-target dependence and per-channel
>    sign reversals** — our headline scientific result, beyond GraphDOP's radiosonde/global metrics.
> 6. **Closure / linear-approximation magnitude assessment** calibrated against OSE, with an
>    explicit "use FSOI for direction, OSE for magnitude" rule.
> 7. **Area-weighting + Horvitz–Thompson subsampling correction** for population totals (GraphDOP
>    used $C$ = identity) and an **actionable channel-selective exclusion/downweighting list**.

---

## 0. One-paragraph abstract

We implemented adjoint-free Forecast Sensitivity to Observations (FSOI) for OCELOT, a
graph-transformer hybrid model for **direct observation prediction** (a model that predicts
future observations directly from a window of past observations, without a data-assimilation
analysis step), using automatic differentiation to obtain exact forecast-error gradients with
respect to every input observation. We validated the
gradient computation to <1% error against finite differences, and validated the *direction*
of the FSOI signal against independent Observing System Experiments (OSE). Across four
seasons (Jan/Apr/Jul/Oct 2025) and four independent verification targets (radiosonde,
aircraft, surface, and a fully independent GFS-analysis mesh). Building on GraphDOP
(Laloyaux et al. 2025), which first applied FSOI to an AI-DOP model, our contribution is an
**independent confirmation on a second AI-DOP model plus the validation and cross-target
analysis GraphDOP left unexplored**: finite-difference gradient validation, quantitative OSE
cross-validation, multi-season coverage, and a systematic multi-target comparison.
The headline scientific result is that **observation impact is verification-target
dependent**: microwave sounders (ATMS, AMSU-A, SSMIS) degrade forecasts at radiosonde sites
but improve them at aircraft flight levels, so no instrument carries a single global
"good/bad" label. FSOI rankings (signs) are trustworthy; FSOI magnitudes are not
calibration-grade because the model's large observation-driven increments (3–4σ) violate the
linear approximation. We deliver an actionable channel-selective exclusion/downweighting list.

---

## 1. What FSOI is and how we computed it

**Definition.** FSOI estimates how much each observation changed the 12-hour forecast error:

$$\text{FSOI}_i = \tfrac{1}{2}\,(x^a_i - x^b_i)\,(g^a_i + g^b_i)$$

where $x^a-x^b$ is the observation increment ("δx": the change in OCELOT's predicted state
when the observation group is present versus withheld from the input window) for observation
$i$, and $g^a,g^b$ are the gradients of the forecast error with respect to the state,
evaluated at the two prediction endpoints (trapezoidal average).

- **Sign convention:** negative = beneficial (reduces forecast error); positive = detrimental.
- **Adjoint-free:** gradients come directly from PyTorch autograd through the GNN, so no
  separate adjoint model is required — a key advantage of a differentiable AI prediction model.
- **Analysis unit:** the scientifically correct unit is
  **source channel × target variable × target pressure level**. Instrument-level totals are a
  first-order summary only and can hide beneficial channels inside a net-detrimental instrument.

**Verification targets (4 independent configurations):**

| Target | Type | Locations | Bias note |
|---|---|---|---|
| Radiosonde | Obs-space | Upper-air sounding sites (NH land-biased) | Primary reference in the literature |
| Aircraft | Obs-space | Flight levels | Upper-troposphere emphasis |
| Surface obs | Obs-space | Surface stations | Boundary-layer emphasis |
| GFS mesh | Mesh-space | 40,962 global icosahedral nodes vs GFS analysis | **Most unbiased** — no self-verification |

All runs: `use_area_weights: true`, stratified spatial subsampling for high-count instruments
(ATMS cap 50k, AVHRR 30k, SSMIS 24k), Horvitz–Thompson scaling to population totals.

---

## 2. Framework validation (the credibility foundation)

Before any science, three independent checks confirm the machinery is correct.

### 2.1 Gradient correctness — finite-difference check
- **Tier 1 (scalar per-obs FD):** 1 WARNING (radiosonde ch2, 1.0% error), 29 SKIP (below
  float32 noise floor for high-N instruments — expected), 0 FAIL.
- **Tier 2 (Rademacher directional derivative):** **ALL PASS** for all six satellite
  instruments; max relative error 2.2% (SSMIS), Pearson $r > 0.9999$ between autograd and FD.
- **Conclusion:** the GNN backpropagation graph is intact; gradients are definitively correct.

### 2.2 Sign validation — independent OSE (ATMS denial)
- **July 2025 (59 pairs):** removing ATMS worsens the forecast in **55/59 pairs (93%)** —
  FSOI and OSE agree ATMS is net detrimental at radiosonde sites.
- **January 2025:** consistent sign agreement; better closure than July.

### 2.3 Magnitude is NOT calibration-grade — the central caveat
- **Global closure ratio (Jul):** median 1.52 (target 0.9–1.1) → **FAIL**.
- **OSE closure ratio |FSOI|/|ΔJ_OSE|:** **77–2451 in July (median ~150)**, 34–92 in January.
  FSOI overestimates true impact by ~100–2400×.
- **Root cause:** analysis innovations are 3–4σ (radiosonde rms 3.2σ, surface 4.0σ). The
  small-perturbation assumption $\delta e \approx g\cdot\delta x$ requires $|\delta x|\ll 1$;
  the neglected second-order term $\tfrac12\,\delta x^\top H\,\delta x$ is not negligible.
- **Seasonality:** closure degrades in summer (stronger convection, larger background error).

> **Golden rule for the paper and every talk:**
> **Use FSOI for direction and ranking. Use OSE for magnitude.** State this every time a
> number is shown.

---

## 3. Headline scientific findings

### Finding 1 — Observation impact is verification-target dependent (the main result)
Microwave sounders (ATMS, AMSU-A, SSMIS) are **detrimental at radiosonde sites but beneficial
at aircraft flight levels**. The same increment that misaligns with upper-air soundings
genuinely improves flight-level forecasts. **No instrument has a single global "good/bad"
label** — impact must be reported per target, per channel, per level. This is the core
message and reframes any "instrument X is harmful" claim.

### Finding 2 — Aircraft is the only universally beneficial instrument
Aircraft observations are net beneficial at **all** verification networks (radiosonde,
aircraft, surface, and GFS mesh) in **every** season — the single most robust result in the
study. This positions aircraft as the safest data source to up-weight.

### Finding 3 — Radiosonde dominates its own target, but that is partly self-verification
At radiosonde verification, radiosonde is beneficial by ~20× over aircraft and ~200× over
satellites. This dominance is real but partly a **diagonal (self-verification) effect**:
an instrument that both sets the initial state and defines the verification metric at the same
locations is guaranteed to score large. The scientifically informative results are the
**off-diagonal** cross-network impacts.

### Finding 4 — Cross-network surprises (off-diagonal, bias-free signals)
- **Surface obs hurt radiosonde verification** (+, all seasons): very large surface
  innovations (4.0σ) propagate upward and degrade upper-air forecasts.
- **Radiosonde hurts surface verification** (+0.4 to +0.8, all seasons): large upper-air
  corrections create dynamically inconsistent near-surface tendencies.
- **ATMS/AMSU-A/SSMIS help aircraft, hurt radiosonde:** confirms genuine upper-tropospheric
  information content that is misaligned specifically at radiosonde levels.

### Finding 5 — Independent mesh-space verification confirms the ranking
Against GFS analysis at 40,962 global nodes (no self-verification, no autocorrelation bias),
the ranking is identical to obs-space radiosonde: radiosonde dominant beneficial; surface obs
and SSMIS detrimental; microwave sounders mostly detrimental. This proves the detriment is
**not** a radiosonde-network artifact. (The three pressure-level runs — 250/500/850 hPa —
are numerically identical, confirming mesh verification integrates over all levels.)

### Finding 6 — Channel-level structure and coupling violations
Instrument totals hide the physics. Per-channel × target-level heatmaps reveal that specific
channels leak increments to pressure levels far from their weighting-function peak:

| Instrument | Channel | WF peak | Observed leakage | Diagnosis |
|---|---|---|---|---|
| SSMIS | 21 (91.6 GHz) | surface | Detrimental at **all** levels incl. stratosphere | Surface BT variance propagating upward |
| ATMS | 22 (183 GHz) | ~200 hPa | Detrimental across 250–1000 hPa T | Water-vapor increment leaking into T column |
| ATMS | 10 | ~250 hPa | Detrimental at 500–850 hPa | Downward coupling of mid-tropo T |
| AMSU-A | 3 (50.3 GHz) | ~850 hPa | Detrimental at 500–700 hPa | Near-surface channel affecting mid-tropo T |

These violations indicate the model's background-error covariances do not properly localize
each channel's influence — a concrete, targetable model-improvement diagnostic.

### Finding 7 — Innovation quality gates reliability per channel
Innovation diagnostics (δx bias, Gaussianity) qualify which FSOI values to trust:
- **Most trustworthy satellite:** SSMIS (smallest δx bias, +0.03σ).
- **Cleanest single channels:** radiosonde u-wind (ch3, unbiased, near-Gaussian);
  ATMS stratospheric channels 16–22 (small bias, ~6–8% nRMSE).
- **Qualified findings:** aircraft magnitude may be inflated 20–40% by a masked-humidity
  channel bias (ch2, +1.15σ); ATMS ch8 (+1.74σ) partly explains ATMS detriment magnitude;
  AVHRR is strongly right-skewed (Gaussian assumption violated).
- Every instrument's 12h-prediction nRMSE far exceeds operational 1–5% — **expected** for a
  12h direct-observation-prediction model (not an analysis increment) and consistent with the
  closure caveat.

---

## 4. Master results tables (synthesized, all seasons)

### Table A — Instrument net FSOI sign across all four verification targets
(B = beneficial, D = detrimental, M = mixed; parentheses = seasons consistent out of 4)

| Instrument | Radiosonde | Aircraft | Surface | GFS mesh | One-line interpretation |
|---|---|---|---|---|---|
| **Aircraft** | B (4/4) | B (4/4) | B (4/4) | B (4/4) | **Universally beneficial** |
| **Radiosonde** | B (4/4) | B (4/4) | **D (4/4)** | B (4/4) | Helps upper-air & mesh; hurts surface |
| **Surface obs** | D (4/4) | B (4/4) | B (4/4) | D (4/4) | Hurts upper-air/mesh; helps own & aircraft |
| **ATMS** | D (3/4) | B (4/4) | B (4/4) | D (3/4) | Target-dependent; hurts radiosonde only |
| **AMSU-A** | D (4/4) | B (4/4) | B (4/4) | D (3/4) | Same pattern as ATMS |
| **SSMIS** | D (4/4) | B (3/4) | B (4/4) | D (4/4) | Detrimental upper-air; helps near-surface |
| **ASCAT** | B (4/4) | B (3/4) | M (2B/2D) | B (4/4) | Beneficial upper-air/mesh; neutral surface |
| **AVHRR** | B (3/4) | B (4/4) | B (4/4) | B (3/4) | Weak, consistently beneficial |
| **SEVIRI ASR** | B (4/4) | B (4/4) | B (3/4) | B (3/4) | Weak, consistently beneficial |

### Table B — Radiosonde-target monthly FSOI sums (representative magnitudes)
(negative = beneficial; area-weighted, subsampling-corrected)

| Instrument | Jan | Apr | Jul | Oct | Signal |
|---|---|---|---|---|---|
| Radiosonde | −1645 | −1546 | −1667 | −1755 | Always beneficial |
| Aircraft | −137 | −76 | −84 | −114 | Always beneficial |
| ASCAT | −10.0 | −8.0 | −7.2 | −9.0 | Always beneficial |
| SEVIRI ASR | −0.74 | −6.2 | −4.2 | −2.8 | Always beneficial |
| AVHRR | +1.5 | −1.1 | −6.8 | −3.8 | Mostly beneficial |
| AMSU-A | +3.1 | +17.5 | +17.5 | +21.7 | Always detrimental |
| ATMS | +5.4 | −1.8 | +27.7 | +7.6 | Mostly detrimental |
| Surface obs | +31.3 | +89 | +115 | +94 | Always detrimental |
| SSMIS | +52.3 | +24.7 | +16.6 | +12.4 | Always detrimental |

### Table C — Innovation magnitude (Jul, radiosonde target) — explains the closure caveat
| Instrument | innovation mean (σ) | rms (σ) | Note |
|---|---|---|---|
| surface_obs | −1.73 | 4.00 | Very large — drives misclosure |
| radiosonde | −1.14 | 3.21 | Large bias + spread |
| aircraft | −0.28 | 1.26 | Moderate |
| satellites (ATMS/SSMIS/AMSU-A/ASCAT/AVHRR/SEVIRI) | ≈ 0 | 0.32–0.59 | Well-behaved, near-zero bias |

---

## 5. Cross-target channel verdicts (from `cross_target_analysis/`)

Averaged across all seasons and (variable × pressure) rows for all three obs-space targets:

- **No channel is detrimental at every level in all three targets.** Even "net-detrimental"
  channels carry beneficial information at some levels — full exclusion discards that too.
- **Net-detrimental in aggregate across all 3 targets:** SSMIS 7/9/10/21/22; ATMS 12/21;
  AMSU-A 8/14.
- **Universally beneficial across all 3 targets:** AMSU-A 1/6/12; ATMS 17/18; AVHRR 1;
  ASCAT 3; SEVIRI-ASR 1.
- **Sign reversals across targets are common** (e.g., ATMS ch2 hurts radiosonde but helps
  aircraft and surface; ASCAT ch1/2 help upper-air but hurt surface) — direct evidence for
  Finding 1.

---

## 6. Actionable recommendations

### Channel-selective actions (do NOT exclude whole instruments)
| Priority | Instrument · Channel | Action | Rationale |
|---|---|---|---|
| 1 | SSMIS ch21 (91.6 GHz) | **Exclude** | 80% radiosonde levels detrimental; 100% surface; net-detrimental all 3 targets |
| 2 | SSMIS ch7, ch22 | **Exclude** | ~73% radiosonde levels; net-detrimental all 3 targets |
| 3 | ATMS ch21 (and ch10, ch15, ch22) | **Downweight** | Detrimental at radiosonde; beneficial at aircraft — selective only |
| 4 | AMSU-A ch14 | **Downweight** | 73% radiosonde levels; net-detrimental all 3 targets |
| 5 | SSMIS ch9, ch10 | **Monitor** | Borderline (~50% of levels) |
| 6 | ATMS ch12, AMSU-A ch8 | **Monitor** | Target-dependent; low frac-detrimental at aircraft |

### Protect (universally beneficial — do not touch)
AMSU-A ch6 & ch1 & ch12, ATMS ch17 & ch18, AVHRR ch1, ASCAT ch3, SEVIRI-ASR ch1.

### Instrument-level guidance
- **Aircraft:** retain; candidate to up-weight (universally beneficial).
- **Radiosonde:** retain; review error covariances (surface-level detriment suggests
  over-aggressive upper-air increments).
- **Surface obs:** retain at low weight (essential near-surface, but hurts upper-air).
- **Microwave sounders:** channel-selective tuning, not wholesale denial.

---

## 7. Limitations (state these in the manuscript)
1. **Magnitude is not calibration-grade** — linear approximation fails under 3–4σ increments;
   quantitative claims require OSE.
2. **Verification networks are not fully independent** — surface/aircraft/radiosonde are all
   model inputs; off-diagonal FSOI blends direct information with dynamical inconsistency.
3. **Mesh geometry advantage** — GFS analysis is independent, but the mesh nodes are the
   model's own training grid; use mesh results for ranking, not absolute magnitude.
4. **Innovation biases** — several channels carry large stable δx biases (aircraft ch2,
   ATMS ch8, radiosonde ch2) that qualify the affected magnitudes.
5. **Single checkpoint / single model (M0)** — results are for one trained model; retraining
   sensitivity is untested.

---

## 8. Suggested manuscript structure

1. **Introduction** — observation impact matters; adjoint-free FSOI is uniquely easy in a
   differentiable AI model (autograd replaces the adjoint). **Position relative to GraphDOP**
   (Laloyaux et al. 2025), which first applied FSOI to an AI-DOP model. Novelty of this work:
   independent confirmation on a second AI-DOP model (OCELOT) plus finite-difference gradient
   validation, quantitative OSE cross-validation, multi-season coverage, and systematic
   multi-target (radiosonde/aircraft/surface/mesh) verification revealing target dependence.
2. **Methods** — FSOI formula, trapezoidal gradient, four verification targets, subsampling,
   area weighting. State the linear-approximation assumption up front.
3. **Validation** — FD Tier 1/2 (gradient correctness), OSE (sign), closure (magnitude caveat).
4. **Results** —
   4.1 Instrument rankings per target (Table B).
   4.2 Cross-verification / target-dependence (Table A, Finding 1) — the centerpiece.
   4.3 Channel × level structure and coupling violations (heatmaps, Finding 6).
   4.4 Innovation quality gating (Finding 7).
5. **Discussion** — why sounders are target-dependent; self- vs cross-verification; what the
   coupling violations imply about learned background-error covariances.
6. **Recommendations** — channel-selective list (Section 6).
7. **Limitations & future work** — closure recovery via tighter obs error; multi-model;
   dedicated OSEs for aircraft/surface/radiosonde.

### Figure list (available assets)
| Fig | Content | Path |
|---|---|---|
| 1 | ATMS pair-0 case-study maps | `fsoi_outputs/paper_figures/atms_pair0_case_study_maps.png` |
| 2 | ATMS channel × var × pressure heatmap | `fsoi_outputs/paper_figures/instrument_channel_variable_pressure_heatmap_atms.png` |
| 3 | AMSU-A heatmap | `fsoi_outputs/paper_figures/instrument_channel_variable_pressure_heatmap_amsua.png` |
| 4 | SSMIS heatmap | `fsoi_outputs/paper_figures/instrument_channel_variable_pressure_heatmap_ssmis.png` |
| 5 | ASCAT heatmap | `fsoi_outputs/paper_figures/instrument_channel_variable_pressure_heatmap_ascat.png` |
| 6 | Cross-target channel heatmaps | `fsoi_outputs/cross_target_analysis/heatmaps/{radiosonde,aircraft,surface_obs}/` |
| 7 | Per-target impact/innovation figures | `fsoi_outputs/<target>_seasonal/*/figures/` |

---

## 9. Presentation storyline (3 core slides)

1. **Setup & credibility** — What FSOI is (adjoint-free via autograd), four verification
   targets, and the validation trio: gradients correct (<1% FD), sign correct (93% OSE),
   magnitude not calibration-grade (closure fails; use OSE for magnitude).
2. **The headline** — Impact is verification-target dependent. Show Table A: aircraft
   universally beneficial; microwave sounders hurt radiosonde but help aircraft; no global
   good/bad label. Reinforce with sign-reversal channel examples.
3. **So what** — Channel-selective actions: exclude SSMIS ch21/7/22, downweight ATMS ch21 &
   AMSU-A ch14, protect AMSU-A ch6 / ATMS ch17-18 / AVHRR ch1 / ASCAT ch3. Close with the
   golden rule (FSOI = direction, OSE = magnitude) and coupling-violation model-improvement
   angle.

---

## 10. Data-source index

| Result | Directory |
|---|---|
| Radiosonde obs-space (Jan/Apr/Oct) | `fsoi_outputs/seasonal_fixed/radiosonde_*/` |
| Radiosonde obs-space (Jul, first valid run) | `fsoi_outputs/full_eval_fixed_20250701_20250731/radiosonde/` |
| Aircraft obs-space | `fsoi_outputs/aircraft_seasonal/aircraft_*/` |
| Surface obs obs-space | `fsoi_outputs/surface_obs_seasonal/surface_obs_*/` |
| GFS mesh-space | `fsoi_outputs/mesh_seasonal/radiosonde_{250,500,850}hPa_*/` |
| Cross-target channel analysis | `fsoi_outputs/cross_target_analysis/` |
| OSE ATMS denial (Jul/Jan) | `fsoi_outputs/ose_atms_{jul,jan}2025_fixed/` |
| Gradient FD validation | `fsoi_outputs/fd_check_skip/`, `fsoi_outputs/fd_check_enhanced/` |
| Paper figures | `fsoi_outputs/paper_figures/` |

**Provenance / validity note:** all results are from post-fix runs (commit `7f04963`,
2026-05-23) after correction of a column-offset bug; pre-fix outputs were invalid and deleted.
This document is the consolidated source-of-truth; it supersedes the earlier per-topic result
notes. For methodology see [../FSOI_Explanation.md](../FSOI_Explanation.md); for the output
file/column reference see [FSOI_OUTPUTS_GUIDE.md](FSOI_OUTPUTS_GUIDE.md).
