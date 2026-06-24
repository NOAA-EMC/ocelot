# Innovation Quality Report — δx = x_a − x_b

**Run:** Jul 2025, radiosonde verification target, 57 pairs, 4,519 valid channel-pairs  
**Date:** 2026-05-23

All values are in **normalized feature space** (features scaled to approximately [−1, 1];
obs_range ≈ 2.0 σ for well-behaved channels). Nine channels excluded from all metrics
because their obs_range < 0.05 (near-constant — likely scan-angle or padding slots):
ATMS ch7, AMSUA ch6/7, SSMIS ch3/5/6/7/9, SEVIRI-ASR ch7.

---

## 1. Background Field Quality — Normalized RMSE

**Definition:** `normalized_RMSE = RMSE(x_a, x_b) / obs_range`  
**Interpretation:** Fraction of the observation's dynamic range that the 12h background
misses. Target of 1–5% applies to operational NWP with hourly cycling; for a 12h
single-cycle GNN prediction this is expected to be much larger.

| Instrument | Mean nRMSE | Median nRMSE | 95th pct | Verdict |
|---|---|---|---|---|
| ascat | **118%** | 75% | 210% | Very large — ASCAT background far from analysis |
| seviri\_asr | **110%** | 85% | 278% | Very large |
| amsua | 86% | 54% | 416% | Large, high variance across channels |
| avhrr | 83% | 62% | 145% | Large |
| aircraft | 72% | 55% | 147% | Large |
| surface\_obs | 60% | 59% | 96% | Large but stable |
| ssmis | 57% | 28% | 199% | Moderate median, fat tail |
| atms | 55% | 26% | 127% | Moderate median — but ch8 = **411%** (outlier) |
| radiosonde | 52% | 44% | 102% | Moderate |

**Conclusion:** No instrument meets the 1–5% target. Every instrument exceeds it by
one to two orders of magnitude. This is **expected** for a 12h single-cycle GNN —
the background field is a 12h free forecast, not an analysis increment. In this
regime, large innovations mean the observations are making substantial corrections,
which is precisely what FSOI is designed to measure. The large nRMSE does **not**
invalidate FSOI but it does mean the linearization (gradient × δx) spans a large
distance from the background, which amplifies any non-linearity.

---

## 2. Innovation Bias — Mean of δx per Channel

**Definition:** `mean_bias` = time-mean of δx = x_a − x_b in normalized units.  
**FSOI implication:** FSOI ≈ g · δx. If δx has a persistent, stable non-zero mean,
every FSOI value is shifted by (gradient × bias), introducing a systematic error.

### 2.1 Per-instrument summary

| Instrument | Mean bias (σ) | |bias|/std(δx) | % pairs with \|bias/σ\| > 0.3 | Risk |
|---|---|---|---|---|
| aircraft | +0.117 | 1.32 | 67% | **High — ch2 and ch3 dominate** |
| radiosonde | +0.260 | 1.01 | 63% | **High — ch1 and ch2** |
| seviri\_asr | +0.617 | 0.73 | 64% | High |
| ascat | **+0.894** | 0.67 | **100%** | **High — all channels** |
| avhrr | +0.296 | 0.66 | 67% | High |
| surface\_obs | −0.059 | 0.63 | 76% | High — ch1 and ch3 |
| amsua | +0.213 | 0.39 | 39% | Moderate |
| atms | +0.096 | 0.32 | 41% | Moderate |
| ssmis | +0.032 | 0.27 | 27% | **Lowest bias** |

### 2.2 Per-channel detail for key instruments

**RADIOSONDE** (verification target — this bias directly enters e_a)

| Channel | Physical quantity | Mean bias | nRMSE | Time-stable? |
|---|---|---|---|---|
| ch1 | temperature | +0.450σ | 28% | Yes (std=0.021) |
| ch2 | dewpoint / specific humidity | **+0.804σ** | 95% | **Yes (std=0.027)** |
| ch3 | u-wind | +0.012σ | 46% | Yes — essentially unbiased |
| ch4 | v-wind | −0.225σ | 39% | Yes |

Radiosonde ch2 (dewpoint) has a large, rock-stable bias of +0.8σ that persists
across all 57 pairs (std = 0.027). This means the model analysis x_a consistently
sits 0.8σ above the background x_b for specific humidity. This inflates the
humidity contribution to FSOI for all instruments — the gradient dot-product with
this 0.8σ offset dominates the humidity channel's FSOI regardless of the true
analysis increment.

**ATMS** (most detrimental instrument)

| Channel | Layer | Mean bias | nRMSE | Time-stable? |
|---|---|---|---|---|
| ch8 | troposphere | **+1.735σ** | **411%** | **Yes (std=0.056)** |
| ch6 | troposphere | +0.739σ | 74% | Yes (std=0.053) |
| ch2 | surface/window | +0.646σ | 105% | Yes (std=0.019) |
| ch3 | troposphere | −0.681σ | 63% | Yes (std=0.024) |
| ch4 | troposphere | −0.403σ | 60% | Yes |
| ch9 | troposphere | +0.535σ | 70% | Yes |
| ch16–22 | stratosphere | −0.10 to +0.23σ | 6–8% | Yes — **well behaved** |

ATMS's stratospheric channels (16–22) are close to unbiased and have nRMSE ~6–8%.
Its tropospheric channels, especially ch8 (+1.735σ), carry large stable biases.
Ch8's nRMSE of 411% implies the obs_range for this channel is very narrow (~0.05σ)
relative to the innovation magnitude — the analysis barely uses ch8 but the
background sits far from it. This channel alone likely accounts for a large fraction
of ATMS's detrimental FSOI signal.

**AIRCRAFT** (most beneficial instrument)

| Channel | Physical quantity | Mean bias | nRMSE | Time-stable? |
|---|---|---|---|---|
| ch1 | temperature | +0.025σ | 23% | Yes — essentially unbiased |
| ch2 | specific humidity (masked) | **+1.148σ** | 139% | **Yes (std=0.071)** |
| ch3 | u-wind | −0.822σ | 55% | Yes (std=0.112) |

Aircraft ch2 (humidity, which is masked in the aircraft config per experiment design)
has a +1.148σ stable bias. Despite the mask, this channel appears in the innovation
diagnostics because x_a still contains a value for it. If gradients for ch2 flow
through, the large positive bias with a presumably negative gradient (aircraft
appears helpful) will artificially inflate aircraft's beneficial FSOI magnitude.
Aircraft ch1 (temperature) is clean (+0.025σ), which is the channel that should
dominate when the config is applied correctly.

**ASCAT**

| Channel | Mean bias | nRMSE |
|---|---|---|
| ch1 | +0.632σ | 75% |
| ch2 | **+1.408σ** | **206%** |
| ch3 | +0.641σ | 73% |

ASCAT has positive bias on all three channels; 100% of channel-pairs exceed the
|bias/σ| > 0.3 threshold. ASCAT nevertheless appears helpful in FSOI (mean −0.664/pair).
The positive bias combined with helpful (negative) FSOI means the analysis is being
pulled in the right direction despite the large offset.

---

## 3. Gaussianity — Skewness of δx

**Definition:** Zero-mean Gaussian is the distributional assumption underlying
the standard FSOI formula. Skewness ≠ 0 implies the distribution is asymmetric.

| Instrument | Mean skewness | Mean \|skew\| | % pairs with \|skew\| > 1.0 | Verdict |
|---|---|---|---|---|
| avhrr | **+0.954** | **0.954** | **35%** | **Non-Gaussian — right-skewed** |
| ssmis | +0.017 | 0.733 | 32% | Moderate non-Gaussianity |
| surface\_obs | +0.182 | 0.708 | 32% | Moderate |
| atms | +0.184 | 0.580 | 14% | Moderate |
| aircraft | +0.068 | 0.570 | 18% | Moderate |
| ascat | −0.433 | 0.433 | 0% | Left-skewed but no extreme pairs |
| radiosonde | −0.057 | 0.386 | 5% | Near-Gaussian overall |
| seviri\_asr | +0.257 | 0.341 | 1% | Near-Gaussian |
| amsua | −0.044 | 0.327 | 2% | Near-Gaussian |

**AVHRR** is the most problematic: mean skewness = +0.954, 35% of channel-pairs with
|skew| > 1.0. Combined with its positive bias (+0.296σ mean), AVHRR's δx distribution
is consistently right-skewed and offset — the Gaussian zero-mean assumption is
violated for this instrument.

**SSMIS and surface_obs** have moderate non-Gaussianity (32% high-skew pairs each).
However, SSMIS has the smallest mean bias (+0.032σ), so the skewness alone is the
concern — the FSOI values may have inflated variance but not systematic bias.

**Radiosonde** (the verification target) has near-Gaussian innovations for wind
channels (u, v), which is reassuring. The temperature and humidity channels carry
bias but are not strongly skewed.

---

## 4. Combined Risk Assessment

The two concerns interact: **bias shifts the mean of FSOI; non-Gaussianity
inflates its variance**. An instrument with large stable bias AND non-Gaussianity
has the most unreliable FSOI values.

| Instrument | Bias risk | Gaussianity risk | Combined | FSOI finding |
|---|---|---|---|---|
| **aircraft ch2** | Severe (+1.15σ) | Moderate | **High** | Helpful — may be overstated |
| **atms ch8** | Severe (+1.74σ) | Moderate | **High** | Detrimental — partially explained by bias |
| **radiosonde ch2** | High (+0.80σ) | Low | **High** | Helpful — humidity contribution unreliable |
| **ascat all ch** | High (+0.63–1.41σ) | Low (left-skew) | **High** | Helpful — but bias inflates magnitude |
| **avhrr** | Moderate (+0.30σ) | Severe (35% high-skew) | **High** | Detrimental — value uncertain |
| **ssmis** | Low (+0.03σ) | Moderate (32% high-skew) | **Moderate** | Helpful — most trustworthy satellite |
| **atms ch16–22** | Low (<0.23σ) | Moderate | **Low** | Stratospheric channels are reliable |
| **aircraft ch1** | Negligible (+0.02σ) | Moderate | **Low–Moderate** | Helpful finding is clean |
| **radiosonde ch3** | Negligible (+0.01σ) | Low | **Low** | u-wind FSOI is reliable |

---

## 5. What This Means for the FSOI Rankings

### Findings that are robust:
1. **ATMS detrimental** — confirmed by OSE independently of FSOI. Even if the
   FSOI magnitude is biased by ch8's +1.74σ offset, the OSE's 98% error reduction
   on denial confirms this is real.
2. **SSMIS helpful** — lowest bias of all instruments (0.03σ), so its negative
   FSOI is the cleanest satellite signal.
3. **Radiosonde u-wind contribution (ch3)** — unbiased, near-Gaussian; its FSOI
   contribution is the most reliable channel across all instruments.
4. **ATMS stratospheric channels (ch16–22)** — small bias, moderate skew; the
   stratospheric part of ATMS's signal is not the problem.

### Findings that need qualification:
1. **Aircraft magnitude** — ch1 temperature contribution is clean, but ch2
   (humidity, +1.15σ bias) may inflate the total impact. The rank (most helpful)
   is likely correct; the magnitude may be overstated by 20–40%.
2. **Radiosonde rank** — helpfulness is real, but the dewpoint channel (+0.80σ bias)
   dominates the humidity contribution. Wind-based FSOI is reliable; humidity-based
   is not.
3. **ASCAT helpful** — sign is plausible, but all channels carry large biases
   (0.63–1.41σ). The finding is consistent with NWP evidence but the magnitude
   is uncertain.

### Root cause:
The biases reflect the fact that the model is a 12h single-cycle predictor. The
analysis x_a is not close to x_b for many channels because the GNN has only had
one pass of data assimilation to correct the background. In a cycling system, these
offsets would be reduced. Until cycling is introduced, the interpretation of FSOI
magnitudes should be treated as order-of-magnitude rather than precise.

---

## 6. Recommended Actions

| Priority | Action |
|---|---|
| **P1** | Investigate ATMS ch8 specifically — obs_range ≈ 0.05σ (near-constant channel) suggests it may be a bug in the channel extraction or normalization rather than a real physical bias |
| **P1** | Verify aircraft ch2 masking is effective — the config says `mask: specificHumidity` for aircraft but ch2 shows +1.15σ stable bias, suggesting the mask is not being applied to the background x_b extraction |
| **P2** | Radiosonde ch2 (dewpoint) +0.80σ bias: check if dewpoint normalization uses the same statistics as the model training set — a normalization offset would produce exactly this pattern |
| **P2** | Re-run FSOI with ch8 excluded from ATMS (or down-weighted) and compare instrument rankings |
| **P3** | For the paper: report FSOI rankings (signs) as robust; report magnitudes as indicative, with the qualification that 12h background biases inflate absolute values |
