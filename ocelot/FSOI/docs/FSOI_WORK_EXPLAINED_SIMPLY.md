# The OCELOT FSOI study, end to end, in plain terms

This note explains what was done, why, what came out, and which problems were
found and fixed along the way. Numbers match the current manuscript draft
(19 September 2026) unless a section says otherwise.

---

## 1. The question

OCELOT is a neural network that forecasts **future observations** directly from
**past observations** (radiosondes, aircraft, surface stations and six satellite
instruments). It is trained with equal loss weights for every observation type.

The goal of the study: **measure how much each observation type actually helps the
forecast**, in a way that is consistent across seasons and instruments, so that
those loss weights could eventually be set on evidence rather than equally.

The tool is **FSOI** (Forecast Sensitivity to Observation Impact): a way to say
"this observation lowered the forecast error by this much".

---

## 2. What FSOI does, with a toy example

Imagine one forecast cycle.

- Run the model with the **real observations** x_c. Its forecast error is **J = 10.0**.
- Run it with each observation replaced by what the model **already expected** from the
  previous forecast (the "background", x_0). Error is **J = 12.0**.
- So the real observations lowered the error by **ΔJ = −2.0**. That is the *realized* change.

Running the model twice per observation would be far too expensive for millions of
observations. FSOI instead uses **gradients**: how sensitive J is to each input value.
Multiplying each observation's surprise (observed minus expected, the
**innovation** d) by the average of the gradients at the two ends gives that
observation's share of the change:

    I_i = ½ · d_i · (gradient at x_c + gradient at x_0)

Adding all the shares gives the **estimate**, say **I = −1.94**.

- **Closure ratio** = estimate / realized = −1.94 / −2.0 = **0.97**. Close to 1 means
  FSOI reproduces what really happened.
- **Sign agreement**: both negative, so FSOI got the direction right.

Negative = beneficial (error went down). Positive = detrimental.

**Why not exactly 1?** The two-end formula is the trapezoid rule for an integral.
If the gradient changes a lot between the two ends, the trapezoid misses some of it.
Checking more points in between (**path integration**) tests that explanation.

---

## 3. The ruler: a fixed verification metric

FSOI needs a definition of "forecast error" (J). If that definition changes from
cycle to cycle, impacts are not comparable. So the first step was to **freeze the
ruler before looking at any impacts**.

- Three separate rulers (**verification networks**): errors are measured against future
  **radiosonde**, **aircraft** or **surface** observations.
- The globe is split into **648 equal-area cells**. Errors are averaged within a cell,
  then across cells, then across pressure levels, then across variables. This stops
  dense regions (e.g. Europe) from dominating just because they have more reports.
- A **coverage audit** (no model runs) decided which variable–level groups have enough
  data: a cell needs ≥ 2 observations; a group needs ≥ 3 such cells and ≥ 10
  observations in them; a group is kept if it passes in ≥ 90% of cycles.
  - Radiosonde: 55 groups (T, u, v at 16 levels; dewpoint only 1000–300 hPa, because
    humidity above 300 hPa is never quality-checked in the source data).
  - Aircraft: 27 groups (1000–200 hPa; above that there are too few reports).
  - Surface: 5 variables.
- If any required group is missing in a cycle, **the whole cycle is skipped** rather
  than scored with a smaller ruler.
- The code fingerprints the targets and weights, so both ends of every comparison
  are guaranteed to use the identical ruler.

---

## 4. The experiments

| Experiment | What it does | Size |
|---|---|---|
| **Seasonal FSOI** | Replace all observations together by the background; attribute the change to every instrument and channel | 12 runs (3 networks × Jan/Apr/Jul/Oct 2025) = **694 cycles** |
| **Directional gradient check** | Nudge an instrument's inputs in random ±directions; compare the change in J with what the gradient predicted | 85 trials, 2 July cycles, radiosonde ruler |
| **Single-instrument replacement ("matched")** | Replace **only one** instrument (aircraft, ATMS, AMSU-A, or SEVIRI) and compare with its FSOI | 683 radiosonde-ruler cycles + 230 surface-ruler (SEVIRI) |
| **Path integration** | Evaluate gradients at 5 points along the way instead of 2 | 17 pre-chosen cycles |
| **Structural denial** | Remove aircraft from the graph entirely (not just replace values) | 58 July cycles |

**Sampling.** Some satellites have too many observations to decode at once, so a
geographically stratified sample is used: ATMS up to 50,000 rows per cycle (of
about 123,000; 62,000 in January), AMSU-A, SEVIRI, SSMIS and AVHRR up to 30,000.
Conventional networks are used in full. Sample totals can be scaled up to the whole
population with **Horvitz–Thompson** weights (each sampled row counts 1/probability
of being picked).

---

## 5. The results, in plain words

**Gradients are correct.** All 85 directional trials passed: the gradient predicted
the change in J to within 0.02–0.13% (median), with correlations ≥ 0.999998.
Single-value checks could not resolve anything (one observation moves J less than
computer rounding), which is a limit of that test, not a failure.

**FSOI reproduces the combined change well.** Over 694 cycles, FSOI got the sign
right every time, and the monthly median closure ratio is 0.97–1.05.
Example: for aircraft verification the monthly medians are 0.971–0.992.

**For one instrument at a time it is less accurate.** Median closure 0.82–0.99;
sign right in 89.5–100% of cycles; FSOI slightly **underestimates** in every month.

**The gap is mostly the trapezoid.** With 5 points instead of 2, 14 of 17 test cases
land between 0.97 and 1.05, and the leftover error shrinks by 91–99.6% in 16 cases.
Example: a SEVIRI case where the 2-point estimate even had the **wrong sign**
(ratio −1.99) becomes 1.12 with 5 points. The 3 cases that stay off all have very
small realized changes, where ratios are touchy.

**Who helps most.** Error-normalized, pooled over rulers:

| Source | Impact (fraction of J; negative = helps) |
|---|---|
| Aircraft | −0.165 |
| Radiosonde | −0.098 |
| Surface | −0.087 |
| ATMS (best satellite) | −0.0105 |
| SEVIRI ASR | −0.0019 pooled, but **+0.0011 (harmful) for surface verification** |

- Conventional networks look biggest partly because each helps most when it is also
  the ruler (aircraft judged by aircraft). Leaving out that "self" term shrinks the
  spread between largest and smallest from 86× to 33×.
- **25 of 27** source–ruler combinations are clearly beneficial. SEVIRI → surface is
  clearly **harmful**; ASCAT → radiosonde is **inconclusive**.
- **SEVIRI was double-checked** by actually removing its increments: surface error drops
  by 0.21% on average, in every month. So the harmful signal is real, for this model and ruler.

**Size in context.** Day-to-day, J itself varies by ~13%. Aircraft's effect (16.5%)
is about one such swing; ATMS's (1.05%) is a tenth of one. Satellite effects are
small per cycle but consistent across hundreds of cycles.

**Channels.** Even helpful instruments have channels that hurt: 22 satellite channels
have harmful averages (6 ATMS, 8 SSMIS, 5 AMSU-A, 3 SEVIRI). Example: ATMS channels 16
and 19 hurt meridional wind at 150–300 hPa against both radiosonde and aircraft. These
are leads for experiments, not proof of bad data.

**Removing vs replacing.** Deleting aircraft from the graph changes error ~15% more
than replacing its values (ratio 1.15). The uncertainty interval touches 1, so this is
suggestive, not established.

---

## 6. How solid is each claim?

| Claim | Status |
|---|---|
| Gradients are numerically correct (radiosonde ruler) | Solid |
| Combined FSOI reproduces realized change (694 cycles, sign 100%) | Solid, with the small missing-entry caveat |
| Single-instrument FSOI underestimates somewhat | Solid for the 4 instruments tested |
| Trapezoid error explains most of the gap | Supported by 17 cases; convergence not proven |
| Instrument ranking and 25/27 beneficial | Solid (robust to bootstrap choice) |
| SEVIRI harmful for surface verification | Solid (confirmed by direct removal) |
| Channel sign patterns | Descriptive; hypotheses for experiments |
| Removing aircraft > replacing it | Suggestive only (interval includes 1) |
| Geographic maps | Solid (drawn on the whole metric) |

---

## 7. What is still open

1. **Manuscript housekeeping**: the AI-use statement, reference DOIs, funding and
   repository placeholders.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **J** | Forecast error on the fixed ruler (lower is better) |
| **x_c / x_0** | Inputs with real observations / with background values |
| **Innovation (d)** | Observation minus what the model expected |
| **FSOI (I)** | Gradient-based estimate of each observation's effect on J |
| **Closure ratio** | FSOI estimate ÷ realized change; 1 is perfect |
| **Background replacement** | Swap observed values for expected values, keep everything else |
| **Structural denial** | Delete the instrument from the graph entirely |
| **Verification network** | Which observations the forecast is scored against |
| **Horvitz–Thompson** | Scaling a sample up to the whole population, 1 / (chance of being sampled) |
| **Bootstrap** | Re-drawing cycles at random many times to see how much a mean could vary |
| **Self-verification (diagonal)** | An observation type scored against itself, which inflates its apparent value |
