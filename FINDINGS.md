# Region-aware GPs: adapt the noise, not the approximation

## Correction, 2026-10-16: statistics, scope, and the hazard decomposition

A statistical review of the code and a second claim-by-claim check found:

1. **The Section 3 decomposition was wrong.** `inference_corruption.py` evaluated all
   inference paths on one model object, and GPyTorch reused the covariance cached by
   the first LOVE call. With the cache cleared per path (`reset_cache`): defaults
   101.92% median error in sigma above n=800; raising `max_cholesky_size` alone 0.00%;
   disabling `fast_pred_var` alone 5.62% (iterative-solve error, up to 38% on sarcos).
   The hazard stands; its control is the Cholesky threshold, not the flag.
2. **Multiple comparisons.** With Holm correction over 11 datasets, regional noise
   beats global on ECE on airfoil, sarcos, calhousing (seeds 0-9), plus
   synthetic_heteroscedastic (seeds 10-19); all five nominal datasets survive only
   when the 20 splits are pooled. Nadeau-Bengio corrected t-test: the three survive,
   concrete and synthetic_heteroscedastic do not. Regional is significantly worse on
   yacht for NLL/CRPS. `src/significance.py`.
3. **Orthogonality spreads were seed-averaged.** Per seed, sigma spread is up to 2.64%
   (was reported 0.37%). The four constructed-data models are similarly accurate, so
   the stronger evidence is the Nystrom rank sweep: RMSE changes 21-155%, mean sigma
   0-12%. Claims now scoped to RFF/Nystrom with homoscedastic noise; FITC/PITC excepted.
4. **"Unbiased" overclaimed.** Pointwise only if the fold model is well specified;
   region means over a partition built from the same evidence are biased; the floor
   binds for 0-59% of test points; yacht's mean evidence is negative in 9 of 10 seeds.
5. **Controls added** (`src/noise_controls.py`): three validation recalibrations of the
   global arm do not close the gap; a fixed configuration with no selection still wins
   on airfoil, sarcos, calhousing.
6. **Het ratio needs a null** (`src/measure_hetero_null.py`): its permutation null is
   1.8-4.1, not 1.
7. **GPyTorch results are not bit-reproducible.** Three runs of the hazard experiment
   (committed 2026-10-13, a full `reproduce.sh` rerun in a clean clone, and the
   cache-fixed run): median sigma error above n=800 101.98 / 100.31 / 101.92%; max at or
   below 800 0.0018% in all; LOVE-off median 5.00 / 4.80 / 5.62%; ECE ratio at n=4000
   2.96-10.16 / 2.90-9.99 / 2.92-10.01. Single cells vary more (synthetic n=2500:
   183 / 107 / 145%; exact ECE up to 0.006). numpy-based results reproduce bit-for-bit.

---

## Correction, 2026-10-13: the noise-model half of the negative result was an estimator bias

A pre-submission audit of the code against the paper found two biases in the noise
evidence shared by every noise-model experiment (`src/hetero_fair.py` and everything
importing `oof_resid2`):

1. **Regional noise double-counted latent variance.** The out-of-fold squared residual
   has expectation sigma^2(x) + var_oof[f(x)]. It was used as the noise estimate and the
   latent variance was then added again, making regional noise under-confident, most of
   all at small n.
2. **The het-GP took a geometric mean.** It smoothed log r^2 and exponentiated, which
   for Gaussian residuals sits at exp(-1.27) = 0.28 of the noise variance.

Both arms now use r^2_oof minus the fold model's latent variance (`oof_noise`). Rerun
results, 10 seeds, 11 datasets (`src/hetero_fair_report.py`):

| | global best | het-GP best | regional best |
|---|---|---|---|
| ECE, before | 9 | 0 | 2 |
| ECE, corrected | 3 | 2 | 6 |
| NLL, corrected | 2 | 9 | 0 |

Regional noise is significantly better calibrated than global (Wilcoxon p<0.05) on 5
datasets (sarcos, airfoil, calhousing, concrete, synthetic_heteroscedastic) and
significantly worse on none; the het-GP is significantly better on NLL on 8. Seeds
10-19 replicate (regional 5 better / 0 worse on ECE; het-GP 9 better on NLL).

Consequences for claims below, all now withdrawn:
- "Global noise wins 9 of 11" and "the het-GP wins none": reversed.
- The sample-size reversal (regional loses below n~2500, wins above): an artifact. With
  fixed validation/test sets and the corrected estimator (`src/adversarial_n.py`) the
  mean gap favours regional noise at every n from 300 to 4000.
- "Regional noise buys sharpness at the cost of calibration": regional intervals are now
  narrower on 5 of 6 datasets and better calibrated on most metrics.
- Shrinkage as a contribution: all input-dependent arms beat global CRPS by 0.4-1.3%;
  shrinkage adds ~0.2% over raw regional, within one SE. The earlier Table 7 also
  averaged CRPS across datasets in target units, so powerplant dominated it.
- Region identification "96.9%": now 96.7% with out-of-fold evidence; the 81.9% case
  (varying_smoothness) has homoscedastic noise, so it tracks model bias.

What survives unchanged: the fast_pred_var hazard (all numbers reproduce), and
orthogonality, now checked on all four constructed datasets (sigma spread <= 1.9% in
every region). The "criterion" was ECE spread mislabelled as sigma spread, and the
combined-dataset routing gain comes from region 0, not region 1; the paper now says so.

The text below is the record as it stood before this correction.

---


This is the result the project actually supports, after the original evaluation was
corrected (`AUDIT.md`) and re-run (`RESULTS_FAIR.md`).

## The claim

AURORA's premise — that a GP should adapt **by region** — is sound. Its mechanism —
adapting *approximation fidelity* by region — cannot work, for a reason that
generalizes beyond this implementation. On the same partitions, adapting the *noise
model* instead yields 2.0–4.4x better calibration.

## Evidence

### 1. On five standard benchmarks, routing gains nothing

Even an oracle selecting the best model per region on validation data loses to using
a single model everywhere (mean +0.0001 ECE, 4 of 5 datasets). `results/mechanism/`.

But those benchmarks were never verified to contain regional structure, so this alone
is uninformative about the idea.

### 2. It gains nothing on data built specifically to favour it

Four datasets (`src/structured_data.py`) with structure true by construction —
verified: 50x noise range, 17x density imbalance, region-varying lengthscale.

| dataset | uniform | true partition + oracle model | learned + oracle | AURORA | random |
|---|---|---|---|---|---|
| hetero_extreme | **0.2299** | 0.2315 | 0.2302 | 0.2322 | 0.2316 |
| varying_smoothness | **0.0656** | 0.0658 | 0.0659 | 0.0661 | 0.0659 |
| varying_density | **0.0069** | 0.0075 | 0.0070 | 0.0069 | 0.0065 |
| combined | 0.0525 | 0.0515 | **0.0504** | 0.0503 | 0.0537 |

Mean (true oracle − uniform) = **+0.0003**. Giving the method the true generative
partition *and* an oracle model choice still buys nothing.

### 3. Why: the candidate models are indistinguishable in exactly the dimension that matters

On `hetero_extreme`, true noise varies 50x across regions (0.020 / 0.141 / 1.000).
Every candidate model emits the same predictive standard deviation everywhere:

| region | RFF | Nystrom | Nystrom-1500 | exact GP | spread | true noise |
|---|---|---|---|---|---|---|
| 0 | 0.4970 | 0.4979 | 0.4979 | 0.4982 | 0.0012 | 0.020 |
| 1 | 0.3792 | 0.3791 | 0.3791 | 0.3751 | 0.0041 | 0.141 |
| 2 | 0.1658 | 0.1648 | 0.1648 | 0.1740 | 0.0092 | 1.000 |

(table entries are per-region ECE; mean predictive std is 0.563 in *all three* regions)

The spread between models within a region is at most 0.009 while the calibration
error is 0.5. **Approximation fidelity and predictive uncertainty are orthogonal**: a
sparse GP's predictive variance is dominated by the shared global noise term, not by
the approximation gap. Routing between homoscedastic models of differing rank cannot
produce region-appropriate uncertainty, no matter how good the partition.

### 4. Changing the knob works

Same partition, same posterior mean, same latent variance — only the noise term is
made regional, fitted on training residuals within each region (no test labels):

| dataset | ECE global | ECE regional noise | ratio | NLL global | NLL regional |
|---|---|---|---|---|---|
| varying_smoothness | 0.0675 | **0.0152** | **4.44x** | −0.131 | −0.357 |
| hetero_extreme | 0.2374 | **0.0937** | **2.53x** | 0.896 | −0.289 |
| combined | 0.0527 | **0.0270** | **1.95x** | 0.837 | 0.667 |
| varying_density | 0.0064 | 0.0063 | 1.01x | −1.571 | −1.571 |

`varying_density` is the specificity control: density varies but noise does not, and
the fix correctly does nothing there. The gain appears exactly where noise varies.


## 5. Against an established heteroscedastic GP — partial, and one arm is not trustworthy

A heteroscedastic GP is a known method. If it dominates regional noise, everything
above reduces to "use a het-GP on het data". Two-stage het-GP (fit a second GP to log
squared residuals), same splits, no test labels:

**ECE**

| dataset | global | het_gp | regional_true | regional_learn | partition recovery |
|---|---|---|---|---|---|
| hetero_extreme | 0.2374 | **0.0505** | 0.0937 | 0.1112 | 96.9% |
| varying_smoothness | 0.0675 | 0.1980 | **0.0152** | 0.0237 | 81.9% |
| varying_density | **0.0064** | 0.1947 | 0.0063 | 0.0069 | 36.7% |
| combined | 0.0527 | 0.1455 | **0.0270** | 0.0410 | 44.3% |

**NLL**

| dataset | global | het_gp | regional_true | regional_learn |
|---|---|---|---|---|
| hetero_extreme | 0.896 | 0.202 | **−0.289** | −0.218 |
| varying_smoothness | −0.131 | −0.379 | −0.357 | **−0.427** |
| varying_density | **−1.571** | −0.957 | −1.571 | −1.564 |
| combined | 0.838 | 1.200 | **0.667** | 0.827 |

### What is solid

**Noise-targeted region identification works.** Replacing uncertainty+sparsity with
kNN-smoothed local residual variance lifts partition recovery from 41.4% to **96.9%**
on hetero_extreme and 41.6% to **81.9%** on varying_smoothness (chance 33%). On
varying_density it sits at 36.7% — correctly finding nothing, because density varies
there but noise does not. On `combined` it reaches only 44.3%: density and noise vary
together, and kNN residual variance is confounded by density. That is a real limit.

**Regional noise never hurts.** It matches `global` on the control and beats it on the
other three, on both metrics. It uses three constants where the het-GP fits a surface.

### What is NOT trustworthy yet

The het-GP looks unstable here — worse than doing nothing on 3 of 4 datasets. **Do not
report that.** The het-GP arm is a single-iteration two-stage implementation written
for this comparison, not the EM procedure of Kersting et al. (2007), and its
hyperparameters were not selected on validation the way the other arms' were.

This is precisely the error this whole audit began by documenting: the original
project's headline came from comparing against a baseline that had not been given a
fair tuning budget. Claiming "regional noise beats heteroscedastic GPs" on the
strength of my own quick het-GP would repeat it exactly.

Required before any such claim:
1. A proper het-GP (EM to convergence, or a published implementation such as GPyTorch's
   `HeteroskedasticNoise`), tuned on validation with the same budget as every other arm.
2. Noise that varies **smoothly**, not in steps. These generators change noise at
   region boundaries, which structurally favours a piecewise-constant noise model. The
   comparison is currently rigged in regional noise's favour.
3. Real datasets with documented heteroscedasticity.

Until then the defensible claim is narrow: *regional noise is a cheap, robust
improvement over a single global noise parameter, and the partition that delivers it
can be learned from training residuals.* Whether it competes with a properly tuned
heteroscedastic GP is **open**.

## Status and what is still needed

- Regional noise is a coarse stand-in for a heteroscedastic GP. The comparison to
  make is against established heteroscedastic baselines, which this does not yet do.
- Partitions here are given or learned on synthetic data; §2 shows `RegionIdentifier`
  recovers the true partition only 41–49% of the time (chance is 33%), which is a
  second, separable problem.
- Four synthetic datasets and three seeds. Real datasets with documented
  heteroscedasticity are the necessary next step.

## Reproducing

```bash
python src/structured_routing.py   # sections 2 and 3
python src/noise_routing.py        # section 4
```


---

# 6. FINAL: the noise-model repair also fails on real data

Sections 4 and 5 proposed that the right knob is the noise model, on the strength of
2.0-4.4x ECE gains on constructed data. **That claim is withdrawn.**

With noise estimated from shared out-of-fold residuals (fixing a 1.1-1.9x in-sample
bias that had handicapped the heteroscedastic baseline), equal tuning budgets for
every arm, and 10 seeds on nine real datasets:

| dataset | het ratio | global | het-GP (EM) | regional | p (reg vs het) |
|---|---|---|---|---|---|
| energy | 7.05 | **0.0385** | 0.0940 | 0.0792 | 0.695 |
| airfoil | 4.44 | **0.0784** | 0.0971 | 0.0920 | 0.557 |
| calhousing | 4.10 | **0.0693** | 0.1549 | **0.0693** | 0.002 |
| concrete | 3.88 | **0.0400** | 0.1259 | 0.0660 | 0.002 |
| yacht | 3.55 | **0.0648** | 0.0741 | 0.2515 | 0.002 |
| wine_white | 2.75 | **0.0216** | 0.1680 | 0.0345 | 0.004 |
| robot_arm | 2.60 | **0.0163** | 0.0421 | 0.0989 | 0.002 |
| powerplant | 2.45 | **0.0197** | 0.1839 | 0.0214 | 0.002 |
| protein | 2.19 | **0.0189** | 0.1127 | 0.0446 | 0.002 |

Adding the final two datasets: sarcos (global 0.0815 vs 0.1104 / 0.1107) and
synthetic_heteroscedastic (regional **0.0172** vs global 0.0242).

**Final tally over 11 datasets — ECE: global 9, regional 2, het-GP 0. NLL: regional 7,
global 3, het-GP 1.**

The two datasets regional noise wins are exactly the two with *constructed* noise
structure. The machinery works where the phenomenon is real; across nine real datasets
the phenomenon is never strong enough to repay its parameters. Het ratio does not
predict the winner either — energy, the most heteroscedastic real dataset at 7.05, is
a clean win for global noise. Regional noise is actively harmful
where it loses (yacht 0.2515 vs 0.0648; robot_arm 0.0989 vs 0.0163). On NLL it is
milder -- regional 5, global 3, het-GP 1 -- but on ECE the conclusion is unambiguous.

The constructed-data gains appear only at het ratios of order 10^3, roughly 200x
beyond anything measured in real data. At realistic levels the extra parameters cost
more in estimation variance than they recover in bias.

Region identification is *not* the bottleneck: it is solvable (41-49% -> 96.9%
recovery by targeting residual variance), and solving it does not make region-aware
modelling pay.

**Net:** region-aware GP modelling fails in both natural forms -- routing the
approximation (mechanism: fidelity and predictive uncertainty are orthogonal) and
routing the noise (mechanism: real heteroscedasticity is too mild to pay for the
parameters). The paper in `paper/` reports both, with the mechanism for each.
