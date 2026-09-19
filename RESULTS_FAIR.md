# Aurora-GP — Results after repairing the evaluation

Produced by `src/fair_benchmark.py` + `src/fair_report.py`. 5 datasets x 5 seeds,
mean +/- std, one shared split per (dataset, seed), hyperparameters fitted on train,
capacity chosen on validation, test read once. Raw data in
`results/fair/raw_results.json`, full tables in `results/fair/report.txt`.

The exact-GP arm and Stage 1 region identification call the repo's **original,
unmodified** `GaussianProcessBaseline` and `RegionIdentifier`, so nothing here
depends on my having reimplemented the baseline.

---

## 1. The headline gain was the bug

Expected Calibration Error, uniform Nystrom vs AURORA:

| dataset | published Nystrom | published AURORA | published "gain" | **fair Nystrom** | **fair AURORA** | **fair gain** |
|---|---|---|---|---|---|---|
| concrete | 0.4092 | 0.2160 | +47.2% | **0.0355** | 0.0365 | **−2.8%** |
| protein | 0.4745 | 0.3139 | +33.9% | **0.0118** | 0.0124 | **−5.1%** |
| robot_arm | 0.3020 | 0.1689 | +44.1% | **0.0238** | 0.0257 | **−8.0%** |
| sarcos | 0.1108 | 0.0596 | +46.2% | **0.0870** | 0.0924 | **−6.2%** |
| synthetic | 0.3973 | 0.2609 | +34.3% | **0.0232** | 0.0233 | **−0.4%** |
| **mean** | | | **+41.1%** | | | **−4.5%** |

Fitting the lengthscale in the space the kernel actually operates in, and fitting
the noise instead of hardcoding 0.1, improves the *uniform baseline* by 10-40x.
On protein its ECE goes 0.4745 -> 0.0118 and its NLL 50.76 -> 2.94. The published
"34-46% improvement" was the distance between AURORA and a broken model.

With the baseline repaired, AURORA is behind it on all five datasets. Paired
per-seed difference: **+0.0018 ECE (AURORA worse)**; it wins 1-2 seeds out of 5 per
dataset, i.e. noise.

## 2. The importance signal mostly does not carry information

The control the original never ran: identical fitted models, region labels randomly
permuted, proportions preserved. If importance-based routing means anything, AURORA
must beat this.

Paired per-seed ECE difference, AURORA − shuffled (negative = genuine signal):

| dataset | all-approx tiers | seeds won | exact-GP tier | seeds won |
|---|---|---|---|---|
| concrete | +0.0020 | 3/5 | +0.0066 | 1/5 |
| **protein** | **−0.0016** | 4/5 | **−0.0152** | **5/5** |
| robot_arm | +0.0066 | 0/5 | +0.0211 | 0/5 |
| sarcos | +0.0081 | 0/5 | +0.0318 | 0/5 |
| synthetic | +0.0000 | 1/5 | +0.0163 | 0/5 |

On 4 of 5 datasets, routing by importance is **no better than routing at random** —
often measurably worse. Protein is the single genuine exception, and there the
effect is consistent (5/5 seeds, −0.0152 +/- 0.0049). That is worth understanding,
but it is one dataset out of five and it still does not beat simply using a uniform
RFF (0.0106).

### The decisive version: even an oracle router cannot win

`src/mechanism.py` evaluates all four models on every test point, then selects the
best model **per region using validation data** — an oracle that upper-bounds what
any region-aware router could achieve with these models and this partition.

| dataset | uniform best | oracle per-region | importance (AURORA) | random |
|---|---|---|---|---|
| concrete | **0.0318** | 0.0353 | 0.0378 | 0.0347 |
| protein | **0.0101** | 0.0139 | 0.0174 | 0.0148 |
| robot_arm | **0.0123** | 0.0160 | 0.0406 | 0.0231 |
| sarcos | 0.0824 | **0.0757** | 0.0992 | 0.0873 |
| synthetic | **0.0230** | 0.0235 | 0.0235 | 0.0238 |

Mean (oracle − uniform best) = **+0.0009**: the best achievable router loses to
using a single model everywhere on 4 of 5 datasets. The oracle's per-region choices
are also different on every seed, which says the per-region differences it selects on
are noise rather than structure.

This is the strongest statement available: the failure is not that AURORA's
importance criterion is poorly designed, but that there is no regional structure for
*any* criterion to exploit on these benchmarks.

## 3. AURORA is never on the efficiency frontier

| dataset | RMSE GP -> AURORA | NLL GP -> AURORA | train s GP -> AURORA |
|---|---|---|---|
| concrete | 6.121 -> 6.125 | 3.186 -> 3.187 | 0.4 -> 4.7 |
| protein | 4.416 -> 4.617 | 2.932 -> 2.954 | 8.1 -> 28.4 |
| robot_arm | 0.0795 -> 0.0846 | −0.712 -> −0.914 | 9.2 -> 26.5 |
| sarcos | 3.094 -> 3.456 | 2.990 -> 2.856 | 9.7 -> 55.7 |
| synthetic | 0.145 -> 0.143 | −0.027 -> −0.396 | 3.4 -> 22.9 |

3-7x slower than the exact GP it contains, and never better on RMSE by a meaningful
margin. AURORA does beat the exact GP on NLL on three datasets — but so does a plain
uniform approximation, at a fraction of the cost.

---

## 4. WITHDRAWN PENDING VERIFICATION — the calibration gap is probably an artifact

**An earlier version of this section claimed that sparse approximations are
dramatically better calibrated than the exact GP (an 11x ECE gap on protein). That
claim is not safe and is retracted here pending `src/convergence_check.py`.**

The exact-GP arm of the fair benchmark used the repo's `GaussianProcessBaseline`,
which optimizes with Adam at lr=0.1 for 50 iterations (`gp_baseline.py:114`). The
rank sweep in `src/mechanism.py` instead fits the same model class by L-BFGS on the
exact marginal likelihood, with restarts. On identical data and identical splits the
two disagree enormously:

| dataset | exact GP ECE (repo baseline, 50 Adam iters) | exact GP ECE (L-BFGS on exact MLL) |
|---|---|---|
| robot_arm | 0.2701 | **0.0168** |
| synthetic | 0.2738 | **0.0243** |
| sarcos | 0.3178 | **0.0932** |
| protein | 0.1170 | **0.0246** |
| concrete | 0.0446 | **0.0378** |

Same model, same data, same splits — only the optimizer differs. The most likely
explanation is that the baseline is undertrained, in which case the "gap" was
measuring an optimizer, not a property of sparse GPs.

The proposed mechanism was also wrong. I suggested the low-rank approximation gap
inflates predictive variance and offsets overconfidence. The rank sweep shows the
opposite: z-dispersion (std of standardized residuals; 1.0 = calibrated) falls
monotonically with rank and reaches ~1.0 at the exact GP, so low-rank models are
*more* overconfident, not less.

| dataset | z_disp at rank 25 | at rank 200 | exact |
|---|---|---|---|
| robot_arm | 3.344 | 2.095 | 1.009 |
| sarcos | 2.945 | 1.389 | 1.008 |
| concrete | 1.968 | 1.162 | 1.078 |
| protein | 1.279 | 1.115 | 0.992 |

### What may still survive

At **matched** hyperparameters — one set fitted per (dataset, seed) and shared by
every rank — ECE has an interior optimum in rank on 4 of 5 datasets:

| dataset | best rank | ECE there | exact GP ECE | ratio |
|---|---|---|---|---|
| sarcos | 200 | 0.0276 | 0.0932 | 3.4x |
| concrete | 200 | 0.0264 | 0.0378 | 1.4x |
| protein | 800 | 0.0173 | 0.0246 | 1.4x |
| synthetic | 50 | 0.0230 | 0.0243 | 1.1x |
| robot_arm | exact | 0.0168 | 0.0168 | — |

This comparison is internally valid in a way the earlier one was not: every row of
the sweep shares one fitted hyperparameter set, so nothing here is an optimizer
artifact. The honest version of the claim is narrow — *rank behaves as a calibration
regularizer with an interior optimum on some datasets* — and the sarcos effect (3.4x)
is the only large one. Do not write the strong version.

## 5. What cannot be salvaged

The region-aware routing claim. Across 25 fair runs it does not beat a uniform
approximation of the same family, and on 4 of 5 datasets it does not beat random
routing. Reporting it as a 34-46% improvement is not a presentation problem — the
comparison it rests on is against a model that, on protein, was predicting a constant.
