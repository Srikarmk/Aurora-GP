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

All numbers below use **exact GP inference** (see AUDIT.md Tier 1 finding 6); the
approximation arms are unaffected by that issue, but the region partition is.

| dataset | published "gain" | fair Nystrom | fair AURORA | **fair gain** |
|---|---|---|---|---|
| concrete | +47.2% | 0.0355 | 0.0365 | **-2.7%** |
| protein | +33.9% | 0.0118 | 0.0162 | **-36.8%** |
| robot_arm | +44.1% | 0.0238 | 0.0244 | **-2.6%** |
| sarcos | +46.2% | 0.0870 | 0.0929 | **-6.7%** |
| synthetic_heteroscedastic | +34.3% | 0.0232 | 0.0233 | **-0.7%** |

Mean published gain **+41.1%**; mean corrected gain **-9.9%**.

Fitting the lengthscale in the space the kernel actually operates in, and fitting the
noise instead of hardcoding 0.1, improves the *uniform baseline* by 10-40x. On protein
its ECE goes 0.4745 -> 0.0118 and its NLL 50.76 -> 2.94. The published "34-46%
improvement" was the distance between AURORA and a broken model; on protein, one that
was predicting a constant.

With the baseline repaired, AURORA is behind it on all five datasets.

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

## 4. The calibration finding, after two retractions

This section previously claimed an 11x calibration gap favouring sparse
approximations. That number was wrong: it compared against an exact GP whose
*inference* was approximate (AUDIT.md Tier 1 finding 6). Two proposed explanations
for the gap were also tested and rejected — an undertrained optimizer
(`src/convergence_check.py`: 1000 Adam iterations do not close it) and
hyperparameter fitting on a subsample (`n_fit` 500/1500/4800 changes nothing).

Here is the measurement with exact inference throughout, 5 seeds:

| dataset | Exact GP | best approximation | ratio |
|---|---|---|---|
| protein | 0.0369 | **0.0106** | 3.5x |
| robot_arm | 0.0815 | **0.0238** | 3.4x |
| synthetic | 0.0426 | **0.0232** | 1.8x |
| concrete | 0.0446 | **0.0329** | 1.4x |
| sarcos | **0.0315** | 0.0854 | **0.37x — exact GP wins** |

The effect is real but modest and **not universal**: approximations win on 4 of 5
datasets with ratios of 1.4-3.5x, and sarcos reverses cleanly in the other direction.

**This is not yet a paper.** Five datasets, one reversal, and no mechanism that has
survived testing. The honest summary is "sparse approximations are sometimes better
calibrated than the exact GP, for reasons we have not established." Publishing that
requires many more datasets and a mechanism isolated by experiment rather than
asserted — note that the two mechanisms proposed so far were both wrong, and that the
rank sweep positively rules out variance inflation (z-dispersion falls monotonically
with rank, reaching ~1.0 at the exact GP, so low-rank models are *more* overconfident).

What can be said confidently today is negative and methodological: **approximate GP
inference silently corrupts calibration measurements**, by 3-9x here, and the
threshold at which it engages (`max_cholesky_size=800`) is invisible in user code.
That is a genuinely useful warning for anyone benchmarking GP uncertainty, and it is
fully supported by the data in `results/`.

## 5. What cannot be salvaged

The region-aware routing claim. Across 25 fair runs it does not beat a uniform
approximation of the same family, and on 4 of 5 datasets it does not beat random
routing. Reporting it as a 34-46% improvement is not a presentation problem — the
comparison it rests on is against a model that, on protein, was predicting a constant.
