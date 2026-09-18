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

## 4. There is one real, publishable finding in here

It is not the one the repo claims. Once every method is given the same tuning budget,
**the cheap sparse approximations are dramatically better calibrated than the exact
GP**, consistently and with large effect sizes:

| dataset | Exact GP ECE | Uniform RFF | Uniform Nystrom |
|---|---|---|---|
| concrete | 0.0446 | **0.0329** | 0.0355 |
| protein | 0.1170 | **0.0106** | 0.0118 |
| robot_arm | 0.2701 | 0.0289 | **0.0238** |
| sarcos | 0.3178 | **0.0854** | 0.0870 |
| synthetic | 0.2738 | 0.0239 | **0.0232** |

An 11x calibration gap on protein and a 3.7x gap on sarcos, in favour of the
*approximation*. The exact GP is simultaneously the most accurate on RMSE almost
everywhere — so this is a clean accuracy-vs-calibration separation, not a case of one
model simply being better. The mechanism is plausible and testable: the low-rank
approximation gap adds predictive variance that partially offsets the exact GP's
well-known overconfidence under model misspecification.

That is a legitimate paper: *"Sparse GP approximations are better calibrated than the
exact GPs they approximate — and the effect is large enough to matter."* It needs
more datasets, a proper mechanism section (decompose the variance inflation), and
comparison against calibrated baselines. But unlike the current claim, it survives
its own controls.

## 5. What cannot be salvaged

The region-aware routing claim. Across 25 fair runs it does not beat a uniform
approximation of the same family, and on 4 of 5 datasets it does not beat random
routing. Reporting it as a 34-46% improvement is not a presentation problem — the
comparison it rests on is against a model that, on protein, was predicting a constant.
