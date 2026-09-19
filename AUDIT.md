# Aurora-GP — Pre-publication Audit

Audited commit `84643a7`. Every number below was recomputed from the repo's own
data and code, not taken from the reported artifacts.

## Verdict

**The headline claim does not survive audit.** The claim in `src/aurora_final.py:5`
— *"achieves 34-46% improvement over uniform approximations"* — is an artifact of a
unit bug in how the baseline's kernel lengthscale is chosen. Correct the bug and the
mean ECE gain goes from **+41.1% to −15.2%**. The repo's own tuning log independently
shows the region-aware mechanism contributes ~1%.

This is not fixable by rewording. It needs a re-run against repaired baselines.

---

## Tier 1 — Blocking

### 1. Lengthscale is measured in raw feature space and applied in standardized space

`src/approximation_benchmark.py:85-88` takes the median pairwise distance of **raw**
`X_train`, but `RandomFourierFeatures.fit` and `NystromApproximation.fit`
(`src/approximations.py:46`, `:171`) standardize `X` before the kernel ever sees it.
The kernel therefore operates on unit-variance inputs with a lengthscale measured in
the original units.

| dataset | ℓ used | ℓ correct (scaled space) | error |
|---|---|---|---|
| concrete | 260.66 | 3.69 | **71×** |
| protein | 484,050 | 3.11 | **155,000×** |
| sarcos | 29.93 | 6.11 | 4.9× |
| synthetic | 5.11 | 1.79 | 2.9× |
| robot_arm | 3.54 | 3.93 | 0.9× (raw features are already ~unit scale) |

At ℓ=484,050 the RBF kernel is ≈1 for every pair, so the model can only emit a
constant. Confirmed: protein RFF RMSE is **6.1346** and the dataset's `y.std()` is
**6.1182** — R² = −0.000, a literal mean predictor. RFF and Nyström return identical
numbers on protein for the same reason.

**This is the baseline that the entire paper's improvement is measured against.**

The same broken value is then piped into AURORA's own low/medium models by
`load_baseline_hyperparameters` (`src/aurora_final.py:42-45`), while the high-region
exact GP silently ignores it and fits its own — which is exactly why the high region
looks so good.

**Effect of repairing it** (repo's own `NystromApproximation`, only ℓ changed):

| dataset | AURORA ECE | vs broken Nyström | vs scale-correct Nyström |
|---|---|---|---|
| concrete | 0.2160 | +47.2% | +29.4% |
| protein | 0.3139 | +33.9% | +21.5% |
| robot_arm | 0.1689 | +44.1% | +46.2% |
| sarcos | 0.0596 | +46.2% | **−165.2%** |
| synthetic | 0.2609 | +34.3% | **−7.9%** |
| **mean** | | **+41.1%** | **−15.2%** |

Add a single untuned noise setting (σ=0.5) to that same uniform baseline and AURORA
loses on 4 of 5 datasets (mean −203%). On RMSE, AURORA beats the scale-corrected
uniform Nyström on **0 of 5** datasets.

### 2. The repo's own tuning log falsifies the proposed mechanism

`results/tuning/concrete_tuning.json`, unmodified:

| config | high-importance model | ECE | improvement |
|---|---|---|---|
| 1: Extreme Ratio | Nyström 1500 | 0.4107 | −0.4% |
| 2: Conservative | Nyström 600 | 0.4049 | +1.1% |
| 4: Very Cheap Low | Nyström 1000 | 0.4097 | −0.1% |
| 5: Extreme High | Nyström 2000 | 0.4044 | +1.2% |
| **3: Exact GP for High** | **Exact GP** | **0.2257** | **+44.8%** |

Region-aware routing is fully active in all five. When every tier is a kernel
approximation, routing buys **≈1%**. The entire effect appears only when one tier is
swapped for the one model that fits its own hyperparameters. The paper's thesis is
regional adaptivity; the measured cause is hyperparameter fitting.

### 3. The winning configuration was selected on the test set

`src/tune_aurora.py:187` splits with `random_state=42`; `:240` evaluates each config
on `X_test, y_test`; `:290` picks the winner by that score. There is no validation
split. `src/aurora_final.py` then reports headline results on **the same
`random_state=42` split**. Config 3 is selected on the data it is scored on.

### 4. Compared rows are not evaluated on the same test set

`src/aurora_final.py:349-352` subsamples to 10,000 points *before* splitting;
`gp_baseline.py` and `approximation_benchmark.py` split the full dataset.

| dataset | Exact GP / Nyström n_test | AURORA n_test |
|---|---|---|
| concrete | 206 | 206 |
| robot_arm | 1639 | 1639 |
| synthetic | 1000 | 1000 |
| **protein** | **9146** | **2000** |
| **sarcos** | **9787** | **2000** |

Two of five rows in `results/final_plots/table1_summary.csv` compare numbers computed
on different data.

### 5. AURORA is strictly dominated by the exact GP it contains

| dataset | RMSE (GP → AURORA) | NLL (GP → AURORA) | train s (GP → AURORA) |
|---|---|---|---|
| concrete | 5.61 → 9.95 | 3.44 → 16.43 | 1.78 → 8.17 |
| protein | 4.47 → 5.53 | 2.94 → 29.30 | 27.47 → 45.58 |
| robot_arm | 0.079 → 0.109 | −0.71 → 3.94 | 14.39 → 46.12 |
| sarcos | 3.03 → 4.38 | 2.99 → 3.63 | 19.38 → 48.15 |
| synthetic | 0.142 → 0.493 | 0.82 → 16.84 | 20.27 → 27.28 |

Worse on **every** dataset on accuracy, likelihood **and** wall-clock. An
approximation method that is slower and less accurate than the exact method it
approximates has no operating point. The cause is structural: AURORA trains *two*
exact GPs (one inside `RegionIdentifier`, one as model 2) plus RFF plus Nyström.
`region_id_time` is ~50% of `training_time` in every run.

### 6. The "Exact GP" baseline is not exact above n=800

`gp_baseline.py:157` wraps prediction in `gpytorch.settings.fast_pred_var()`. That is
LOVE (Lanczos Variance Estimates) — an **approximation of the predictive variance**.
GPyTorch additionally falls back from Cholesky to iterative CG/Lanczos solves above
`max_cholesky_size`, which defaults to **800**. Both approximate exactly the quantity
ECE and NLL are computed from.

Same fitted model, same data, same splits; only the inference mode changes:

| dataset | n_train | ECE as written | ECE with exact inference |
|---|---|---|---|
| concrete | 618 | 0.0451 | 0.0451 |
| synthetic | 3000 | 0.2708 | **0.0437** |
| robot_arm | 4800 | 0.2681 | **0.0869** |
| protein | 4800 | 0.1198 | **0.0381** |
| sarcos | 4800 | 0.3141 | **0.0332** |

Concrete is unchanged because 618 < 800 — it is the only dataset that was ever
receiving exact inference. Every other Exact GP calibration number in the original
`results/` is a measurement of an approximation, off by 3-9x.

This propagates further than the baseline row. `RegionIdentifier.get_uncertainty_map`
calls the same `predict()`, so the importance scores — and therefore the 30/40/30
partition itself — are computed from approximated uncertainties whenever
n_train > 800. Measured region agreement between approximate and exact inference:

| dataset | n_train | region agreement | approx L/M/H | exact L/M/H |
|---|---|---|---|---|
| concrete | 618 | 100.0% | 28/36/36 | 28/36/36 |
| synthetic | 3000 | 99.8% | 29/40/31 | 29/40/31 |
| sarcos | 4800 | 88.8% | 29/41/30 | 27/36/37 |
| robot_arm | 4800 | 79.3% | 29/38/34 | 22/32/47 |
| protein | 4800 | **70.2%** | 26/40/34 | 20/27/53 |

On protein nearly a third of test points are assigned to a different region. Any
experiment that conditions on this partition — including the oracle-routing test in
`results/mechanism/` — must be re-run with exact inference before it can be trusted.
See `src/routing_recheck.py`.

Fix: set `gpytorch.settings.max_cholesky_size` above n and drop `fast_pred_var()`
when the predictive variance is the reported quantity. Cost is O(n^3) rather than
iterative, which is the price of the word "exact".

---

## Tier 2 — Serious

6. **`sigma_noise` is never fitted.** Hardcoded `0.1` at `approximations.py:17,143`,
   `approximation_benchmark.py:93,130`, `aurora_final.py:49,72`. Meanwhile the exact
   GP fits its noise by marginal likelihood (`gp_baseline.py:106,117`). On
   standardized targets a fixed 0.1 is a variance floor of 0.01 where the truth is
   often ~0.3–1.0 — this alone produces the NLL values of 16–29. The function is
   named and documented as "Load **tuned** hyperparameters" but returns a literal.
7. **No signal variance in the approximations.** `k_star_star = 1.0`
   (`approximations.py:211`); no outputscale anywhere, while the GP uses `ScaleKernel`.
   The approximations cannot represent target amplitude.
8. **Nyström predictive variance is wrong, then clipped.** `approximations.py:218`
   omits the data-dependent term (uses `K_mm_inv`, not `A_inv`), then `:219` hard-clips
   variance to `[σ², 3.0]`. A magic constant `3.0` is setting the calibration number
   the paper reports.
9. **Two different ECE estimators.** `gp_baseline.py:254` draws an **unseeded**
   Monte-Carlo z-score from `np.random.randn(10000)`; every other file uses
   `norm.ppf`. The Exact GP row is a random variable (per-level z deviates up to
   ±0.07) and is not the same metric as the AURORA row it is tabled against.
10. **Figure 2 and Table 1 disagree.** `comp_plots.py:184` uses `min(RFF, Nyström)`;
    `:338` uses Nyström only. sarcos is therefore published as **+45.4%** in Figure 2
    and **+46.2%** in Table 1.
11. **Fabricated fallback denominators.** `comp_plots.py:333` (`uniform_ece = 0.4`),
    `tune_aurora.py:197,205,206,368` (`0.41`, `0.29`). If a baseline file is missing,
    an "improvement" is computed against an invented number with no warning.
12. **No error bars.** One seed, one split, five datasets, no repeats anywhere. None
    of the reported gaps have a stated uncertainty.
13. **Stated range is wrong.** Code claims "34-46%"; the values actually produced are
    33.9%–**47.2%**.

---

## Tier 3 — Correctness and hygiene

14. **`density_map` is sparsity, not density.** `region_identification.py:81-84`
    scores by *mean kNN distance* — larger = sparser — and stores it as
    `density_map`, printed as "Computing Data Density". Importance therefore rises
    with sparsity. Defensible as a design, but any paper sentence saying "high-density
    regions receive the exact GP" states the opposite of the code.
15. **sarcos contradicts the mechanism.** Its high-importance region has the *worst*
    ECE (0.3083) versus low 0.1249 and medium 0.1438 — routing actively hurts there.
    It is still reported as the second-largest improvement (+46.2%).
16. `predict_regions` refits `NearestNeighbors` on the entire training set on every
    call (`region_identification.py:135-144`) — inside the timed inference path.
17. README documents only the datasets. No method description, no reproduction steps,
    no seeds, no entry point, no license.
18. `results/final_plots/table1_summary.txt` contains mojibake ("Uniform Nystr?m").
19. Dead/superseded code retained: `region_aware.py` v1 collapsed every test point
    into a single region (`results/region_aware/summary.json`: low 0, medium 0,
    high 206) with `std_std` = 1.1e-05.

---

## What is sound

These parts are correct and worth keeping:

- **Exact GP baseline** (`gp_baseline.py`) is structurally standard — `ConstantMean`,
  `ScaleKernel(RBFKernel)`, marginal-likelihood training, correct rescaling of
  predictive variance by `scaler_y.scale_`. But see Tier 1 finding 6: its *inference*
  is approximate above n=800, so its reported calibration is not the model's.
- **Nyström posterior mean** (`approximations.py:193-195`) is the correct SoR/DTC form.
- **RFF posterior** (`approximations.py:59-64`) is a correct Bayesian ridge solution
  including the posterior covariance, given a unit prior.
- **Region identification is leak-free at prediction time.** This was done carefully:
  `adaptive_thresholds=False` reuses training thresholds, test uncertainties are
  clipped to the training range, kNN is fit on training data only, and the explicit
  leakage warning at `region_identification.py:160` is correct. The test-time region
  proportions hold at ~30/40/30 as designed — I verified this against the per-region
  counts rather than assuming it.
- **Metrics** (RMSE, MAE, R², Gaussian NLL) are correctly implemented, and the ECE
  definition — average absolute deviation between nominal and empirical central
  interval coverage — is legitimate.
- **Data handling is deterministic**: seeded splits and seeded subsampling throughout.

---

## Path to a publishable result

The infrastructure is real; the experiment is not yet an experiment. In order:

1. **Fix the lengthscale** — take the median heuristic *after* standardization, or
   better, fit ℓ and σ by marginal likelihood for RFF/Nyström too.
2. **Give every method the same tuning budget.** As it stands, one arm is tuned and
   the others are not; that difference alone explains the result. Add an outputscale
   and a fitted noise to the approximations, and drop the `3.0` variance clip.
3. **Separate validation from test.** Select the config on a validation split, report
   on an untouched test split.
4. **Use one test set per dataset** across all methods.
5. **Re-run with ≥5 seeds** and report mean ± std. With n=5 datasets, a 5-point ECE
   difference without error bars is not evidence.
6. **Then ask the real question.** The honest version of the finding already in this
   repo is: *"On these benchmarks, the gain attributed to region-aware routing is
   ~1%; effectively all of it comes from per-region model capacity/hyperparameter
   fitting."* That is a legitimate, publishable negative result and a useful one —
   `results/tuning/concrete_tuning.json` is already the evidence for it. It is a much
   stronger paper than the current claim, because the current claim is not true.

Reproduction of every figure in this audit:
`src/approximation_benchmark.py:85-88`, `results/tuning/concrete_tuning.json`,
`results/*/summary.json`, and the re-run of the repo's own `NystromApproximation`
with only the lengthscale changed.
