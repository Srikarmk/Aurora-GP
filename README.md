# Aurora-GP

Region-aware Gaussian Process approximation: does it help to route different parts
of the input space to approximations of different quality?

**Short answer, after a corrected evaluation: no.** This repository contains the
original implementation, a full audit of why the original results were wrong, and a
rebuilt benchmark that answers the question properly. All three are kept so the
provenance is inspectable.

| document | what it is |
|---|---|
| [`AUDIT.md`](AUDIT.md) | Every defect found in the original evaluation, with file:line and recomputed numbers |
| [`RESULTS_FAIR.md`](RESULTS_FAIR.md) | Results after repairing the protocol — 5 datasets x 5 seeds |
| [`results/fair/report.txt`](results/fair/report.txt) | Full aggregated tables |
| [`results/mechanism/report.txt`](results/mechanism/report.txt) | Rank sweep and oracle-routing study |

---

## The idea

AURORA partitions the input space into low / medium / high **importance** regions
(importance = predictive uncertainty + local sparsity, 30/40/30 by training
percentile) and routes test points to approximations of increasing cost: Random
Fourier Features -> Nystrom -> exact GP.

## What actually happened

The original evaluation reported a 34–46% calibration improvement over uniform
approximations. That number was an artifact. The kernel lengthscale was estimated as
the median pairwise distance of **raw** `X`, then applied to **standardized** `X`
([`src/approximation_benchmark.py:85-88`](src/approximation_benchmark.py#L85)). On
protein that is wrong by a factor of 155,000, which drove the baseline's kernel to a
constant — its RMSE (6.1346) equals the target standard deviation (6.1182) exactly,
i.e. R^2 = 0. The reported gain was the distance between AURORA and a model that was
predicting the mean.

With the lengthscale fitted in the space the kernel actually operates in, and the
observation noise fitted rather than hardcoded to 0.1:

| dataset | published gain | corrected gain |
|---|---|---|
| concrete | +47.2% | −2.8% |
| protein | +33.9% | −5.1% |
| robot_arm | +44.1% | −8.0% |
| sarcos | +46.2% | −6.2% |
| synthetic_heteroscedastic | +34.3% | −0.4% |
| **mean** | **+41.1%** | **−4.5%** |

Region-aware routing also fails a control the original never ran: with identical
fitted models and the region labels randomly permuted (proportions preserved),
importance-based routing is no better than random routing on 4 of 5 datasets.

## What did survive

Once every method is given the same tuning budget, the cheap sparse approximations
are **substantially better calibrated than the exact GP they approximate** — an 11x
ECE gap on protein, 3.7x on sarcos — while the exact GP remains more accurate on
RMSE. That separation is robust across seeds and is the finding worth pursuing. See
[`RESULTS_FAIR.md`](RESULTS_FAIR.md) §4 and the rank sweep in
[`results/mechanism/report.txt`](results/mechanism/report.txt).

---

## Reproducing

```bash
pip install -r requirements.txt
./reproduce.sh
```

Roughly 15 minutes for the fair benchmark and 60–90 minutes for the mechanism study
on a laptop CPU. Every number in `AUDIT.md` and `RESULTS_FAIR.md` comes out of these
scripts.

## Layout

```
src/
  fair_benchmark.py     rebuilt protocol: shared splits, fitted hyperparameters,
                        validation-selected capacity, 5 seeds, one ECE estimator
  fair_report.py        aggregation + paired per-seed contrasts
  mechanism.py          rank sweep (isolates the approximation gap) and
                        oracle-routing headroom test
  mechanism_report.py   aggregation for the above

  gp_baseline.py        ORIGINAL exact GP (GPyTorch). Used unchanged by the rebuilt
                        benchmark, so results do not depend on a reimplementation.
  region_identification.py  ORIGINAL Stage 1. Used unchanged; it is leak-free.
  approximations.py     ORIGINAL RFF / Nystrom. Superseded — fixed noise, no signal
                        variance, clipped predictive variance. Kept for provenance.
  aurora_final.py       ORIGINAL AURORA. Superseded.
  approximation_benchmark.py  ORIGINAL benchmark containing the lengthscale defect.
  tune_aurora.py        ORIGINAL tuning. Selects its winner on the test set.
  *_viz.py, comp_plots.py, res_comp.py, diagnose_*.py, test_*.py   original scripts

results/
  fair/         corrected results (authoritative)
  mechanism/    rank sweep + oracle routing
  <everything else>   ORIGINAL results. Superseded by results/fair — retained so the
                      audit can be checked against what was actually produced.
data/           five benchmark datasets as .npz (X, y, feature_names)
```

> **Note on the original artifacts.** `results/final_plots/table1_summary.csv`,
> `results/aurora_final/`, `results/approximations/` and the original figures contain
> the superseded claims. They are kept deliberately — `AUDIT.md` cites them as
> evidence — but they should not be read as findings.

## Datasets

| name | n | d | source |
|---|---|---|---|
| concrete | 1,030 | 8 | UCI concrete compressive strength |
| protein | 45,730 | 9 | CASP physicochemical properties |
| robot_arm | 8,192 | 8 | robot arm kinematics |
| sarcos | 48,933 | 21 | SARCOS 7-DOF inverse dynamics |
| synthetic_heteroscedastic | 5,000 | 2 | generated, region-varying noise |

The corrected benchmark caps every dataset at 8,000 points before splitting so that
all methods see identical training data; protein and sarcos are therefore subsampled.

## License

MIT — see [`LICENSE`](LICENSE).
