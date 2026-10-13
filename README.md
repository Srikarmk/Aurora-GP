# Aurora-GP

Region-aware Gaussian processes: does it help to treat different parts of the input
space differently, and if so, what should differ?

**Short answer: adapt the noise model by region, not the approximation.** Routing
regions to sparse approximations of different fidelity cannot change the uncertainty
a region receives, because those approximations alter only the latent term of the
predictive variance and leave the shared noise term untouched. Making the noise term
input-dependent does change it, and improves calibration where the data have
input-dependent residual variance.

The paper is in [`paper/`](paper/) (`main.tex`, built with
`cd paper && tectonic -X compile main.tex`).

| document | what it is |
|---|---|
| [`paper/main.tex`](paper/main.tex) | The paper |
| [`FINDINGS.md`](FINDINGS.md) | Chronological record of what was claimed, tested, and withdrawn, newest correction first |
| [`AUDIT.md`](AUDIT.md) | Every defect found in the original AURORA evaluation, with file:line and recomputed numbers |
| [`data/README.md`](data/README.md) | Dataset sources, licences, checksums and known issues |
| [`HANDOFF.md`](HANDOFF.md) | Current status, open work, environment and gotchas |

## Main results

- **Fidelity and uncertainty are orthogonal.** On four constructed datasets with known
  regional structure, RFF, two Nystrom ranks and the exact GP emit mean predictive
  standard deviations within 2% of one another in every region, while the true noise
  varies 50x. An oracle router given the true partition gains nothing (+0.0003 ECE).
- **Input-dependent noise pays.** With an unbiased out-of-fold noise estimate, regional
  noise is significantly better calibrated than a single global noise parameter on 5 of
  11 datasets and significantly worse on none; a two-stage heteroscedastic GP is
  significantly better on NLL on 8. Replicates on disjoint seeds.
- **A measurement hazard.** GPyTorch's `fast_pred_var()` is a no-op below
  `max_cholesky_size` (default 800) and perturbs predictive standard deviations by a
  median of ~100% above it, inflating ECE 3-10x at n=4000.

## How the project got here

It started as an implementation of AURORA, which routes "important" regions
(predictive uncertainty + sparsity) to more faithful approximations. The original
evaluation reported a 41% calibration improvement; that was an artifact of a kernel
lengthscale computed on raw inputs and applied to standardised ones, which made the
baseline a constant predictor (`AUDIT.md`). Corrected, the mean gain is -9.9%.

Later, the project's own noise-model comparison reached the opposite of its current
conclusion because its noise estimator double-counted latent variance and took a
geometric mean of squared residuals. The correction is the top entry of
`FINDINGS.md`, and Section 7 of the paper records every claim that failed and what
caught it.

## Reproducing

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python ./reproduce.sh
```

`reproduce.sh` runs every script behind a number in the paper, then regenerates the
figures (`src/make_figures.py`) and the result tables (`src/paper_tables.py`). It takes
a few hours on a laptop CPU.

## Layout

```
src/
  hetero_fair.py        main noise-model comparison; oof_noise() is the shared estimator
  adversarial*.py       adversarial audit of the noise-model claim, incl. the n sweep
  shrinkage_noise.py    shrinkage toward global noise
  hetero_independent.py independent GPyTorch heteroscedastic GP
  orthogonality.py      fidelity vs predictive uncertainty (paper Table 2)
  structured_*.py       constructed datasets and routing on them
  inference_*.py        the fast_pred_var hazard
  paper_tables.py, make_figures.py   paper tables and figures from results/

  gp_baseline.py, region_identification.py, approximations.py, aurora_final.py,
  approximation_benchmark.py, tune_aurora.py, ...   ORIGINAL code, deliberately
                        unmodified because AUDIT.md cites it; fixes are applied in
                        new files or monkeypatched (src/routing_recheck.py)
  datacollect.py        original dataset preparation, recovered from history

results/                stored results; original subdirectories are kept as audit evidence
data/                   datasets; see data/README.md
```

> **Note on the original artifacts.** `results/final_plots/`, `results/aurora_final/`,
> `results/approximations/`, `RESULTS_FAIR.md` and the original figures contain
> superseded claims. They are kept deliberately as evidence and should not be read as
> findings.

## License

MIT, see [`LICENSE`](LICENSE). Dataset licences are listed in `data/README.md`.
