# Aurora-GP, handoff

Continuation notes for picking this up on a different machine. Everything described
here is committed and pushed to `main` at
`https://github.com/Srikarmk/Aurora-GP`.

## What this project is now

It started as an implementation of AURORA, a region-aware Gaussian process
approximation method. An audit found the headline result was an artifact of a units
bug, and the project became a paper about when region-aware GPs can and cannot work.

Three documents carry the state:

| file | what it holds |
|---|---|
| `AUDIT.md` | Every defect found in the original evaluation, with file:line and recomputed numbers |
| `FINDINGS.md` | Chronological record of what was claimed, tested, and withdrawn |
| `paper/main.tex` | The paper. Compiles cleanly with tectonic; result tables are generated into `paper/tables/` by `src/paper_tables.py`. |
| `data/README.md` | Dataset sources, licences, checksums and known issues |

## Current status (2026-10-13)

A full pre-submission audit (data, code, results, text) found that the noise-model
comparison used a biased noise estimator. Fixing it reversed Section 6: input-dependent
noise now beats a global noise parameter (see the correction at the top of
`FINDINGS.md`). Every affected experiment was rerun and the paper rewritten around the
corrected results; title subtitle, abstract, Sections 4, 6, 7 and the conclusion changed.
Other fixes: Table 2 ran on all four constructed datasets, het ratios are now over 10
seeds, dataset citations and a datasets table were added, figures were redrawn with
seed spread, and Table 8's unsourced 0.0146 was removed.

Build: `cd paper && tectonic -X compile main.tex` (tectonic via Homebrew).
Environment: `.venv` built from `requirements.txt` on Python 3.13.

## Immediate next steps

1. Read the compiled paper end to end, especially Sections 4 and 6, which are new.
2. **Co-author sign-off.** The conclusion of Section 6 reversed; Alekhya Bulusu should
   agree to the revised paper, not only to authorship.
3. **Decide the author contact email.** `manda.k@northeastern.edu` expires in roughly
   four months and arXiv v1 is permanent.
4. Submit to arXiv: primary `cs.LG`, cross-list `stat.ML`. Upload `main.tex`,
   `refs.bib`, `figs/`, `tables/` and the generated `main.bbl`.
5. Optional: SARCOS and California housing have no explicit licence; consider replacing
   the committed raw files with a fetch script.

## Open work, in descending value

1. **A structurally different heteroscedastic baseline.** Both het-GP implementations
   (numpy kNN in `src/hetero_fair.py`, GPyTorch GP in `src/hetero_independent.py`) are
   two-stage constructions, and they disagree on the two smallest datasets. A
   variational heteroscedastic GP (Lazaro-Gredilla & Titsias 2011) would test the
   algorithm rather than the implementation.
2. **Larger n for the noise comparison.** The main comparison caps n_train at 1500;
   the sweep reaches 4000 on three datasets. Extend `src/adversarial_n.py`.
3. **Scaling.** Everything runs at n <= 4800. Sparse approximations matter most at
   10^5 and above, and orthogonality should be checked there.
4. **Smoothly varying noise.** `src/structured_data.py` varies noise in steps at
   region boundaries, which favours a piecewise-constant noise model.
5. **`src/noise_routing.py`** still uses in-sample residuals and the true partition. It
   no longer backs any number in the paper; either port it to `oof_noise` or retire it.

## Environment

- Python: `.venv` (3.13, arm64, from `requirements.txt`). Previously `/usr/local/bin/python3`; the default `python3` on that
  machine had no numpy. On a new machine, any Python with the pinned
  `requirements.txt` will do.
- No LaTeX toolchain was installed. Overleaf was the intended path.
- Long jobs were run with `nohup ... &` writing to `/tmp`, not to the session
  scratchpad, because the scratchpad is wiped when a session ends and one run was
  lost that way.

## Reproducing

`./reproduce.sh` regenerates every number in `AUDIT.md`, `FINDINGS.md` and the paper
from the committed data. It runs 21 scripts, including
`src/make_figures.py` (writes `paper/figs/*.pdf`), and takes a few hours.

## Conventions

- **Commits are authored as Srikarmk.** Set
  `git config user.name "Srikarmk"` and `user.email "srikarmks@gmail.com"` and do not
  add `Co-Authored-By` trailers.
- Push to `main` directly.
- The original code (`src/gp_baseline.py`, `src/region_identification.py`,
  `src/approximations.py`, `src/aurora_final.py`, and the original `results/`
  subdirectories) is **deliberately unmodified**: `AUDIT.md` cites it as evidence.
  Fixes are applied in new files or monkeypatched at the point of use, as in
  `src/routing_recheck.py`.

## Hard-won gotchas

- **GPyTorch's `fast_pred_var()` is a no-op below `max_cholesky_size` (default 800)
  and corrupts predictive variance above it** by a median of about 100%. Any
  calibration measurement on an "exact" GP above $n=800$ is suspect. This was the
  single most consequential bug found.
- **Estimate any residual-derived quantity out of fold.** In-sample residuals from a
  GP conditioned on those same points underestimate noise by 1.1 to 1.9 times, and it
  silently handicaps whichever arm uses them pointwise.
- **A median-heuristic lengthscale must be computed in the space the kernel operates
  in.** Computing it on raw inputs and applying it to standardised inputs was wrong by
  155,000 times on one dataset and made the baseline a constant predictor.
- **Screen datasets for the property under study.** `synthetic_heteroscedastic`, built
  as a controlled heteroscedastic test case, measures *less* input-dependent noise
  than an ordinary robotics benchmark.
- **zsh does not word-split unquoted variables.** An audit script printed
  `CLEAN` three times on greps that had errored, because `$MY` reached grep as a
  single argument and a `|| echo` fallback reported the pass. Use an array.
- **A pipeline's exit status is its last stage.** A crashing script reported `EXIT=0`
  because its traceback was piped to `tail`.
- **Mechanisms are cheap, measurements are not.** Four of five explanatory mechanisms
  proposed in this project were falsified by their own predictions, while every claim
  resting on direct measurement survived adversarial audit. State a mechanism only
  alongside an experiment that isolates it.
