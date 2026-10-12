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
| `paper/main.tex` | The paper. Complete, no placeholders. Never compiled. |

## Current status: ready to submit, not yet compiled

The paper is finished in content. **It has never been run through LaTeX**, because no
TeX distribution was installed on the machine it was written on. The first compile
will almost certainly need fixes.

`paper/arxiv_abstract.txt` is a 1906-character plain-text abstract for the arXiv
metadata field (the field caps at 1920 and cannot render `\cref` or `\emph`).

## Immediate next steps

1. **Compile the paper.** Upload `paper/` to Overleaf, or install MacTeX
   (`brew install --cask mactex-no-gui`), then
   `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.
   Read it end to end. Check figure placement, table widths, and that every
   `\cref` resolves rather than printing `??`.
2. **Decide the author contact email.** `manda.k@northeastern.edu` expires in roughly
   four months and arXiv v1 is permanent. Recommended: keep the institutional
   affiliation and add a durable personal address.
3. **Confirm Alekhya Bulusu has agreed to authorship** before submitting.
4. **Submit to arXiv**: primary `cs.LG`, cross-list `stat.ML`. Register with the
   `.edu` address to get auto-endorsement, then change the account email to a
   permanent one before it expires.

## Open work, in descending value

1. **Characterise the crossover.** The strongest open result. Regional noise loses to
   a single global noise parameter below $n\approx2500$ and wins above it
   (`results/adversarial_n.json`). Running the noise comparison at $n = 4000$ to
   $20000$ would turn "our conclusion reverses" from a caveat into a result about when
   input-dependent noise starts paying. Script to extend: `src/adversarial_n.py`.
2. **A structurally different heteroscedastic baseline.** Both current het-GP
   implementations (ours in numpy, and the GPyTorch one in
   `src/hetero_independent.py`) are two-stage Goldberg-style constructions. A
   variational heteroscedastic GP would test the algorithm rather than the
   implementation. This matters: four separate headline numbers in this project
   turned on baseline configuration.
3. **Scaling.** Everything runs at $n \le 4800$. Sparse approximations matter most at
   $10^5$ and above, and the orthogonality criterion should be checked there.
4. **Explain the shrinkage result.** `src/shrinkage_noise.py` improves CRPS at every
   sample size, but the mechanism we proposed (variance reduction at small $n$) was
   falsified: the selected $\lambda$ shows no trend in $n$. The effect is real and
   unexplained. Do not ship an explanation without an experiment isolating it.
5. **Smoothly varying noise.** `src/structured_data.py` varies noise in steps at
   region boundaries, which favours a piecewise-constant noise model. Regenerating
   with smooth variation would remove a known bias in our own favour.

## Environment

- Python: `/usr/local/bin/python3` (3.13, arm64). The default `python3` on that
  machine had no numpy. On a new machine, any Python with the pinned
  `requirements.txt` will do.
- No LaTeX toolchain was installed. Overleaf was the intended path.
- Long jobs were run with `nohup ... &` writing to `/tmp`, not to the session
  scratchpad, because the scratchpad is wiped when a session ends and one run was
  lost that way.

## Reproducing

`./reproduce.sh` regenerates every number in `AUDIT.md`, `FINDINGS.md` and the paper
from the committed data. It runs 20 scripts and takes a few hours. Figures come from
`src/make_figures.py`, which reads `results/` and writes `paper/figs/*.pdf`.

## Conventions

- **Commits are authored as Srikarmk, with no Claude attribution.** Set
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
