# Region-aware GPs: the right knob is the noise model, not the approximation

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
