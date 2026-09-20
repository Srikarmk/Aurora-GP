# Region-aware GPs: a negative result in both of its natural forms

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
