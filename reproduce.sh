#!/usr/bin/env bash
# Reproduce every number in RESULTS_FAIR.md and AUDIT.md from scratch.
# Runtime: ~15 min for the fair benchmark, ~60-90 min for the mechanism study.
set -euo pipefail
cd "$(dirname "$0")/src"
PY="${PYTHON:-python3}"

echo "==> Fair benchmark (5 datasets x 5 seeds, shared splits)"
$PY fair_benchmark.py

echo "==> Aggregate -> results/fair/report.txt"
$PY fair_report.py

echo "==> Mechanism study (rank sweep + oracle routing)"
$PY mechanism.py

echo "==> Aggregate -> results/mechanism/report.txt"
$PY mechanism_report.py

echo "==> Convergence check on the original exact-GP baseline"
$PY convergence_check.py

echo "==> Orthogonality: fidelity vs predictive uncertainty (paper Table 2)"
$PY orthogonality.py

echo "==> Heteroscedasticity screen across all datasets (paper Table 3)"
$PY measure_hetero.py

echo "==> Inference corruption sweep (paper Table 1)"
$PY inference_corruption.py
$PY inference_report.py

echo "==> Structured-data routing and noise studies (paper Tables 5, 6)"
$PY structured_routing.py
$PY noise_routing.py
$PY hetero_fair.py

echo "==> Routing headroom re-checked with exact GP inference"
$PY routing_recheck.py

echo "==> Fair benchmark re-run with exact GP inference"
$PY fair_benchmark_exact.py

echo "==> Adversarial audit: metric battery, granularity, tuning parity"
$PY adversarial.py

echo "==> Adversarial audit: sample-size sweep"
$PY adversarial_n.py

echo "==> Shrinkage-regularised regional noise"
$PY shrinkage_noise.py

echo "==> Routing headroom at large n"
$PY routing_largen.py

echo "==> Diff test: same protocol on disjoint seeds 10-19"
$PY diff_seeds.py

echo
echo "Done. See RESULTS_FAIR.md, results/fair/report.txt, results/mechanism/report.txt"
