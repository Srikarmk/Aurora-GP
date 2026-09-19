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

echo "==> Routing headroom re-checked with exact GP inference"
$PY routing_recheck.py

echo "==> Fair benchmark re-run with exact GP inference"
$PY fair_benchmark_exact.py

echo
echo "Done. See RESULTS_FAIR.md, results/fair/report.txt, results/mechanism/report.txt"
