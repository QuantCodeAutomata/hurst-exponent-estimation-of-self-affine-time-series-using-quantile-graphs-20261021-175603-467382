#!/usr/bin/env bash
# reproduce.sh — run all experiments and tests for the Hurst-exponent paper
# replication.  Requires Python 3.9+ and packages in requirements.txt.
#
# Usage:  bash reproduce.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Hurst exponent estimation via quantile graphs — reproduce"
echo "============================================================"

# ---- ensure dependencies ----
python3 -m pip install -q -r requirements.txt

# ---- run tests ----
echo ""
echo "---- Running tests ----"
python3 -m pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed — aborting experiment run."
    exit 1
fi

# ---- run experiments ----
echo ""
echo "---- Experiment 1: Transition topology across self-affine processes ----"
python3 -m exp.exp1

echo ""
echo "---- Experiment 2: Hurst estimation from mean-jump scaling (fBm) ----"
python3 -m exp.exp2

echo ""
echo "---- Experiment 3: Q-sensitivity of Hurst estimation ----"
python3 -m exp.exp3

echo ""
echo "---- Experiment 4: fGn extension via cumulative summation ----"
python3 -m exp.exp4

echo ""
echo "============================================================"
echo " All experiments complete.  Results are in results/"
echo "============================================================"
