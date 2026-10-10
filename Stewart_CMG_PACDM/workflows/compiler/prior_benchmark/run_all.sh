#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_comparison.py --profile full --native required --out results_local
