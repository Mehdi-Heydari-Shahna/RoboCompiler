#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_stewart.py --profile full --native off --out results_local
