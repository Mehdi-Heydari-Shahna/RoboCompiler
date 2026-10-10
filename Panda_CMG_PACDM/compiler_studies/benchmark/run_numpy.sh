#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python run_franka.py --profile full --native off --out results_local
