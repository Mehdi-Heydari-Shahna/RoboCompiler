#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${ISAAC_SIM_PATH:-}" ]]; then
    exec "$ISAAC_SIM_PATH/python.sh" "$PWD/run_validation.py" "$@"
else
    exec python "$PWD/run_validation.py" "$@"
fi
