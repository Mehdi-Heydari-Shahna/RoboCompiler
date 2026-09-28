#!/usr/bin/env python3
"""Regenerate Stewart mechanics, five rollouts, audits, and six figures."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    print(f"Running {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *args],
                   cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--render', action='store_true',
                      help='Also render the simulation video during the full run.')
    mode.add_argument('--postprocess-only', action='store_true',
                      help='Use existing results/ trajectories to rebuild audits and figures.')
    args = parser.parse_args()
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    if args.postprocess_only:
        required = [f'{name}.{ext}'
                    for name in ('reference', 'nominal', 'fine', 'heavy_payload',
                                 'no_feedforward', 'pacdm')
                    for ext in ('npz', 'json')]
        missing = [name for name in required if not (ROOT / 'results' / name).is_file()]
        if missing:
            parser.error('Run python reproduce.py first. Missing results: ' + ', '.join(missing))
    else:
        run('run_stewart.py', *(['--render'] if args.render else []))

    run('Supplement/audit_stewart_evidence.py')
    run('Supplement/make_stewart_figures.py')
    run('Supplement/draw_stewart_cmg.py')
    run('Supplement/draw_stewart_pipeline.py')
    print('Completed. Figures: Supplement/figures/; numerical evidence: results/.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
