#!/usr/bin/env python3
"""Audit saved Pinocchio missions into a separate directory without rerunning them."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True,
                        help='New or empty directory for the audit output')
    parser.add_argument('--samples', type=int, default=111,
                        help='Native constraint samples per saved mission (default: 111)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    if out == root or out == root / 'results' or (root / 'results') in out.parents:
        parser.error('--out must not overwrite the workflow or recorded results')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('--out must be a new or empty directory')

    import pinocchio as pin
    import numpy as np
    import scipy
    if pin.__version__ != '3.8.0':
        raise RuntimeError('This audit requires Pinocchio 3.8.0; use environment.yml.')
    from stewart.pin_checks import validate_mechanics, audit_rollout, validate_audit_negative_controls
    from stewart.pin_evidence import audit_case
    from stewart.pin_results import CASES

    out.mkdir(parents=True, exist_ok=True)
    records = {}
    def record(name, value):
        records[name] = bool(value['passed'])
        (out / (name + '.json')).write_text(
            json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'{name}: {"PASS" if records[name] else "FAIL"}', flush=True)

    cmg = json.loads((root / 'data/stewart.cmg.json').read_text(encoding='utf-8'))
    record('mechanics', validate_mechanics(cmg))
    record('negative_controls', validate_audit_negative_controls(cmg, root / 'results/nominal.npz'))
    for name, dt, payload, ff in CASES:
        actual = json.loads((root / f'results/{name}.cmg.json').read_text(encoding='utf-8'))
        record('trajectory_' + name, audit_rollout(actual, root / f'results/{name}.npz', samples=args.samples))
        record('evidence_' + name, audit_case(root, name, dt, payload, ff))
    summary = dict(passed=all(records.values()), checks=records,
                   samples_per_trajectory=args.samples, trajectories_reintegrated=False,
                   versions=dict(python=platform.python_version(), pinocchio=pin.__version__,
                                 numpy=np.__version__, scipy=scipy.__version__))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return int(not summary['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
