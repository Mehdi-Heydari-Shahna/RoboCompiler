#!/usr/bin/env python
"""Extract the excavator benchmark route from the validated Pinocchio nominal run.

Source: ``Excavator_Pinocchio_Validation_v1/results/nominal.npz`` and
``nominal.json`` (the closed-loop dig-lift-slew-dump cycle, 42 s, 10 ms
controller samples). Written: ``inputs/nominal_route.npz`` and
``inputs/nominal_route.json`` (provenance and SHA-256 of both sources).

Arrays (named coordinate order is stored with them):
* ``time`` (s) and ``phase`` (mission phase index 0..10);
* ``desired`` - the original SoilMission reference of the seven joint-space
  independent coordinates q23 q7 q4 q0 q1 q21 q22 (the benchmark input route);
* ``measured`` - the plant's measured independent coordinates (the measured-signal routes);
* ``tree`` - the plant's closed 23-coordinate tree (named order below), used as
  an independent start state and for branch identification.
No value is modified; only the plant's depth-first tree order is renamed.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

JOINT_SPACE = ['q23', 'q7', 'q4', 'q0', 'q1', 'q21', 'q22']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validated', required=True, help='Excavator_Pinocchio_Validation_v1 directory')
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'inputs'))
    a = p.parse_args()
    root = Path(a.validated); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    npz, meta_path = root / 'results/nominal.npz', root / 'results/nominal.json'
    meta = json.loads(meta_path.read_text())
    order = list(meta['plant_tree_order'])
    with np.load(npz) as z:
        data = dict(time=z['time'].copy(), phase=z['phase'].copy(), desired=z['desired'].copy(),
                    measured=z['u'].copy(), tree=z['q'].copy())
    if data['desired'].shape[1] != 7 or data['tree'].shape[1] != len(order):
        raise ValueError('Unexpected nominal trace layout')
    np.savez_compressed(out / 'nominal_route.npz', **data, independent_ids=np.array(JOINT_SPACE),
                        tree_ids=np.array(order))
    record = dict(
        description='Benchmark route: original SoilMission reference (desired) of the nominal validated run',
        source_package='Excavator_Pinocchio_Validation_v1 (26 September 2026)',
        source_files={'results/nominal.npz': sha(npz), 'results/nominal.json': sha(meta_path)},
        case=meta['case'], spec=meta['spec'], samples=int(len(data['time'])),
        sample_period_s=float(data['time'][1] - data['time'][0]), duration_s=float(data['time'][-1]),
        phase_names=meta.get('phase_names'), independent_ids=JOINT_SPACE, tree_ids=order,
        note='desired is the benchmark input; measured and tree are the closed-loop plant record '
             '(tree is used for the common initial state and as an independent branch reference).',
        written_sha256=sha(out / 'nominal_route.npz'))
    (out / 'nominal_route.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'spec'}, indent=2))


if __name__ == '__main__':
    main()
