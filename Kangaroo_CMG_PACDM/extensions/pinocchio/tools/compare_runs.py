"""Compare a rerun case with a recorded case: every NPZ array and every summary field.

Standard library + NumPy only.  Timing fields (``elapsed_s``) are reported,
not compared.  Example::

    python tools/compare_runs.py recorded/results rerun/results landing_nominal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TIMING = {'elapsed_s'}


def compare(a_dir, b_dir, name):
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    with np.load(a_dir / f'{name}.npz', allow_pickle=False) as x, np.load(b_dir / f'{name}.npz', allow_pickle=False) as y:
        keys = sorted(set(x.files) | set(y.files))
        arrays = {}
        for key in keys:
            if key not in x.files or key not in y.files:
                arrays[key] = dict(identical=False, reason='missing in one file')
                continue
            u, v = x[key], y[key]
            same = u.shape == v.shape and u.dtype == v.dtype and np.array_equal(u, v)
            record = dict(identical=bool(same), shape=list(u.shape))
            if not same and u.shape == v.shape and np.issubdtype(u.dtype, np.number):
                record['max_absolute_difference'] = float(np.max(np.abs(u.astype(float) - v.astype(float))))
            arrays[key] = record
    s = json.loads((a_dir / f'{name}.json').read_text())
    t = json.loads((b_dir / f'{name}.json').read_text())
    fields = {k: s.get(k) == t.get(k) for k in sorted(set(s) | set(t)) if k not in TIMING}
    return dict(case=name, all_arrays_bit_identical=all(v['identical'] for v in arrays.values()),
                all_summary_fields_identical=all(fields.values()),
                differing_arrays=[k for k, v in arrays.items() if not v['identical']],
                differing_summary_fields=[k for k, v in fields.items() if not v],
                elapsed_s=dict(first=s.get('elapsed_s'), second=t.get('elapsed_s')), arrays=arrays)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first')
    parser.add_argument('second')
    parser.add_argument('case')
    args = parser.parse_args()
    result = compare(args.first, args.second, args.case)
    print(json.dumps({k: v for k, v in result.items() if k != 'arrays'}, indent=2))
    raise SystemExit(0 if result['all_arrays_bit_identical'] and result['all_summary_fields_identical'] else 1)
