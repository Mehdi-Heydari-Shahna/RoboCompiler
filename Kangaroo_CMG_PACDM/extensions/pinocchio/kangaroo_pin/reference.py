"""MuJoCo-free regeneration of the accepted PACDM contact-task reference.

The accepted ``contact_reference.build_reference`` (foot-pose IK through the
unchanged PACDM mapping, whole-body feedforward from the accepted NumPy
source dynamics) is executed verbatim through ``legacy.load`` with its output
root redirected to ``results/reference_rebuild``.  The simulations use the
stored ``original_v22/data/contact_reference.npz``; this module shows that
the stored reference is reproduced by the accepted code without MuJoCo.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .legacy import SOURCE, load

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild(output=None):
    out = ROOT / 'results' / 'reference_rebuild' if output is None else Path(output)
    if out.exists():
        shutil.rmtree(out)
    (out / 'data').mkdir(parents=True)
    (out / 'results').mkdir(parents=True)
    shutil.copy(SOURCE / 'data/configurations.npz', out / 'data/configurations.npz')
    started = time.perf_counter()
    accepted = load(reference_root=out)
    accepted.build_reference()
    elapsed = time.perf_counter() - started
    stored = SOURCE / 'data/contact_reference.npz'
    rebuilt = out / 'data/contact_reference.npz'
    comparison = {}
    with np.load(stored, allow_pickle=False) as a, np.load(rebuilt, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('Rebuilt reference has different arrays')
        for key in sorted(a.files):
            x, y = a[key], b[key]
            same_shape = x.shape == y.shape
            comparison[key] = dict(
                bit_identical=bool(same_shape and np.array_equal(x, y)),
                max_absolute_difference=float(np.max(np.abs(x - y))) if same_shape and x.size else 0.)
    summary = json.loads((out / 'results/contact_reference.json').read_text())
    summary.pop('elapsed_seconds', None)
    result = dict(stored_sha256=_sha(stored), rebuilt_sha256=_sha(rebuilt),
                  file_bit_identical=_sha(stored) == _sha(rebuilt),
                  all_arrays_bit_identical=all(v['bit_identical'] for v in comparison.values()),
                  maximum_array_difference=max(v['max_absolute_difference'] for v in comparison.values()),
                  arrays=comparison, rebuilt_summary=summary, elapsed_s=elapsed,
                  method='accepted contact_reference.build_reference executed verbatim via legacy.load; MuJoCo not imported')
    # Keep only the comparison record; the rebuilt arrays duplicate the stored file.
    (out / 'data/contact_reference.npz').unlink()
    (out / 'data/configurations.npz').unlink()
    shutil.rmtree(out)
    return result
