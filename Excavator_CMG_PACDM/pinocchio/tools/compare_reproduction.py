"""Compare a reproduction run with the shipped results.

Usage (from the package root, after ``python run_validation.py --results OTHER_DIR``):
    python tools/compare_reproduction.py results OTHER_DIR [--output reproduction.json]

For every case the NPZ traces are compared array by array (bitwise equality and
the largest absolute difference), and every numeric field of the case, mechanics,
route, cross-engine and validation JSON files is compared, ignoring wall-clock
timings, time stamps and file paths. The verdicts of all checks must agree.
Videos are compared by SHA-256 (informational: video encoders need not be
bit-reproducible).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

IGNORED = ('wall', 'elapsed', 'finished_utc', 'measured_control', 'setup_wall', 'total_wall', 'audit_mode',
           'environment', 'video_sha256', 'video', 'poster')


def _flatten(value, prefix=''):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten(item, f'{prefix}{key}.'))
    elif isinstance(value, list) and value and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                   for v in value):
        out[prefix[:-1]] = np.asarray(value, dtype=float)
    elif isinstance(value, list):
        for k, item in enumerate(value):
            out.update(_flatten(item, f'{prefix}{k}.'))
    else:
        out[prefix[:-1]] = value
    return out


def compare_json(a_path, b_path):
    a, b = _flatten(json.loads(Path(a_path).read_text())), _flatten(json.loads(Path(b_path).read_text()))
    keys = sorted(set(a) | set(b))
    worst, differing, missing = 0., [], []
    for key in keys:
        if any(token in key for token in IGNORED):
            continue
        if key not in a or key not in b:
            missing.append(key)
            continue
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            if x.shape != y.shape:
                differing.append(key)
                continue
            diff = float(np.max(np.abs(x - y))) if x.size else 0.
            if not np.array_equal(x, y):
                differing.append(key)
                worst = max(worst, diff if np.isfinite(diff) else np.inf)
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if x != y:
                differing.append(key)
                worst = max(worst, abs(float(x) - float(y)))
        elif x != y:
            differing.append(key)
    return dict(fields=len(keys), differing=differing, missing=missing, max_numeric_difference=worst)


def compare_npz(a_path, b_path):
    with np.load(a_path, allow_pickle=False) as fa, np.load(b_path, allow_pickle=False) as fb:
        keys = sorted(set(fa.files) | set(fb.files))
        worst, unequal = 0., []
        for key in keys:
            if key not in fa.files or key not in fb.files:
                unequal.append(key)
                continue
            x, y = fa[key], fb[key]
            if x.shape != y.shape or not np.array_equal(x, y):
                unequal.append(key)
                if x.shape == y.shape and x.dtype.kind in 'fiu':
                    worst = max(worst, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    return dict(arrays=len(keys), unequal_arrays=unequal, max_abs_difference=worst,
                file_sha256_equal=hashlib.sha256(Path(a_path).read_bytes()).hexdigest()
                == hashlib.sha256(Path(b_path).read_bytes()).hexdigest())


def main(shipped, reproduced, output=None):
    shipped, reproduced = Path(shipped), Path(reproduced)
    report = dict(shipped=str(shipped), reproduced=str(reproduced), npz={}, json={}, videos={})
    for path in sorted(shipped.glob('*.npz')):
        other = reproduced / path.name
        report['npz'][path.name] = compare_npz(path, other) if other.exists() else 'missing'
    for path in sorted(shipped.glob('*.json')):
        other = reproduced / path.name
        report['json'][path.name] = compare_json(path, other) if other.exists() else 'missing'
    for path in sorted(shipped.glob('*.mp4')):
        other = reproduced / path.name
        report['videos'][path.name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                       == hashlib.sha256(other.read_bytes()).hexdigest()) if other.exists() else 'missing'
    a = json.loads((shipped / 'validation.json').read_text())
    b = json.loads((reproduced / 'validation.json').read_text())
    verdicts = {k: (a['checks'][k]['passed'], b['checks'].get(k, {}).get('passed')) for k in a['checks']}
    report['verdicts'] = dict(shipped_passed=a['passed'], reproduced_passed=b['passed'],
                              checks=len(verdicts), disagreeing=[k for k, v in verdicts.items() if v[0] != v[1]],
                              only_in_reproduction=sorted(set(b['checks']) - set(a['checks'])))
    npz_ok = all(isinstance(v, dict) and not v['unequal_arrays'] for v in report['npz'].values())
    report['summary'] = dict(
        all_npz_arrays_identical=npz_ok,
        all_npz_files_byte_identical=all(isinstance(v, dict) and v['file_sha256_equal']
                                         for v in report['npz'].values()),
        json_files_with_differences={k: v['differing'][:10] for k, v in report['json'].items()
                                     if isinstance(v, dict) and v['differing']},
        verdicts_agree=not report['verdicts']['disagreeing'] and not report['verdicts']['only_in_reproduction'],
        videos_byte_identical=all(v is True for v in report['videos'].values()))
    text = json.dumps(report, indent=2, default=str)
    if output:
        Path(output).write_text(text + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    return 0 if npz_ok and report['summary']['verdicts_agree'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('shipped')
    parser.add_argument('reproduced')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    sys.exit(main(args.shipped, args.reproduced, args.output))
