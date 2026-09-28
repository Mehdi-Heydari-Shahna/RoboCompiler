"""JSON/CSV writers, environment record and timing aggregation.

Adapted from the Go2 benefit study's ``src/io_utils.py`` (same aggregation
rules: median, empirical P95 by linear interpolation, attempts/accepted).
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        raise ValueError('Non-finite value in a saved record')
    return x


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: clean(v) for k, v in r.items()})


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    cpu, source, lscpu = platform.processor(), 'platform.processor', None
    if sys.platform.startswith('linux'):
        try:
            info = Path('/proc/cpuinfo').read_text()
            cpu = next(l.split(':', 1)[1].strip() for l in info.splitlines() if l.startswith('model name'))
            source = '/proc/cpuinfo (host-exposed identification)'
            lscpu = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=3).stdout
        except (OSError, StopIteration, subprocess.SubprocessError):
            pass
    if sys.platform == 'win32' and not cpu:
        try:
            cpu = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            source = 'Windows CIM'
        except (OSError, subprocess.SubprocessError):
            pass
    versions = {}
    for p in ('numpy', 'scipy', 'threadpoolctl', 'pin'):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    try:
        import pinocchio as pin
        versions['pinocchio_imported'] = pin.__version__
    except ImportError:
        versions['pinocchio_imported'] = None
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), processor_model=cpu or None, processor_source=source,
                visible_logical_cpus=os.cpu_count(), lscpu=lscpu, versions=versions, threadpools=pools,
                mujoco_imported='mujoco' in sys.modules,
                note='Host-reported CPU identification is not a dedicated allocation. Timing repeats are descriptive.')


def quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def timing_summary(rows, groups=('method',)):
    result = []
    keys = sorted({tuple(r[k] for k in groups) for r in rows}, key=str)
    for key in keys:
        a = [r for r in rows if tuple(r[k] for k in groups) == key]
        v = np.array([r['time_ms'] for r in a])
        out = dict(zip(groups, key))
        out.update(attempts=len(a), accepted=sum(bool(r['success']) for r in a),
                   median_ms=float(np.median(v)), p95_ms=quantile(v, .95), mean_ms=float(np.mean(v)),
                   total_s=float(np.sum(v)) / 1000.,
                   max_gap_m=max((r.get('max_gap_m', 0.) or 0. for r in a), default=0.),
                   median_evaluations=float(np.median([r.get('evaluations', 0) or 0 for r in a])),
                   median_solved_modules=float(np.median([r.get('solved_modules', 0) or 0 for r in a])),
                   median_residual_calls=float(np.median([r.get('residual_calls', 0) or 0 for r in a])),
                   fallback_count=sum(bool(r.get('fallback', False)) for r in a))
        result.append(out)
    return result
