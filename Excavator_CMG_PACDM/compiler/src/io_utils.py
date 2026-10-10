from __future__ import annotations
from pathlib import Path
import csv, gzip, hashlib, io, json, platform, sys, subprocess, os, importlib.metadata
import numpy as np


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _open(path, mode):
    path = Path(path)
    if path.suffix == '.gz':
        # mtime=0 keeps the compressed bytes deterministic for identical content
        return io.TextIOWrapper(gzip.GzipFile(filename=str(path), mode=mode[0] + 'b', mtime=0),
                                encoding='utf-8', newline='')
    return path.open(mode, newline='', encoding='utf-8')


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with _open(path, 'wt') as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: clean(v) for k, v in r.items()})


def read_csv(path):
    with _open(path, 'rt') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    cpu = platform.processor()
    source = 'platform.processor'
    lscpu = None
    if sys.platform.startswith('linux'):
        try:
            info = Path('/proc/cpuinfo').read_text()
            cpu = next(l.split(':', 1)[1].strip() for l in info.splitlines() if l.startswith('model name'))
            source = '/proc/cpuinfo (host-exposed identification; may be virtualized)'
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
    for pkg in ('numpy', 'scipy', 'threadpoolctl', 'pin'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    try:
        import pinocchio as pin
        versions['pinocchio_imported'] = pin.__version__
    except ImportError:
        versions['pinocchio_imported'] = None
    from threadpoolctl import threadpool_info
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
                processor_model=cpu or None, processor_source=source, visible_logical_cpus=os.cpu_count(), lscpu=lscpu,
                versions=versions, threadpools=threadpool_info(),
                note='Host-reported CPU identification is not a dedicated bare-metal allocation. Timing repeats are descriptive.')


def timing_summary(rows, groups=('method',)):
    """Per-group attempt counts and step-time statistics (one pass over the rows)."""
    buckets = {}
    for r in rows:
        buckets.setdefault(tuple(r[k] for k in groups), []).append(r)
    result = []
    for key in sorted(buckets, key=str):
        a = buckets[key]
        v = np.array([r['time_ms'] for r in a], float)
        out = dict(zip(groups, key))
        out.update(attempts=len(a), accepted=sum(bool(r['success']) for r in a),
                   median_ms=float(np.median(v)), p95_ms=float(np.quantile(v, .95)),
                   min_ms=float(v.min()), max_ms=float(v.max()),
                   max_gap_m=max((r.get('max_gap_m', 0.) or 0. for r in a), default=0.),
                   median_evaluations=float(np.median([r.get('evaluations', 0) or 0 for r in a])),
                   median_solved_modules=float(np.median([r.get('solved_modules', 0) or 0 for r in a])),
                   no_solve_attempts=int(sum(1 for r in a if r.get('solved_modules') == 0)),
                   fallback_count=int(sum(bool(r.get('fallback', False)) for r in a)))
        result.append(out)
    return result
