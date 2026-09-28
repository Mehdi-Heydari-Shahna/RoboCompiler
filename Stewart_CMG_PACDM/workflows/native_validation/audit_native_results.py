#!/usr/bin/env python3
"""Audit the recorded native Stewart comparison without changing the original run.

Default: Python standard-library checks of hashes, complete row coverage,
finite values, numerical gates, aggregates and the curvature-ablation table.
--code: also compare the 23 recorded source hashes with the original package.
--source-updates: explicitly recognize the documented presentation-exporter update.
--replay: also recompute every saved state using the supplied NumPy backend.
--replay-native: additionally rerun native Pinocchio; requires its installation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

GATES = {
    'relative_acceleration_difference': 1e-8,
    'body_velocity_difference': 1e-9,
    'body_acceleration_difference': 2e-6,
    'physical_acceleration_constraint_m_s2': 2e-6,
    'virtual_power_defect_W': 1e-9,
    'reduced_mass_relative': 1e-9,
    'native_acceleration_relative': 1e-7,
    'native_mass_inf': 1e-8,
    'native_bias_inf': 1e-7,
}
METRICS = [
    ('native_acceleration_relative', 'PACDM vs native Pinocchio acceleration', 'normalized infinity discrepancy'),
    ('native_mass_inf', 'NumPy vs native Pinocchio mass matrix', 'max absolute entry; mixed coordinate SI'),
    ('native_bias_inf', 'NumPy vs native Pinocchio bias vector', 'max absolute entry; mixed N and Nm'),
    ('relative_acceleration_difference', 'PACDM vs NumPy KKT acceleration', 'normalized infinity discrepancy'),
    ('body_acceleration_difference', 'Compiled vs original-chart body acceleration', 'max absolute scalar component; mixed m/s2 and rad/s2'),
    ('body_velocity_difference', 'Compiled vs original-chart body velocity', 'max absolute scalar component; mixed m/s and rad/s'),
    ('physical_acceleration_constraint_m_s2', 'Physical point acceleration-constraint residual', 'm/s2; infinity norm'),
    ('virtual_power_defect_W', 'Virtual-power consistency defect', 'W'),
    ('reduced_mass_relative', 'Compiled vs original-chart reduced inertia', 'normalized max absolute entry'),
]

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows_csv(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))

def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def save_csv(path: Path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def boolean(s):
    if s not in ('True', 'False'):
        raise ValueError(f'Invalid recorded boolean: {s!r}')
    return s == 'True'

def safe_join(root: Path, rel: str):
    path = (root / rel.replace('\\', '/')).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe manifest path: {rel}')
    return path

def validate_source_updates(document):
    """Validate the single permitted presentation-only source update."""
    updates = document.get('updates') if isinstance(document, dict) else None
    if not isinstance(updates, dict) or set(updates) != {'make_paper_assets.py'}:
        raise ValueError('Source updates may list only make_paper_assets.py')
    entry = updates['make_paper_assets.py']
    required = {'original_sha256', 'release_sha256', 'reason'}
    if not isinstance(entry, dict) or set(entry) != required:
        raise ValueError('Presentation update requires original/release SHA-256 and reason')
    for field in ('original_sha256', 'release_sha256'):
        if not isinstance(entry[field], str) or not re.fullmatch(r'[0-9a-f]{64}', entry[field]):
            raise ValueError('Presentation update requires lowercase SHA-256 digests')
    if not isinstance(entry['reason'], str) or not entry['reason'].strip():
        raise ValueError('Presentation update requires a nonempty reason')
    return updates


def source_hash_verdict(name, recorded_digest, actual_digest, updates):
    """Match original bytes, or the one explicitly declared presentation update."""
    exact = actual_digest is not None and actual_digest == recorded_digest
    entry = updates.get(name)
    if entry is not None:
        # The exact filename remains guarded here as well as at metadata loading.
        permitted = (name == 'make_paper_assets.py'
                     and recorded_digest == entry['original_sha256']
                     and actual_digest == entry['release_sha256'])
        return dict(passed=bool(permitted), exact_match=bool(exact),
                    documented_update=bool(permitted and not exact),
                    recorded_sha256=recorded_digest, actual_sha256=actual_digest,
                    reason=entry['reason'])
    return dict(passed=bool(exact), exact_match=bool(exact),
                documented_update=False, recorded_sha256=recorded_digest,
                actual_sha256=actual_digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results_native')
    parser.add_argument('--out', type=Path, default=Path('native_audit_local'))
    parser.add_argument('--code', type=Path, help='Root of the supplied Stewart_Framework_Benefits code package')
    parser.add_argument('--source-updates', type=Path, help='Explicit presentation_source_update.json; requires --code')
    parser.add_argument('--replay', action='store_true', help='Recompute all saved states using NumPy; requires --code')
    parser.add_argument('--replay-native', action='store_true', help='Also execute Pinocchio on all saved states; requires --code')
    args = parser.parse_args()
    if (args.replay or args.replay_native) and not args.code:
        parser.error('--replay and --replay-native require --code')
    if args.source_updates and not args.code:
        parser.error('--source-updates requires --code')
    updates = validate_source_updates(read_json(args.source_updates)) if args.source_updates else {}
    p = args.results.resolve(); out = args.out.resolve()
    if out == p or out.is_relative_to(p):
        parser.error('--out must be outside the original results directory')
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    def check(name, ok, detail=None):
        checks.append(dict(check=name, passed=bool(ok), detail=detail))

    summary = read_json(p / 'summary.json'); protocol = read_json(p / 'protocol.json')
    status = read_json(p / 'native_status.json'); manifest = read_json(p / 'RUN_MANIFEST.json')
    source_hashes = read_json(p / 'source_hashes.json')
    raw = rows_csv(p / 'dynamics_raw.csv'); curv = rows_csv(p / 'curvature_ablation.csv')
    states = read_json(p / 'dynamics_states.json'); section = summary.get('native', summary.get('dynamics'))
    check('Execution completed and all_verification_passed', summary['status'] == 'completed' and summary['all_verification_passed'])
    check('Native-only requested stage', summary['stages'] == ['native'] and summary['native_request'] == 'required')
    check('Native stage actually recorded complete', status['status'] == 'completed' and section['native'] == status, status)
    check('Canonical protocol hash', canonical(protocol) == summary['protocol_sha256'] == manifest['protocol_sha256'])
    check('Two recorded source hash dictionaries agree', source_hashes == manifest['source_sha256'])
    for rel, digest in manifest['files'].items():
        f = safe_join(p, rel)
        check('Result hash: ' + rel, f.is_file() and sha(f) == digest)
    source_files_exact_match = 0
    documented_presentation_updates = []
    if args.code:
        if updates and not set(updates).issubset(source_hashes):
            raise ValueError('Documented presentation update is absent from recorded source hashes')
        for rel, digest in source_hashes.items():
            f = safe_join(args.code, rel)
            verdict = source_hash_verdict(rel, digest, sha(f) if f.is_file() else None, updates)
            source_files_exact_match += int(verdict['exact_match'])
            if verdict['documented_update']:
                documented_presentation_updates.append(dict(file=rel, **verdict))
            check('Source hash: ' + rel, verdict['passed'], verdict)

    n = protocol[summary['profile']]['dynamics_samples']
    check('All declared witness records are present', len(raw) == section['attempts'] == n == len(states), dict(expected=n, rows=len(raw), states=len(states)))
    check('Sample IDs are unique and complete', [int(r['sample']) for r in raw] == list(range(n)))
    check('Recorded pass count matches raw rows', sum(boolean(r['success']) for r in raw) == section['passed'] == n)
    numeric = [k for k in raw[0] if k not in ('success', 'error')]
    for r in raw:
        i = int(r['sample'])
        finite = all(math.isfinite(float(r[k])) for k in numeric)
        passed = finite and all(float(r[k]) < threshold for k, threshold in GATES.items()) and float(r['min_reduced_mass_eigenvalue']) > 0
        check(f'Witness {i}: independently re-evaluated gates', passed and boolean(r['success']) and not r['error'])
        check(f'Witness {i}: prescribed sample time', math.isclose(float(r['time_s']), .4 + (21.6-.4)*i/(n-1), abs_tol=1e-12))
    for key, value in section['maxima'].items():
        computed = max(float(r[key]) for r in raw)
        check('Summary maximum: ' + key, math.isclose(computed, value, abs_tol=0, rel_tol=1e-14), computed)
    metric_rows = []
    for key, label, units in METRICS:
        values = [float(r[key]) for r in raw]; maximum = max(values)
        metric_rows.append(dict(metric=label, raw_column=key, maximum=maximum, acceptance_limit=GATES[key],
                                status='PASS' if maximum < GATES[key] else 'FAIL', units=units,
                                maximum_sample=int(raw[values.index(maximum)]['sample']), source='results_native/dynamics_raw.csv'))

    check('Curvature row count matches summary', len(curv) == section['curvature_ablations'] == 4*min(12,n))
    check('Curvature full-pass count', sum(boolean(r['success']) for r in curv) == section['curvature_full_passed'] == len(curv))
    by_pair = {(int(r['sample']), float(r['speed_scale'])): r for r in curv}
    expected_pairs = {(i, s) for i in range(min(12,n)) for s in (.5, 1., 2., 4.)}
    check('Curvature is 12 states x 4 scales, not 48 independent states', len(by_pair) == len(curv) and set(by_pair) == expected_pairs)
    for (sample, speed), r in by_pair.items():
        full = float(r['full_curvature_residual_m_s2']); omitted = float(r['omitted_curvature_residual_m_s2'])
        baseline = float(by_pair[(sample,1.)]['omitted_curvature_residual_m_s2'])
        check(f'Curvature {sample}/{speed}: full constraint gate', math.isfinite(full) and math.isfinite(omitted) and full < 2e-6 and boolean(r['success']))
        check(f'Curvature {sample}/{speed}: omitted defect speed-squared', math.isclose(omitted, baseline*speed*speed, rel_tol=1e-10, abs_tol=1e-12))
    curve_rows = []
    for speed in (.5,1.,2.,4.):
        rr = [r for r in curv if float(r['speed_scale']) == speed]
        full = [float(r['full_curvature_residual_m_s2']) for r in rr]
        omit = [float(r['omitted_curvature_residual_m_s2']) for r in rr]
        curve_rows.append(dict(speed_multiplier=speed, configurations=len(rr), full_curvature_max_m_s2=max(full),
                               omitted_curvature_min_m_s2=min(omit), omitted_curvature_median_m_s2=statistics.median(omit),
                               omitted_curvature_max_m_s2=max(omit), full_limit_m_s2=2e-6,
                               interpretation='Intentional component ablation; not a competent external baseline'))

    replay_rows=[]; native_replayed=False
    if args.replay or args.replay_native:
        import numpy as np
        sys.path.insert(0,str(args.code.resolve()))
        from src.bootstrap import PRIOR
        from src.compiler import compile_graph, from_original_cmg
        from src.experiments import make_state, reduced_acceleration, kkt_acceleration, body_motion_errors
        from numpy_backend import NumpyTree
        from vendor.pacdm_original import PointGraph, PACDM
        original = read_json(PRIOR / 'original/data/stewart.cmg.json')
        source = from_original_cmg(original); compiled = compile_graph(source)
        reference = NumpyTree(original); backend = NumpyTree(compiled.cmg)
        q0, old0, _ = make_state(compiled, original, reference, 0.)
        graph = compiled.graph(q0); oldgraph = PointGraph(original, old0)
        oracle = None
        if args.replay_native:
            from stewart.pin_checks import PinConstraintOracle
            oracle = PinConstraintOracle(compiled.cmg)
        for i, state in enumerate(states):
            z = {k:np.asarray(v,dtype=float) for k,v in state.items()}
            acceleration, details = reduced_acceleration(graph, backend, graph.lift(z['q']), z['active_velocity'], z['forces'], z['wrench'])
            kkt, kd = kkt_acceleration(backend,z['q'],details['v'],z['forces'],z['wrench'])
            relative = float(np.max(abs(acceleration-kkt))/max(1.,np.max(abs(kkt))))
            saved_difference=float(np.max(abs(acceleration-z['acceleration'])))
            constraint=float(np.max(abs(details['geometry']['jacobian']@acceleration+details['geometry']['acceleration_bias'])))
            Nold, info = PACDM(oldgraph).mapping(oldgraph.lift(z['q_original']))
            vold=Nold[:oldgraph.nt]@z['active_velocity']; aold,_=kkt_acceleration(reference,z['q_original'],vold,z['forces'],z['wrench'])
            ve,ae=body_motion_errors(backend,z['q'],details['v'],acceleration,reference,z['q_original'],vold,aold,[b['id'] for b in source['bodies']])
            ok=bool(relative<1e-8 and saved_difference<1e-7 and constraint<2e-6 and ve<1e-9 and ae<2e-6 and info['success'])
            result=dict(sample=i, numpy_kkt_relative=relative, recomputed_vs_saved_acceleration_inf=saved_difference,
                        physical_constraint_m_s2=constraint, body_acceleration_difference=ae, passed=ok)
            if oracle is not None:
                an=oracle.acceleration(z['q'],details['v'],z['forces'],z['wrench'])
                reln=float(np.max(abs(an-acceleration))/max(1.,np.max(abs(an))))
                result['native_acceleration_relative']=reln; result['passed']=bool(ok and reln<1e-7)
            check(f'Recomputed physical state {i}', result['passed'], result)
            replay_rows.append(result)
        native_replayed=oracle is not None
        save_csv(out/'recomputed_states.csv',replay_rows)
    save_csv(out/'native_metrics.csv',metric_rows); save_csv(out/'curvature_summary.csv',curve_rows)
    save_csv(out/'row_checks.csv',[dict(check=r['check'],passed=r['passed']) for r in checks])
    package_versions={}
    for name in ('numpy','scipy'):
        try:package_versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:package_versions[name]=None
    result=dict(status='PASS' if all(r['passed'] for r in checks) else 'FAIL',
                checks=len(checks),passed=sum(r['passed'] for r in checks),
                original_results=str(p),
                native_record_status=status,recorded_native_witnesses=len(raw),native_reexecuted_in_this_audit=native_replayed,
                physical_states_recomputed=len(replay_rows),recorded_result_files=len(manifest['files']),
                source_files_compared=len(source_hashes) if args.code else 0,
                source_files_exact_match=source_files_exact_match,
                documented_presentation_update_count=len(documented_presentation_updates),
                documented_presentation_updates=documented_presentation_updates,
                auditor_environment=dict(python=platform.python_version(),platform=platform.platform(),packages=package_versions),
                checks_detail=checks)
    save_json(out/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks_detail'},indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, ImportError) as exc:
        print(f'Audit could not complete: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(2)
