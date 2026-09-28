#!/usr/bin/env python
"""Audit hashes, scalar gates, aggregates and saved-state replay of a Kangaroo benefit run.

Usage (from benchmark/):  python audit_results.py --results results_here --replay full --out audit_local

The replay uses the independent NumPy routes and the unchanged sources; it never
modifies the result directory.  Native Pinocchio values are audited for
consistency with the saved records (and recomputed if Pinocchio is importable
and --native is given), not silently replaced.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json
import math
from pathlib import Path

import numpy as np

from src import bootstrap  # noqa: F401
from src.bootstrap import ROOT
from src.compiler import compile_graph
from src.evaluator import GeneratedSupportGraph
from src.io_utils import save_json, save_csv, read_csv, sha, quantile, clean
from src.physics import NumpyReference, dynamics_witness, curvature_ablation, normmax, sole_frame_records
from src.runner import Acceptance, Layout, accepted, supplied_reference, digest, TANGENT_GATE

RAW = ['pipeline_raw.csv', 'verification_tests.csv', 'evaluator_raw.csv', 'warm_raw.csv', 'derivative_raw.csv',
       'incremental_raw.csv', 'acquisition_raw.csv', 'dynamics_raw.csv', 'curvature_ablation.csv']
SUMMARIES = [('evaluator_raw.csv', 'evaluator_summary.csv', ['method']),
             ('warm_raw.csv', 'warm_summary.csv', ['method']),
             ('derivative_raw.csv', 'derivative_summary.csv', ['method']),
             ('incremental_raw.csv', 'incremental_summary.csv', ['changed_modules', 'method']),
             ('acquisition_raw.csv', 'acquisition_summary.csv', ['method'])]


def audit(results, replay, out, native=False):
    results, out = Path(results), Path(out)
    checks, recomputed = [], []

    def check(name, passed, **extra):
        checks.append(dict(check=name, passed=bool(passed), **extra))

    manifest = json.loads((results / 'RUN_MANIFEST.json').read_text())
    for name, value in manifest.items():
        check('result hash: ' + name, (results / name).is_file() and sha(results / name) == value)
    for name, value in json.loads((results / 'source_hashes.json').read_text()).items():
        check('executed source: ' + name, (ROOT / name).is_file() and sha(ROOT / name) == value)
    v22 = json.loads((ROOT / 'original/original_v22/MANIFEST_SHA256.json').read_text())['files']
    for path in sorted((ROOT / 'original/original_v22').rglob('*')):
        if path.is_file() and path.name != 'MANIFEST_SHA256.json':
            rel = path.relative_to(ROOT / 'original/original_v22').as_posix()
            check('unchanged supplied v22 file: ' + rel, v22.get(rel) == sha(path))
    origin = json.loads((ROOT / 'original/ORIGIN_VERIFICATION.json').read_text())
    for name, value in origin['kangaroo_pin_files'].items():
        check('unchanged delivered file: ' + name, sha(ROOT / 'original' / name) == value)
    summary = json.loads((results / 'summary.json').read_text())
    check('protocol hash', sha(results / 'protocol.json') == summary['protocol_sha256'])
    check('completed status', summary['status'] == 'completed')
    counts = {}
    for name in RAW:
        path = results / name
        if not path.exists():
            continue
        rows = read_csv(path)
        counts[name] = len(rows)
        check('all success flags: ' + name, all(r.get('success') == 'True' for r in rows), records=len(rows))
        bad = []
        for i, r in enumerate(rows):
            for k, v in r.items():
                if not v or k in ('state_digest', 'error', 'message'):
                    continue  # identifiers and text (a hex digest such as 5053e93 would parse as a float)
                try:
                    number = float(v)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    bad.append((i, k))
        check('finite numerical fields: ' + name, not bad)
        if rows and 'max_gap_m' in rows[0]:
            check('independent closure gap gate 1e-8 m: ' + name,
                  all(float(r['max_gap_m']) <= 1e-8 for r in rows if r['max_gap_m']))
        if rows and 'tangent_residual_independent' in rows[0]:
            check(f'independent tangent gate {TANGENT_GATE:g}: ' + name,
                  all(float(r['tangent_residual_independent']) <= TANGENT_GATE for r in rows
                      if r['tangent_residual_independent']))
        if rows and 'known_branch_error' in rows[0]:
            check('known-branch gate 2e-6: ' + name,
                  all(float(r['known_branch_error']) <= 2e-6 for r in rows if r['known_branch_error']))
    for rawname, summaryname, groups in SUMMARIES:
        if not (results / rawname).exists():
            continue
        raw = read_csv(results / rawname)
        for s in read_csv(results / summaryname):
            rr = [r for r in raw if all(r[g] == s[g] for g in groups)]
            v = np.array([float(r['time_ms']) for r in rr])
            tag = rawname + ': ' + ','.join(s[g] for g in groups)
            check('attempt count ' + tag, len(rr) == int(s['attempts']))
            check('accepted count ' + tag, sum(r['success'] == 'True' for r in rr) == int(s['accepted']))
            check('median ' + tag, abs(np.median(v) - float(s['median_ms'])) <= 1e-9 * max(1., abs(np.median(v))))
            check('p95 ' + tag, abs(quantile(v, .95) - float(s['p95_ms'])) <= 1e-9 * max(1., quantile(v, .95)))
            check('mean ' + tag, abs(np.mean(v) - float(s['mean_ms'])) <= 1e-9 * max(1., np.mean(v)))
    if (results / 'warm_raw.csv').exists():
        raw = read_csv(results / 'warm_raw.csv')
        groups = {}
        for r in raw:
            groups.setdefault((r['method'], r['sample']), set()).add(r['state_digest'])
        check('warm repeats bitwise identical per method/sample', all(len(v) == 1 for v in groups.values()),
              groups=len(groups))
    if (results / 'curvature_ablation.csv').exists():
        raw = read_csv(results / 'curvature_ablation.csv')
        for case in sorted({r['case'] for r in raw}, key=int):
            rr = {float(r['speed_scale']): r for r in raw if r['case'] == case}
            one = float(rr[1.]['omitted_residual'])
            check('omitted curvature scales with speed squared: case ' + case,
                  all(abs(float(r['omitted_residual']) - s * s * one) <= 1e-6 * max(1., s * s * one)
                      for s, r in rr.items()))
    if (results / 'rollout_comparison.json').exists():
        c = json.loads((results / 'rollout_comparison.json').read_text())
        with np.load(results / 'rollout_accepted_cutgraph.npz') as a, \
                np.load(results / 'rollout_generated_evaluator.npz') as g:
            check('rollout base difference recomputed',
                  abs(float(np.max(np.abs(a['base'] - g['base']))) - c['max_base_pose_difference']) == 0.)
            check('rollout motor difference recomputed',
                  abs(float(np.max(np.abs(a['motor'] - g['motor']))) - c['max_motor_difference_m']) == 0.)
            check('rollout stored-state difference <= 1e-6', c['max_stored_state_difference'] <= 1e-6)
            check('rollout repeated runs bitwise identical', c['repeat_runs_bitwise_identical'])
            check('rollout mean wall times recomputed',
                  all(abs(np.mean(v) - c['mean_elapsed_s'][k]) == 0. for k, v in c['elapsed_s'].items()))
    replayed = {}
    if replay != 'none':
        from threadpoolctl import threadpool_limits
        acc = accepted()
        comp = compile_graph(json.loads((ROOT / 'inputs/physical_graph.json').read_text()))
        check('compiled plan reproduced', json.loads((results / 'compiled_plan.json').read_text())
              == json.loads(json.dumps(clean(comp.plan))))
        acceptance = Acceptance(comp)
        nt = comp.plan['physical_coordinates']
        with threadpool_limits(limits=1):
            if (results / 'warm_states.npz').exists():
                inputs = json.loads((results / 'warm_inputs.json').read_text())
                active = np.array(inputs['active'])
                raw = {(r['method'], int(r['sample'])): r for r in read_csv(results / 'warm_raw.csv')
                       if r['repeat'] == '0'}
                layout = Layout(comp, acc)
                ref = supplied_reference()
                truth = {i: layout(ref['qaug'][k])[:nt] for i, k in zip(
                    inputs['known_branch_samples'], [int(round(inputs['time'][i] / .025))
                                                     for i in inputs['known_branch_samples']])}
                with np.load(results / 'warm_states.npz') as z:
                    q_all, meth, samp = z['q'], z['method'], z['sample']
                    Nm, Ns, NN = z['N_method'], z['N_sample'], z['N']
                maps = {(str(m), int(s)): N for m, s, N in zip(Nm, Ns, NN)}
                ids = np.arange(len(q_all)) if replay == 'full' else np.arange(0, len(q_all), 7)
                bad = gap_max = 0
                digests = 0
                for i in ids:
                    key = (str(meth[i]), int(samp[i]))
                    q = q_all[i]
                    geo = acceptance.ref.loop_geometry(q)
                    gap = max(normmax(geo['point_gap']), normmax(geo['universal_residual']))
                    gap_max = max(gap_max, gap)
                    good = gap <= 1e-8 and normmax(q[acceptance.motors] - active[key[1]]) <= 1e-12
                    if key in maps:
                        r = acceptance(q, maps[key], active[key[1]], truth.get(key[1]))
                        good = good and r['success']
                        digests += digest(q, maps[key]) == raw[key]['state_digest']
                    bad += not good
                check('independent saved warm-state replay', bad == 0, records=len(ids), max_gap_m=gap_max)
                check('warm state digests recomputed', digests == sum(1 for i in ids
                                                                      if (str(meth[i]), int(samp[i])) in maps))
                replayed['warm_states'] = int(len(ids))
            if (results / 'incremental_states.npz').exists():
                inputs = read_csv(results / 'incremental_raw.csv')
                raw = {(r['method'], r['condition'], int(r['block']), int(r['step'])): r for r in inputs
                       if r['repeat'] == '0'}
                bad = 0
                with np.load(results / 'incremental_states.npz') as z:
                    for m, c, b, s, q, N in zip(z['method'], z['condition'], z['block'], z['step'], z['q'], z['N']):
                        r = raw[(str(m), str(c), int(b), int(s))]
                        bad += digest(q, N) != r['state_digest']
                        geo = acceptance.ref.loop_geometry(q)
                        bad += max(normmax(geo['point_gap']), normmax(geo['universal_residual'])) > 1e-8
                        bad += normmax(geo['jacobian'] @ N) > TANGENT_GATE
                    replayed['incremental_states'] = int(len(z['q']))
                check('incremental saved-state replay (digest, gap, tangent)', bad == 0)
            if (results / 'dynamics_states.json').exists():
                states = json.loads((results / 'dynamics_states.json').read_text())
                recorded = {int(r['case']): r for r in read_csv(results / 'dynamics_raw.csv')}
                chart = NumpyReference(comp.chart_cmg())
                for s in states:
                    r, _ = dynamics_witness(comp, np.array(s['x']), s['sites'], np.array(s['active_velocity']),
                                            np.array(s['motors']), np.array(s['wrench']), native=native,
                                            chart_ref=chart)
                    old = recorded[int(s['case'])]
                    same = abs(r['relative_acceleration_difference']
                               - float(old['relative_acceleration_difference'])) <= 1e-12
                    check(f'dynamics replay {s["case"]}', r['success'] and same)
                    recomputed.append(dict(case=s['case'], **{k: v for k, v in r.items()
                                                              if not isinstance(v, (list, dict))}))
                replayed['dynamics_states'] = len(states)
            if (results / 'curvature_inputs.json').exists():
                states = json.loads((results / 'curvature_inputs.json').read_text())
                chart = NumpyReference(comp.chart_cmg())
                count = 0
                for s in states:
                    for speed in (.5, 1., 2., 4.):
                        full, _ = curvature_ablation(comp, np.array(s['x']), s['sites'],
                                                     np.array(s['active_velocity']) * speed,
                                                     np.array(s['active_acceleration']), chart)
                        check(f'curvature replay {s["case"]} x{speed}', full < 2e-6)
                        count += 1
                replayed['curvature_cases'] = count
    result = dict(passed=all(r['passed'] for r in checks), checks=len(checks),
                  passed_checks=sum(r['passed'] for r in checks), raw_record_counts=counts, replay=replay,
                  replayed=replayed,
                  note='Hash and numerical consistency checks are heterogeneous, not independent experiments.',
                  details=checks)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'audit.json', result)
    save_csv(out / 'checks.csv', checks)
    if recomputed:
        save_csv(out / 'recomputed_dynamics.csv', recomputed)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results_here')
    p.add_argument('--replay', choices=['none', 'sampled', 'full'], default='sampled')
    p.add_argument('--out', default='audit_local')
    p.add_argument('--native', action='store_true', help='also recompute native values (requires Pinocchio)')
    a = p.parse_args()
    raise SystemExit(audit(a.results, a.replay, a.out, a.native))
