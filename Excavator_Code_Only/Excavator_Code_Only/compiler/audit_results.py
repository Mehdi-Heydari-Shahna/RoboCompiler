#!/usr/bin/env python
"""Audit a result directory: hashes, every gate from the raw records, aggregates,
determinism across repeats, and an independent replay of saved states.

    python audit_results.py --results results_here --replay full --out audit_local

The saved-state replay uses the original NumPy-only ``SourceBiasDynamics`` two-point
closures on the original spanning tree (no Pinocchio, no PACDM, no generated code). The
route inputs (truth projections and cylinder commands) are rebuilt and compared bit for
bit, and the 48 dynamics witnesses are re-run with the shipped code. Check counts are
heterogeneous consistency checks, not independent experiments.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import bootstrap  # noqa: E402,F401
from src.io_utils import read_csv, save_json, save_csv, sha  # noqa: E402
from src.runner import digest, PHASE_GROUPS, ROUTES  # noqa: E402

GATES = dict(max_gap_m=1e-8, target_error=1e-12, tangent_residual_ref=1e-8, map_error=1e-8, known_branch_error=1e-6)
TOL = 1e-12


def quantile(values, q):
    return float(np.quantile(np.asarray(values, float), q))


def close(a, b):
    return abs(float(a) - float(b)) <= TOL * max(1., abs(float(b)))


def audit(results, replay, out):
    results, out = Path(results), Path(out)
    checks = []

    def check(name, passed, **extra):
        checks.append(dict(check=name, passed=bool(passed), **extra))

    manifest = json.loads((results / 'RUN_MANIFEST.json').read_text())
    for name, value in manifest.items():
        check('result hash: ' + name, (results / name).is_file() and sha(results / name) == value)
    listed = {str(f.relative_to(results)).replace('\\', '/') for f in results.rglob('*') if f.is_file()}
    check('no unlisted result files', listed - set(manifest) == {'RUN_MANIFEST.json'},
          unlisted=sorted(listed - set(manifest) - {'RUN_MANIFEST.json'}))
    original = json.loads((ROOT / 'original/ORIGINAL_HASHES.json').read_text())
    for name, value in original['files'].items():
        check('unchanged original: ' + name, sha(ROOT / 'original' / name) == value['sha256'])
    for name, value in json.loads((results / 'source_hashes.json').read_text()).items():
        check('executed source: ' + name, (ROOT / name).is_file() and sha(ROOT / name) == value)
    summary = json.loads((results / 'summary.json').read_text())
    protocol = json.loads((results / 'protocol.json').read_text())
    check('protocol hash', sha(results / 'protocol.json') == summary['protocol_sha256'])
    check('completed status', summary['status'] == 'completed' and not summary['failed_stages'])
    check('route definitions equal the shipped code', protocol['routes'] == json.loads(json.dumps(ROUTES)))

    def rows(name):
        for candidate in (name, name + '.gz'):
            if (results / candidate).exists():
                return read_csv(results / candidate)
        return None

    def finite_fields(tag, data):
        bad = 0
        for r in data:
            for k, v in r.items():
                if not v or k in ('error', 'message', 'expected') or k.endswith('_digest'):   # hex identifiers
                    continue
                try:
                    x = float(v)
                except ValueError:
                    continue
                bad += not math.isfinite(x)
        check('finite numerical fields: ' + tag, bad == 0, bad=bad)

    def gate_checks(tag, data):
        check('all success flags: ' + tag, all(r['success'] == 'True' and not r['error'] for r in data), records=len(data))
        finite_fields(tag, data)
        for key, limit in GATES.items():
            vals = [float(r[key]) for r in data if r.get(key) not in (None, '')]
            check(f'gate {key} <= {limit:g}: {tag}', len(vals) == len(data) and max(vals) <= limit, maximum=max(vals))
        check('passive rank 16: ' + tag, all(r['passive_rank'] == '16' for r in data))
        check('inside branch window: ' + tag, all(r['inside_branch_window'] == 'True' for r in data))

    def summary_checks(tag, data, stored, groups):
        buckets = {}
        for r in data:
            buckets.setdefault(tuple(r[g] for g in groups), []).append(r)
        seen = set()
        for s in stored:
            key = tuple(s[g] for g in groups)
            seen.add(key)
            rr = buckets.get(key, [])
            v = np.array([float(r['time_ms']) for r in rr])
            label = f'{tag}: ' + ','.join(key)
            ok = (len(rr) == int(s['attempts']) == int(s['accepted']) and len(rr) > 0
                  and close(np.median(v), s['median_ms']) and close(quantile(v, .95), s['p95_ms'])
                  and close(v.min(), s['min_ms']) and close(v.max(), s['max_ms'])
                  and int(s['no_solve_attempts']) == sum(1 for r in rr if r.get('solved_modules') == '0'))
            check('aggregate ' + label, ok)
        check(f'aggregate coverage {tag}', seen == set(buckets), groups=len(buckets))

    # ---------------------------------------------------------- per-attempt gates
    counts = {}
    warm = []
    for route in protocol['routes']:
        data = rows(f'warm_raw/{route}.csv')
        check(f'raw route file present: {route}', data is not None)
        if data is None:
            continue
        counts[f'warm_raw/{route}.csv.gz'] = len(data)
        gate_checks(f'warm {route}', data)
        warm += data
    for name in ('local_raw.csv', 'derivative_raw.csv'):
        data = rows(name)
        if data is None:
            continue
        counts[name] = len(data)
        gate_checks(name, data)
    ws = summary['stages'].get('warm', {})
    check('warm attempts equal the recorded stage count', len(warm) == ws.get('attempts') == ws.get('passed'),
          attempts=len(warm))

    # ---------------------------------------------------------- aggregates
    for r in warm:
        r['phase_group'] = next(g for g, ph in PHASE_GROUPS.items() if int(r['phase']) in ph)
    if warm:
        summary_checks('warm_summary', warm, read_csv(results / 'warm_summary.csv'), ('route', 'partition', 'method'))
        summary_checks('warm_per_repeat', warm, read_csv(results / 'warm_per_repeat.csv'),
                       ('route', 'partition', 'method', 'repeat'))
        summary_checks('warm_phase_summary', warm, read_csv(results / 'warm_phase_summary.csv'),
                       ('route', 'partition', 'phase_group', 'method'))
        sums, no_solve = {}, {}
        for r in warm:
            key = (r['route'], r['method'], r['repeat'])
            sums[key] = sums.get(key, 0.) + float(r['time_ms'])
            if r['repeat'] == '0' and r['solved_modules'] == '0':
                no_solve[(r['route'], r['method'])] = no_solve.get((r['route'], r['method']), 0) + 1
        totals = read_csv(results / 'warm_route_totals.csv')
        for s in totals:
            per = [sums[(s['route'], s['method'], str(rep))] / 1e3 for rep in range(int(s['repeats']))]
            check(f"route total {s['route']},{s['method']}",
                  close(np.median(per), s['median_total_s']) and close(min(per), s['min_total_s'])
                  and close(max(per), s['max_total_s'])
                  and int(s['samples_without_solve']) == no_solve.get((s['route'], s['method']), 0))
        check('route totals cover every route and method', len(totals) == len({(k[0], k[1]) for k in sums}))
        # determinism: identical states in every repeat
        by = {}
        for r in warm:
            by.setdefault((r['route'], r['method'], r['sample']), set()).add(r['state_digest'])
        different = sum(len(v) > 1 for v in by.values())
        check('bit-identical states across repeats (all routes, methods, samples)', different == 0, differing=different,
              groups=len(by))
    local = rows('local_raw.csv')
    if local:
        summary_checks('local_summary', local, read_csv(results / 'local_summary.csv'), ('partition', 'condition', 'method'))
    der = rows('derivative_raw.csv')
    if der:
        summary_checks('derivative_summary', der, read_csv(results / 'derivative_summary.csv'), ('partition', 'method'))
    # benefit and attribution summaries recomputed from the summaries
    from src.reporting import load, benefit_rows, attribution_rows  # noqa: E402
    d = load(results)
    stored = read_csv(results / 'benefit_summary.csv')
    again = benefit_rows(d)
    check('benefit summary recomputed', len(stored) == len(again) and all(
        a['comparison'] == b['comparison'] and a['reference'] == b['reference'] and a['proposed'] == b['proposed']
        and close(a['reduction_percent'], b['reduction_percent']) for a, b in zip(stored, again)), rows=len(stored))
    stored = read_csv(results / 'attribution_summary.csv')
    again = attribution_rows(d)
    check('attribution summary recomputed', len(stored) == len(again) and all(
        close(a['exclusion_effect_percent'], b['exclusion_effect_percent'])
        and close(a['decomposition_effect_percent'], b['decomposition_effect_percent']) for a, b in zip(stored, again)),
          rows=len(stored))
    # other raw files
    for name in ('pipeline_raw.csv', 'verification_tests.csv', 'dynamics_raw.csv', 'curvature_magnitude.csv'):
        data = rows(name)
        if data is None:
            continue
        counts[name] = len(data)
        check('all success flags: ' + name, all(r['success'] == 'True' for r in data), records=len(data))
        finite_fields(name, data)
    pipe = rows('pipeline_raw.csv')
    if pipe:
        check('pipeline gates', all(float(r['native_closure_gap_m']) < 1e-9 and float(r['fd_jacobian_error']) < 1e-6
                                    and float(r['sparsity_zero_error']) == 0. and float(r['modular_vs_global_map']) < 1e-9
                                    and float(r['map_vs_native']) < 1e-9 and float(r['tangent_residual_native']) < 1e-9
                                    and float(r['min_reduced_inertia_eigenvalue']) > 0 for r in pipe))
    tests = rows('verification_tests.csv')
    if tests:
        rejections = [r for r in tests if r['expected'].startswith('reject: ')]
        check('designed rejections raised the expected message',
              all(r['expected'][len('reject: '):] in r['message'] for r in rejections), rejections=len(rejections))
    dyn = rows('dynamics_raw.csv')
    if dyn:
        body = [r for r in dyn if r['partition'] != 'invariance']
        check('dynamics gates', all(float(r['pacdm_vs_kkt_force']) < 1e-10 and float(r['pacdm_vs_native_force']) < 1e-10
                                    and float(r['pacdm_constraint_acceleration_m_s2']) < 2e-6
                                    and float(r['pacdm_tangent_dynamics_relative']) < 1e-9 for r in body))
        inv = [r for r in dyn if r['partition'] == 'invariance']
        check('partition invariance gates', all(float(r['partition_acceleration_force']) < 1e-10
                                                and float(r['partition_velocity_relative']) < 1e-10 for r in inv))

    # ---------------------------------------------------------- independent replays
    replayed = {}
    if replay != 'none':
        from src.context import Context  # noqa: E402
        from src.runner import route_truth, route_inputs  # noqa: E402
        from source_bias import SourceBiasDynamics  # noqa: E402
        ctx = Context()
        numpy_ref = SourceBiasDynamics(ctx.cmg, ctx.ref_ids)
        cut_ids = list(ctx.mapping['cut_joint_ids'])
        with np.load(results / 'route_inputs.npz', allow_pickle=False) as z:
            inputs = {k: z[k].copy() for k in z.files}
        # route inputs rebuilt from the shipped route: truth projections and commands, bit for bit
        ctx._truth = route_truth(ctx, ctx.joint_route())
        ctx._truth_measured = route_truth(ctx, np.asarray(ctx.route['measured'], float))
        for label, again, key in (('planned reference', ctx._truth, 'truth_desired'),
                                  ('measured signal', ctx._truth_measured, 'truth_measured')):
            diff = float(np.max(np.abs(again - inputs[key])))
            check(f'route truth reproduced ({label})', diff <= 1e-12, max_difference=diff,
                  bit_identical=bool(np.array_equal(again, inputs[key])))
        targets, _, deviations = route_inputs(ctx)
        for name, value in targets.items():
            diff = float(np.max(np.abs(value - inputs[f'targets_{name}'])))
            check(f'route commands reproduced: {name}', diff <= 1e-12, max_difference=diff,
                  bit_identical=bool(np.array_equal(value, inputs[f'targets_{name}'])))
            check(f'held-command deviation reproduced: {name}',
                  deviations[name] < 1e-12 and ws['routes'][name]['held_command_max_deviation'] < 1e-12,
                  deviation=deviations[name], recorded=ws['routes'][name]['held_command_max_deviation'])
        with np.load(results / 'warm_states.npz', allow_pickle=False) as z:
            keys = [str(x) for x in z['state_digest']]
            Q, NN = z['q'], z['N']
            index = dict(zip(keys, zip(z['q_index'], z['N_index'])))
        check('stored state digests', all(digest(Q[qi], NN[ni]) == k for k, (qi, ni) in index.items()), states=len(index))
        truth = {'desired': inputs['truth_desired'], 'measured': inputs['truth_measured']}
        records = [r for r in warm if r['state_digest'] in index]
        if replay == 'sampled':
            records = records[::10]
        cache, worst = {}, dict(gap=0., tangent=0., target=0., branch=0.)
        bad = 0
        for r in records:
            key = (r['route'], r['state_digest'], r['sample'])
            if key in cache:
                continue
            qi, ni = index[r['state_digest']]
            q, N = Q[qi], NN[ni]
            ev = numpy_ref.evaluate(q, np.zeros(len(q)), [0., 0., -9.81])
            res, J, _ = numpy_ref.closure(ev, cut_ids)
            spec = ROUTES[r['route']]
            active = ctx.reference.active[spec['partition']]
            target = inputs[f"targets_{r['route']}"][int(r['sample'])]
            known = truth[spec['signal']][int(r['sample'])]
            m = dict(gap=float(np.max(np.abs(res))), tangent=float(np.max(np.abs(J @ N))),
                     target=float(np.max(np.abs(q[active] - target))), branch=float(np.max(np.abs(q - known))))
            cache[key] = m
            bad += not (m['gap'] <= 1e-8 and m['tangent'] <= 1e-8 and m['target'] <= 1e-12 and m['branch'] <= 1e-6)
            for k in worst:
                worst[k] = max(worst[k], m[k])
        check('independent NumPy replay of saved route states', bad == 0 and len(cache) > 0, attempts=len(warm),
              attempts_with_stored_state=len(records), unique_replays=len(cache),
              **{f'max_{k}': v for k, v in worst.items()})
        replayed['route_attempts_total'] = len(warm)
        replayed['route_attempts_with_stored_state'] = len(records)
        replayed['route_unique_replays'] = len(cache)
        replayed['stored_states'] = len(index)
        # dynamics witnesses
        states = json.loads((results / 'dynamics_states.json').read_text())
        if states and dyn:
            from src.physics import Dynamics  # noqa: E402
            D = Dynamics(ctx)
            stored_rows = {r['case']: r for r in dyn if r['partition'] == 'joint_space'}
            recomputed, worst_force, worst_diff = [], 0., 0.
            for s in states:
                q32 = ctx.full_state(np.array(s['tree']))
                data, _ = D.witness('joint_space', q32, np.array(s['ua']), np.array(s['motors']), np.array(s['wrench']))
                recomputed.append(dict(case=s['case'], **data))
                worst_force = max(worst_force, data['pacdm_vs_kkt_force'], data['pacdm_vs_native_force'])
                ref = stored_rows[str(s['case'])]
                for k in ('pacdm_vs_kkt_force', 'pacdm_vs_native_force', 'pacdm_vs_native_relative'):
                    worst_diff = max(worst_diff, abs(data[k] - float(ref[k])))
            check('dynamics witnesses re-run (joint space)', worst_force < 1e-10, witnesses=len(recomputed),
                  max_force=worst_force, max_difference_to_stored_metrics=worst_diff)
            save_csv(out / 'recomputed_dynamics.csv', recomputed)
            replayed['dynamics_states'] = len(recomputed)
    result = dict(passed=all(c['passed'] for c in checks), checks=len(checks), passed_checks=sum(c['passed'] for c in checks),
                  raw_record_counts=counts, replay=replay, replayed=replayed,
                  note='Heterogeneous hash, gate, aggregate and replay checks; not independent experiments.', details=checks)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'audit.json', result)
    save_csv(out / 'checks.csv', checks)
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))
    if not result['passed']:
        for c in checks:
            if not c['passed']:
                print('FAILED:', c)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results_here')
    p.add_argument('--replay', choices=['none', 'sampled', 'full'], default='full')
    p.add_argument('--out', default='audit_local')
    a = p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):          # the same BLAS setting as the benchmark run
        code = audit(a.results, a.replay, a.out)
    raise SystemExit(code)
