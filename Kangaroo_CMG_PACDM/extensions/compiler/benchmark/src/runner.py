"""Reproducible Kangaroo framework-benefit stages. Raw attempts, failures and states retained.

New extension code. The PACDM core (``original/original_v22/pacdm.py``) and the
accepted v22 ``CutGraph`` / ``polish`` definitions are used unchanged; the
compiler, generated evaluator, module scheduler, independent NumPy routes and
native chart-base oracle are identified as new code.
"""
from __future__ import annotations

import argparse
import json
import traceback
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from . import bootstrap  # noqa: F401
from .bootstrap import ROOT
from .compiler import compile_graph, variant_inputs, support_sets, export_physical, ModelError
from .evaluator import (GeneratedLoopGraph, GeneratedSupportGraph, ModularSolver, GlobalSolver, TRFOracle,
                        TRF_OPTIONS, METHODS, create_solver, counted)
from .physics import NumpyReference, dynamics_witness, curvature_ablation, normmax, sole_frame_records
from .io_utils import save_json, save_csv, sha, environment, timing_summary
from pacdm import PACDM, rank

DERIVATIVE_METHODS = ['accepted_analytic_dense', 'analytic_dense', 'fd3_dense', 'analytic_sparse', 'fd3_colored']
INCREMENTAL_METHODS = ['original_monolithic', 'compiled_monolithic', 'compiled_monolithic_reuse', 'compiled_modular',
                       'modular_no_reuse', 'trf_compiled_predictor', 'trf_modular_predictor']
ACQUISITION_METHODS = ['original_monolithic', 'compiled_monolithic', 'compiled_modular']
PROFILES = {
    'smoke': dict(route_samples=101, repeats=1, variants=2, witnesses=2, derivative_cases=4, increment_blocks=1,
                  increment_steps=4, dynamics_states=8, curvature_states=4, acquisition_states=2,
                  evaluator_states=4, rollout_duration_s=.1),
    'standard': dict(route_samples=501, repeats=1, variants=12, witnesses=3, derivative_cases=24,
                     increment_blocks=2, increment_steps=12, dynamics_states=24, curvature_states=8,
                     acquisition_states=6, evaluator_states=24, rollout_duration_s=2.),
    'full': dict(route_samples=1001, repeats=3, variants=12, witnesses=5, derivative_cases=48, increment_blocks=4,
                 increment_steps=12, dynamics_states=48, curvature_states=12, acquisition_states=12,
                 evaluator_states=48, rollout_duration_s=10.)}
SEEDS = dict(pipeline=2026092601, warm=2026092602, derivatives=2026092603, incremental=2026092604,
             dynamics=2026092605, acquisition=2026092606, evaluator=2026092607)
EVALUATOR_ROUTES = ['accepted_cutgraph', 'generated_global', 'generated_modules', 'generated_global_residual_only']
DERIVATIVE_BUDGET = 5000
# Independent tangent gate: the 24 cuts carry 16 redundant physical rows, and the unchanged PACDM corrector
# accepts at 1e-9 (selected rows) / 1e-8 (all rows). The exact tangent of such a state differs from the
# selected-row tangent by O(1e-8); 1e-6 is 100x the all-row tolerance and far below the O(1) residual of a
# wrong or incomplete map. Declared before the timing runs (see METHODS.md).
TANGENT_GATE = 1e-6
MODES = [[], ['left_sole'], ['right_sole'], ['left_sole', 'right_sole']]


def mode_name(sites):
    return 'floating_loops_only' if not sites else '+'.join(sites)


# ---------------------------------------------------------------------------
# Supplied data
# ---------------------------------------------------------------------------
def supplied_cmg():
    return json.loads((ROOT / 'original/original_v22/data/whole_body_cmg.json').read_text())


def supplied_reference():
    with np.load(ROOT / 'original/original_v22/data/contact_reference.npz', allow_pickle=False) as r:
        return {k: r[k].copy() for k in r.files}


def accepted():
    from kangaroo_pin import legacy
    return legacy.load()


class Layout:
    """Maps supplied v22 augmented states (supplied CMG order) into a compiled layout, by record id."""

    def __init__(self, comp, acc, cmg0=None):
        cmg0 = cmg0 or supplied_cmg()
        self.comp = comp
        g0 = acc.CutGraph(cmg0, np.asarray(cmg0['initial_seed']))
        gc = acc.CutGraph(comp.cmg, np.asarray(comp.cmg['initial_seed']))
        self.perm = np.array([cmg0['coordinate_ids'].index(k) for k in comp.cmg['coordinate_ids']], int)
        source = {c['id']: cols for c, cols in zip(cmg0['closures'], g0.chart_columns)}
        self.charts = [(np.array(cols), np.array(source[c['id']]))
                       for c, cols in zip(comp.cmg['closures'], gc.chart_columns)]
        self.n = gc.n

    def __call__(self, xo):
        x = np.zeros(self.n)
        x[:len(self.perm)] = np.asarray(xo)[self.perm]
        for dst, src in self.charts:
            x[dst] = xo[src]
        return x


def chart_base_state(ref, k, x):
    R = Rotation.from_rotvec(ref['rotvec'][k]).as_matrix()
    return np.r_[ref['base'][k], Rotation.from_matrix(R).as_euler('ZYX'), x]


# ---------------------------------------------------------------------------
# Independent acceptance (NumPy loop geometry; no PACDM rows, no generated evaluator)
# ---------------------------------------------------------------------------
class Acceptance:
    def __init__(self, comp):
        self.comp = comp
        self.ref = NumpyReference(comp.cmg)
        records = {j['id']: j for j in comp.cmg['joints']}
        ids = comp.cmg['coordinate_ids']
        self.lo = np.array([records[k]['limits']['lower'] for k in ids])
        self.hi = np.array([records[k]['limits']['upper'] for k in ids])
        self.motors = np.array(comp.plan['motor_indices'], int)
        self.passive = np.setdiff1d(np.arange(len(ids)), self.motors)
        self.expected_rank = comp.plan['physical_closure_rank_expected']

    def __call__(self, q, N, active, truth=None, branch_limit=2e-6):
        geo = self.ref.loop_geometry(q)
        J = geo['jacobian']
        gap = max(normmax(geo['point_gap']), normmax(geo['universal_residual']))
        tangent = normmax(J @ N)
        r = int(rank(J[:, self.passive]))
        motor = normmax(q[self.motors] - active)
        eye = normmax(N[self.motors] - np.eye(len(self.motors)))
        inside = bool(np.all(q >= self.lo - 1e-12) and np.all(q <= self.hi + 1e-12))
        out = dict(max_gap_m=gap, motor_error=motor, tangent_residual_independent=tangent, rank=r,
                   known_branch_error=None if truth is None else normmax(q - truth))
        ok = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(N)) and gap <= 1e-8 and motor <= 1e-12
                  and eye < 1e-12 and tangent <= TANGENT_GATE and r == self.expected_rank and inside)
        if truth is not None:
            ok = ok and out['known_branch_error'] <= branch_limit
        out['success'] = ok
        return out


# ---------------------------------------------------------------------------
# Stage 1: compiler coverage over twelve predefined variants
# ---------------------------------------------------------------------------
def _id_map(nominal, variant, key):
    nom = [r['id'] for r in nominal[key]]
    var = [r['id'] for r in variant[key]]
    if set(var) <= set(nom):
        return {k: k for k in var}
    return dict(zip(var, nom))  # positional (same record order, renamed ids)


def transfer_state(comp_nom, x_nom, comp_var, physical_nom, physical_var, variant, acc):
    """Known-feasible variant state from a nominal state (by record correspondence), then PACDM + polish."""
    jmap = _id_map(physical_nom, physical_var, 'joints')
    values = dict(zip(comp_nom.cmg['coordinate_ids'], x_nom[:comp_nom.plan['physical_coordinates']]))
    scale = float(variant.split('_')[-1]) if variant.startswith('geometry_scale_') else 1.
    kinds = {j['id']: j['type'] for j in comp_var.cmg['joints']}
    q = np.array([values[jmap[k]] * (scale if kinds[k] == 'prismatic' else 1.)
                  for k in comp_var.cmg['coordinate_ids']])
    graph = acc.CutGraph(comp_var.cmg, np.asarray(comp_var.cmg['initial_seed']))
    x = graph.lift(q)
    gen = GeneratedLoopGraph(comp_var)
    solver = PACDM(gen)
    xa, info = solver.acquire(x[gen.active], x)
    if not info['success']:
        raise RuntimeError(f'variant acquisition failed: {info.get("message")}')
    return acc.polish(gen, xa)


def pipeline_stage(source, cfg, out, acc):
    rows, compiles, states = [], [], []
    ref_data = supplied_reference()
    comp_nom = compile_graph(source)
    layout = Layout(comp_nom, acc)
    picks = np.linspace(20, 380, cfg['witnesses']).astype(int)
    for variant, physical in variant_inputs(source)[:cfg['variants']]:
        start = perf_counter()
        comp = compile_graph(physical)
        ms = (perf_counter() - start) * 1000
        directory = out / 'generated' / variant
        save_json(directory / 'physical_graph.json', physical)
        save_json(directory / 'compiled_cmg.json', comp.cmg)
        save_json(directory / 'plan.json', comp.plan)
        save_json(directory / 'support_plans.json', {mode_name(s): comp.support_plan(s) for s in support_sets(comp)})
        compiles.append(dict(variant=variant, compilation_ms=ms, loop_cuts=comp.plan['loop_cuts'],
                             modules=len(comp.plan['modules']), module_sizes=str(comp.plan['module_sizes']),
                             largest_module_block=comp.plan['largest_module_block'],
                             augmented_coordinates=comp.plan['augmented_coordinates'],
                             support_plans=len(support_sets(comp))))
        acc_graph = acc.CutGraph(comp.cmg, np.asarray(comp.cmg['initial_seed']))
        gen = GeneratedLoopGraph(comp)
        fixed = NumpyReference(comp.cmg)
        chart = NumpyReference(comp.chart_cmg())
        sparsity = np.asarray(comp.plan['passive_sparsity'])
        for w, k in enumerate(picks):
            x = transfer_state(comp_nom, layout(ref_data['qaug'][k]), comp, source, physical, variant, acc)
            states.append(dict(variant=variant, witness=w, source_index=int(k), x=x))
            r0, J0, _ = acc_graph.residual(x)
            r1, J1, _ = gen.residual(x)
            evaluator_error = max(normmax(r0 - r1), normmax(J0 - J1))
            h = 1e-7
            Jfd = np.zeros_like(J1)
            for j in range(gen.n):
                d = np.zeros(gen.n)
                d[j] = h
                Jfd[:, j] = (gen.residual_only(x + d) - gen.residual_only(x - d)) / (2 * h)
            fd_error = normmax(J1 - Jfd)
            zero_error = normmax(J1[:, gen.passive][sparsity == 0])
            N, info = PACDM(gen).mapping(x)
            ms_solver = ModularSolver(comp, x)
            q_mod, N_mod, _ = ms_solver.step(x[gen.active])
            geo = fixed.loop_geometry(x[:comp.plan['physical_coordinates']])
            gap = max(normmax(geo['point_gap']), normmax(geo['universal_residual']))
            core = dict(evaluator_vs_accepted=evaluator_error, fd_jacobian_error=fd_error,
                        sparsity_zero_error=zero_error, closure_gap_m=gap)
            core_ok = evaluator_error <= 1e-12 and fd_error <= 1e-6 and zero_error == 0. and gap <= 1e-8
            # Loop-closure plan (motors independent)
            nt = comp.plan['physical_coordinates']
            Np = N[:nt] if N is not None else np.full((nt, 12), np.nan)
            M = fixed.dynamics_terms(x[:nt], np.zeros(nt))[0]
            Jn = geo['jacobian']
            mapping_error = normmax(Np - N_mod)
            tangent = normmax(Jn @ Np)
            mr = float(np.linalg.eigvalsh(Np.T @ M @ Np).min())
            rows.append(dict(variant=variant, witness=w, plan='loop_closure', support_count=0,
                             rank=int(rank(Jn[:, gen.passive[gen.passive < nt]])),
                             expected_rank=comp.plan['physical_closure_rank_expected'], mobility=len(gen.active),
                             mapping_discrepancy=mapping_error, tangent_residual=tangent,
                             min_reduced_inertia_eigenvalue=mr, rcond=info['rcond'], **core,
                             success=bool(core_ok and info['success'] and mapping_error < 1e-9 and tangent < 1e-9
                                          and mr > 0)))
            # Ideal sole-support plans in chart-base coordinates
            xc = chart_base_state(ref_data, k, x)
            qc = xc[:6 + nt]
            Mc = chart.dynamics_terms(qc, np.zeros(6 + nt))[0]
            for sites in support_sets(comp):
                frames = sole_frame_records(comp, sites)
                anchors = chart.weld_geometry(qc, None, frames)['poses']
                g = GeneratedSupportGraph(comp, sites, anchors)
                Ns, si = PACDM(g).mapping(xc)
                Jc = np.vstack([chart.loop_geometry(qc)['jacobian'], chart.weld_geometry(qc, None, frames)['jacobian']])
                act = np.array([i for i in g.active if i < 6 + nt], int)
                dep = np.setdiff1d(np.arange(6 + nt), act)
                Ni = np.zeros((6 + nt, len(act)))
                Ni[act] = np.eye(len(act))
                Ni[dep] = -np.linalg.lstsq(Jc[:, dep], Jc[:, act], rcond=None)[0]
                if Ns is None:
                    rows.append(dict(variant=variant, witness=w, plan=mode_name(sites), support_count=len(sites),
                                     success=False, error=str(si.get('message')), **core))
                    continue
                Nsp = Ns[:6 + nt]
                me = normmax(Nsp - Ni)
                ta = normmax(Jc @ Nsp)
                mrs = float(np.linalg.eigvalsh(Nsp.T @ Mc @ Nsp).min())
                exp_rank = g.plan['expected_physical_constraint_rank']
                rk = int(rank(Jc))
                rows.append(dict(variant=variant, witness=w, plan=mode_name(sites), support_count=len(sites),
                                 rank=rk, expected_rank=exp_rank, mobility=len(g.active),
                                 mapping_discrepancy=me, tangent_residual=ta, min_reduced_inertia_eigenvalue=mrs,
                                 rcond=si['rcond'], **core,
                                 success=bool(core_ok and si['success'] and rk == exp_rank and me < 1e-9
                                              and ta < 1e-9 and mrs > 0)))
    save_csv(out / 'pipeline_raw.csv', rows)
    save_csv(out / 'compilation_times.csv', compiles)
    save_json(out / 'pipeline_states.json', states)
    return dict(variants=len(compiles), physical_models=len(compiles),
                plans=len(compiles) * (1 + len(support_sets(comp_nom))), checks=len(rows),
                passed=sum(bool(r['success']) for r in rows),
                median_compilation_ms=float(np.median([r['compilation_ms'] for r in compiles])),
                note='One loop-closure plan and three ideal sole-support plans per variant; known-feasible '
                     'configurations (not cold starts).')


# ---------------------------------------------------------------------------
# Stage 2: designed compiler/precondition/component checks
# ---------------------------------------------------------------------------
def fault_stage(source, out, acc):
    raw = []

    def reject(name, mutate):
        s = deepcopy(source)
        mutate(s)
        message, detected = '', False
        try:
            compile_graph(s)
        except ModelError as e:
            detected, message = True, str(e)
        raw.append(dict(test=name, expected='reject at compilation', success=detected, message=message))

    j0 = source['joints'][0]['id']
    reject('duplicate_body', lambda s: s['bodies'].append(deepcopy(s['bodies'][1])))
    reject('duplicate_joint', lambda s: s['joints'].append(deepcopy(s['joints'][0])))
    reject('negative_mass', lambda s: s['bodies'][1].update(mass_kg=-1.))
    reject('invalid_inertia', lambda s: s['bodies'][1].update(inertia_com_kg_m2=np.diag([1., 1., -1.]).tolist()))
    reject('missing_joint_endpoint', lambda s: s['joints'][0].update(body_a='missing'))
    reject('disconnected_graph', lambda s: s['joints'].pop())
    reject('nonunit_axis', lambda s: next(j for j in s['joints'] if j['type'] != 'fixed').update(axis=[2., 0., 0.]))
    reject('improper_transform', lambda s: s['joints'][0].update(T_AJ=np.zeros((4, 4)).tolist()))
    reject('reversed_limits', lambda s: next(j for j in s['joints'] if j['type'] != 'fixed').update(
        limits=dict(lower=1., upper=-1.)))
    reject('unsupported_joint', lambda s: s['joints'][0].update(type='spherical'))
    reject('duplicate_actuator', lambda s: s['actuators'].append(deepcopy(s['actuators'][0])))
    reject('actuator_on_missing_joint', lambda s: s['actuators'][0].update(joint_id='missing'))
    reject('invalid_sole_site_body', lambda s: s['sole_sites'][0].update(body='missing'))
    reject('preauthored_modules', lambda s: s.update(modules=[[0], [1]]))
    reject('preauthored_coordinate_order', lambda s: s.update(coordinate_ids=['x']))
    reject('missing_named_seed', lambda s: s['seed']['joints'].pop(next(iter(s['seed']['joints']))))
    reject('seed_outside_range', lambda s: s['seed']['joints'].update(
        {next(j['id'] for j in s['joints'] if j['type'] == 'revolute'): 10.}))
    reject('unsupported_cut_type', lambda s: s['loop_cuts'][0].update(type='gear'))
    reject('improper_universal_frame', lambda s: next(c for c in s['loop_cuts'] if c['type'] == 'universal').update(
        frame1_R=np.diag([1., 1., -1.]).tolist()))
    reject('cut_on_one_body', lambda s: s['loop_cuts'][0].update(body2=s['loop_cuts'][0]['body1']))
    hip = [c['id'] for c in source['loop_cuts'] if 'hip_yaw' in c['id']]
    reject('unconstrained_passive_coordinates', lambda s: s.update(
        loop_cuts=[c for c in s['loop_cuts'] if c['id'] not in hip]))
    comp = compile_graph(source)
    ref = supplied_reference()
    layout = Layout(comp, acc)
    x = layout(ref['qaug'][150])
    motors = np.array(comp.plan['motor_indices'])
    solver = ModularSolver(comp, x)
    _, _, i = solver.step(x[motors])
    raw.append(dict(test='unchanged_modules_reused', expected='six exact reuses', success=i['skipped_modules'] == 6,
                    message=str(i)))
    check = Acceptance(comp)
    target = x[motors].copy()
    for m_index, label in ((2, 'hip_yaw'), (1, 'hip_differential'), (0, 'knee_ankle')):
        mod = comp.plan['modules'][m_index]
        k = list(motors).index(mod['motors'][0])
        target[k] += 5e-4
        q, N, i = solver.step(target)
        ok = i['solved_modules'] == 1 and i['skipped_modules'] == 5 and check(q, N, target)['success']
        raw.append(dict(test=f'one_{label}_motor_change_recomputes_one_module', expected='1 solved, 5 reused',
                        success=bool(ok), message=str(i)))
    underactuated = deepcopy(source)
    underactuated['actuators'] = underactuated['actuators'][1:]
    cu = compile_graph(underactuated)
    info = PACDM(GeneratedLoopGraph(cu)).mapping(Layout(cu, acc)(ref['qaug'][150]))[1]
    raw.append(dict(test='missing_actuator_rejected_by_unchanged_rank_gate', expected='PACDM rank gate rejects',
                    success=not info['success'], message=str(info.get('message'))))
    s = deepcopy(source)
    for j in s['joints']:
        j['body_a'], j['body_b'] = j['body_b'], j['body_a']
        j['T_AJ'], j['T_BJ'] = j['T_BJ'], j['T_AJ']
        if j['type'] != 'fixed':
            j['axis'] = (-np.asarray(j['axis'])).tolist()
    cr = compile_graph(s)
    xr = Layout(cr, acc)(ref['qaug'][150])
    rr, Jr, _ = GeneratedLoopGraph(cr).residual(xr)
    ro, Jo, _ = GeneratedLoopGraph(comp).residual(x)
    same = cr.cmg['coordinate_ids'] == comp.cmg['coordinate_ids']
    raw.append(dict(test='reverse_edge_storage_preserves_evaluator', expected='same residual and Jacobian',
                    success=bool(same and normmax(rr - ro) < 1e-12 and normmax(Jr - Jo) < 1e-12),
                    message='all joint records stored reversed'))
    save_csv(out / 'verification_tests.csv', raw)
    return dict(tests=len(raw), passed=sum(bool(r['success']) for r in raw),
                note='Designed compiler/precondition/component checks; not a real-world failure detection rate.')


# ---------------------------------------------------------------------------
# Stage 3: full-route continuation
# ---------------------------------------------------------------------------
def route(cfg, comp):
    """Supplied v22 motor inputs u(t) (reordered by joint id into the compiled motor order), PCHIP."""
    ref = supplied_reference()
    cmg0 = supplied_cmg()
    ids = [comp.cmg['coordinate_ids'][i] for i in comp.plan['motor_indices']]
    order = [cmg0['independent_ids'].index(k) for k in ids]
    time = np.round(np.linspace(0., 10., cfg['route_samples']), 10)
    active = PchipInterpolator(ref['t'], ref['u'][:, order], axis=0)(time)
    known = {}
    for i, t in enumerate(time):
        k = np.flatnonzero(np.abs(ref['t'] - t) < 1e-9)
        if len(k):
            known[i] = int(k[0])
    return time, active, known


def digest(q, N):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(q, float).tobytes() + np.ascontiguousarray(N, float).tobytes()
                          ).hexdigest()[:16]


def _calls(solver):
    if isinstance(solver, ModularSolver):
        return sum(m['g'].calls for m in solver.modules)
    return getattr(solver.g, 'calls', 0)


def warm_stage(comp, cfg, out, acc):
    ref = supplied_reference()
    layout = Layout(comp, acc)
    time, active, known = route(cfg, comp)
    x0 = layout(ref['qaug'][0])
    check = Acceptance(comp)
    truth = {i: layout(ref['qaug'][k])[:comp.plan['physical_coordinates']] for i, k in known.items()}
    rng = np.random.default_rng(SEEDS['warm'])
    raw, records = [], []
    save_json(out / 'warm_inputs.json', dict(time=time, active=active, x_initial=x0, methods=METHODS,
                                             known_branch_samples=sorted(known),
                                             interpolation='scipy PchipInterpolator of supplied contact_reference u'))
    for repeat in range(cfg['repeats']):
        solvers = {name: create_solver(name, comp, x0, acc) for name in METHODS}
        for i, t in enumerate(time):
            moving = bool(i and normmax(active[i] - active[i - 1]) > 0.)
            for name in rng.permutation(METHODS):
                name = str(name)
                row = dict(method=name, repeat=repeat, sample=i, time_s=float(t), moving=moving, success=False,
                           error='')
                calls0 = _calls(solvers[name])
                t0 = perf_counter()
                try:
                    q, N, info = solvers[name].step(active[i])
                    row['time_ms'] = (perf_counter() - t0) * 1000
                    row['residual_calls'] = _calls(solvers[name]) - calls0
                    row.update(info)
                    row.update(check(q, N, active[i], truth.get(i)))
                    row['state_digest'] = digest(q, N)
                    if repeat == 0:
                        records.append((name, i, q, N if i % 10 == 0 else None))
                except Exception as e:  # retained as a failed attempt
                    row['time_ms'] = (perf_counter() - t0) * 1000
                    row['error'] = type(e).__name__ + ': ' + str(e)
                raw.append(row)
            if i % 200 == 0:
                print(f'warm repeat {repeat + 1}/{cfg["repeats"]}, sample {i}/{len(time)}', flush=True)
    save_csv(out / 'warm_raw.csv', raw)
    summary = timing_summary(raw)
    save_csv(out / 'warm_summary.csv', summary)
    save_csv(out / 'warm_moving_summary.csv', timing_summary([r for r in raw if r['moving']]))
    save_csv(out / 'warm_hold_summary.csv', timing_summary([r for r in raw if not r['moving']]))
    save_csv(out / 'warm_per_repeat.csv', timing_summary(raw, ('method', 'repeat')))
    np.savez_compressed(out / 'warm_states.npz', method=np.array([r[0] for r in records]),
                        sample=np.array([r[1] for r in records]), q=np.array([r[2] for r in records]),
                        N_method=np.array([r[0] for r in records if r[3] is not None]),
                        N_sample=np.array([r[1] for r in records if r[3] is not None]),
                        N=np.array([r[3] for r in records if r[3] is not None]))
    groups = {}
    for r in raw:
        if r.get('state_digest'):
            groups.setdefault((r['method'], r['sample']), set()).add(r['state_digest'])
    repeat_identical = all(len(v) == 1 for v in groups.values())
    return dict(route_samples=len(time), repeats=cfg['repeats'], duration_s=10., attempts=len(raw),
                repeats_bitwise_identical=bool(repeat_identical),
                saved_states='repeat 0: q for every attempt, N every 10th sample; digests for all attempts',
                passed=sum(bool(r['success']) for r in raw), moving_samples=int(sum(r['moving'] for r in raw
                                                                                   if r['method'] == METHODS[0]
                                                                                   and r['repeat'] == 0)),
                known_branch_samples=len(known), methods=summary)


# ---------------------------------------------------------------------------
# Stage 4: analytical vs numerical derivatives (corrector only)
# ---------------------------------------------------------------------------
def derivative_stage(comp, cfg, out, acc):
    ref = supplied_reference()
    layout = Layout(comp, acc)
    check = Acceptance(comp)
    n = cfg['derivative_cases']
    indices = np.linspace(58, 250, n).astype(int)
    rng = np.random.default_rng(SEEDS['derivatives'])
    raw, inputs = [], []
    sparsity = csr_matrix(np.asarray(comp.plan['passive_sparsity']))
    motors = np.array(comp.plan['motor_indices'])
    for case, k in enumerate(indices):
        x = layout(ref['qaug'][k])
        target = layout(ref['qaug'][k + 2])
        qa = target[motors]
        g = GeneratedLoopGraph(comp)
        N, info = PACDM(g).mapping(x)
        pred = x[g.passive] + N[g.passive] @ (qa - x[g.active])
        clipped = bool(np.any(pred <= g.lower[g.passive]) or np.any(pred >= g.upper[g.passive]))
        pred = np.clip(pred, g.lower[g.passive] + 1e-10, g.upper[g.passive] - 1e-10)
        truth = target[:comp.plan['physical_coordinates']]
        inputs.append(dict(case=case, source_index=int(k), target_index=int(k + 2), x_seed=x, active_target=qa,
                           passive_prediction=pred, prediction_clipped=clipped))
        for repeat in range(cfg['repeats']):
            for method in rng.permutation(DERIVATIVE_METHODS):
                method = str(method)
                analytic = 'analytic' in method
                sparse = method.endswith('sparse') or method.endswith('colored')
                graph = (acc.CutGraph(comp.cmg, np.asarray(comp.cmg['initial_seed'])) if method.startswith('accepted')
                         else GeneratedLoopGraph(comp))
                oracle = TRFOracle(graph, qa, analytic=analytic, sparse=sparse, budget=DERIVATIVE_BUDGET)
                kwargs = {}
                if not analytic and sparse:
                    kwargs['jac_sparsity'] = sparsity
                row = dict(method=method, case=case, repeat=repeat, source_index=int(k), success=False, error='')
                start = perf_counter()
                try:
                    res = least_squares(oracle.fun, pred.copy(), jac=oracle.jac if analytic else '3-point',
                                        bounds=(graph.lower[graph.passive], graph.upper[graph.passive]),
                                        tr_solver='lsmr' if sparse else 'exact', **TRF_OPTIONS, **kwargs)
                    row['time_ms'] = (perf_counter() - start) * 1000
                    row['evaluations'] = oracle.evaluations
                    full = oracle.full(res.x)
                    nn, mi = PACDM(graph).mapping(full)
                    nt = comp.plan['physical_coordinates']
                    c = check(full[:nt], nn[:nt] if nn is not None else np.full((nt, 12), np.nan), qa, truth)
                    row.update(c)
                    row['success'] = bool(c['success'] and mi['success'] and res.success)
                    row['scipy_nfev'] = int(res.nfev)
                except Exception as e:
                    row.setdefault('time_ms', (perf_counter() - start) * 1000)
                    row['evaluations'] = oracle.evaluations
                    row['error'] = type(e).__name__ + ': ' + str(e)
                raw.append(row)
        print(f'derivative case {case + 1}/{n}', flush=True)
    save_json(out / 'derivative_inputs.json', inputs)
    save_csv(out / 'derivative_raw.csv', raw)
    s = timing_summary(raw)
    save_csv(out / 'derivative_summary.csv', s)
    return dict(cases=n, repeats=cfg['repeats'], attempts=len(raw), passed=sum(bool(r['success']) for r in raw),
                fresh_evaluation_budget=DERIVATIVE_BUDGET, lookahead_s=.05, methods=s)


# ---------------------------------------------------------------------------
# Stage 5: localized motor-input updates (module reuse)
# ---------------------------------------------------------------------------
def incremental_stage(comp, cfg, out, acc):
    ref = supplied_reference()
    layout = Layout(comp, acc)
    check = Acceptance(comp)
    X = np.array([layout(x) for x in ref['qaug']])
    rng = np.random.default_rng(SEEDS['incremental'])
    raw, inputs, inc_states = [], [], []
    motors = np.array(comp.plan['motor_indices'])
    modules = comp.plan['modules']
    nt = comp.plan['physical_coordinates']
    leg = [[i for i, m in enumerate(modules) if all(prefix in comp.cmg['coordinate_ids'][c] for c in m['motors'])]
           for prefix in ('leg_left', 'leg_right')]
    if sorted(leg[0] + leg[1]) != list(range(len(modules))):
        raise RuntimeError('Could not assign loop modules to legs from motor ids')
    starts = np.linspace(70, 190, cfg['increment_blocks']).astype(int)
    conditions = {1: 'one_module', 3: 'one_leg_three_modules', 6: 'all_six_modules'}
    for count, label in conditions.items():
        for block, k0 in enumerate(starts):
            index = [int(k0)] * len(modules)
            state = X[k0].copy()
            targets = []
            for step in range(cfg['increment_steps']):
                changed = [step % 6] if count == 1 else (leg[step % 2] if count == 3 else list(range(6)))
                for m in changed:
                    index[m] += 2
                    cols = np.r_[modules[m]['motors'], modules[m]['passive']].astype(int)
                    state[cols] = X[index[m]][cols]
                target = state[motors].copy()
                targets.append((target, state[:nt].copy(), changed))
                inputs.append(dict(condition=label, changed_modules=count, block=block, step=step,
                                   start_index=int(k0), module_reference_index=list(index), changed=changed,
                                   active=target))
            for repeat in range(cfg['repeats']):
                solvers = {n: create_solver(n, comp, X[k0], acc) for n in INCREMENTAL_METHODS}
                for step, (target, truth, changed) in enumerate(targets):
                    for name in rng.permutation(INCREMENTAL_METHODS):
                        name = str(name)
                        row = dict(method=name, condition=label, changed_modules=count, block=block, step=step,
                                   repeat=repeat, success=False, error='')
                        calls0 = _calls(solvers[name])
                        start = perf_counter()
                        try:
                            q, N, info = solvers[name].step(target)
                            row['time_ms'] = (perf_counter() - start) * 1000
                            row['residual_calls'] = _calls(solvers[name]) - calls0
                            row.update(info)
                            row.update(check(q, N, target, truth))
                            row['state_digest'] = digest(q, N)
                            if repeat == 0:
                                inc_states.append((name, label, block, step, q, N))
                        except Exception as e:
                            row['time_ms'] = (perf_counter() - start) * 1000
                            row['error'] = type(e).__name__ + ': ' + str(e)
                        raw.append(row)
        print(f'incremental condition {label} done', flush=True)
    save_json(out / 'incremental_inputs.json', inputs)
    save_csv(out / 'incremental_raw.csv', raw)
    np.savez_compressed(out / 'incremental_states.npz', method=np.array([r[0] for r in inc_states]),
                        condition=np.array([r[1] for r in inc_states]), block=np.array([r[2] for r in inc_states]),
                        step=np.array([r[3] for r in inc_states]), q=np.array([r[4] for r in inc_states]),
                        N=np.array([r[5] for r in inc_states]))
    s = timing_summary(raw, ('changed_modules', 'method'))
    save_csv(out / 'incremental_summary.csv', s)
    return dict(attempts=len(raw), passed=sum(bool(r['success']) for r in raw), methods=s,
                note='Motor-input edits taken from the supplied reference per module; the supplied passive '
                     'solution of each module is the known branch. Unchanged module inputs are reused exactly.')


# ---------------------------------------------------------------------------
# Stage 6: cold acquisition from the named seed (defect homotopy)
# ---------------------------------------------------------------------------
def acquisition_stage(comp, cfg, out, acc):
    ref = supplied_reference()
    layout = Layout(comp, acc)
    check = Acceptance(comp)
    rng = np.random.default_rng(SEEDS['acquisition'])
    nt = comp.plan['physical_coordinates']
    motors = np.array(comp.plan['motor_indices'])
    seed_graph = counted(acc.CutGraph(comp.cmg, np.asarray(comp.cmg['initial_seed'])))
    seed = seed_graph.augment(np.asarray(comp.cmg['initial_seed']))
    indices = np.linspace(0, 400, cfg['acquisition_states']).astype(int)
    raw = []
    for case, k in enumerate(indices):
        target = layout(ref['qaug'][k])
        qa = target[motors]
        for repeat in range(cfg['repeats']):
            for name in rng.permutation(ACQUISITION_METHODS):
                name = str(name)
                row = dict(method=name, case=case, repeat=repeat, source_index=int(k), success=False, error='')
                start = perf_counter()
                try:
                    if name == 'compiled_modular':
                        x = np.zeros(comp.plan['augmented_coordinates'])
                        steps = 0
                        calls = 0
                        for mod in comp.plan['modules']:
                            g = GeneratedLoopGraph(comp, mod['cuts'], mod)
                            xm, info = PACDM(g).acquire(qa[[list(motors).index(m) for m in mod['motors']]],
                                                        seed[g.columns].copy())
                            if not info['success']:
                                raise RuntimeError(f'module acquisition failed: {info.get("message")}')
                            x[g.columns] = xm
                            steps += info['accepted_steps'] + info['rejected_steps']
                            calls += g.calls
                        modular = ModularSolver(comp, x)  # module mappings at the acquired state
                        q_out, N_out, _ = modular.step(qa)
                        calls += _calls(modular)
                        x = np.r_[q_out, x[nt:]]
                        N, mi = np.r_[N_out, np.zeros((len(x) - nt, len(motors)))], dict(success=True)
                    else:
                        g = seed_graph if name == 'original_monolithic' else GeneratedLoopGraph(comp)
                        g0 = getattr(g, 'calls', 0)
                        x, info = PACDM(g).acquire(qa, seed.copy())
                        if not info['success']:
                            raise RuntimeError(f'acquisition failed: {info.get("message")}')
                        steps = info['accepted_steps'] + info['rejected_steps']
                        calls = getattr(g, 'calls', 0) - g0
                        N, mi = PACDM(g).mapping(x)
                    row['time_ms'] = (perf_counter() - start) * 1000
                    row['homotopy_steps'] = steps
                    row['residual_calls'] = calls
                    row.update(check(x[:nt], N[:nt], qa, target[:nt], branch_limit=2e-6))
                    row['success'] = bool(row['success'] and mi['success'])
                except Exception as e:
                    row['time_ms'] = (perf_counter() - start) * 1000
                    row['error'] = type(e).__name__ + ': ' + str(e)
                raw.append(row)
        print(f'acquisition case {case + 1}/{len(indices)}', flush=True)
    save_csv(out / 'acquisition_raw.csv', raw)
    s = timing_summary(raw)
    save_csv(out / 'acquisition_summary.csv', s)
    return dict(attempts=len(raw), passed=sum(bool(r['success']) for r in raw), methods=s,
                note='Initialization from the named v22 seed with the unchanged PACDM defect homotopy; known '
                     'branch = supplied reference state. The mapping at the result is included in the time.')


# ---------------------------------------------------------------------------
# Stage 7: evaluator micro-benchmark (residual + Jacobian of all 24 cuts)
# ---------------------------------------------------------------------------
def evaluator_stage(comp, cfg, out, acc):
    ref = supplied_reference()
    layout = Layout(comp, acc)
    rng = np.random.default_rng(SEEDS['evaluator'])
    indices = np.linspace(58, 254, cfg['evaluator_states']).astype(int)
    accepted_graph = acc.CutGraph(comp.cmg, np.asarray(comp.cmg['initial_seed']))
    global_graph = GeneratedLoopGraph(comp)
    module_graphs = [GeneratedLoopGraph(comp, m['cuts'], m) for m in comp.plan['modules']]
    raw = []
    for case, k in enumerate(indices):
        x = layout(ref['qaug'][k])
        r0, J0, _ = accepted_graph.residual(x)
        r1, J1, _ = global_graph.residual(x)
        stacked = np.zeros_like(J0)
        rm = np.zeros_like(r0)
        for m, g in zip(comp.plan['modules'], module_graphs):
            r, J, _ = g.residual(x[g.columns])
            rows = np.array(m['rows'])
            rm[rows] = r
            stacked[np.ix_(rows, g.columns)] = J
        equality = dict(generated_vs_accepted=max(normmax(r1 - r0), normmax(J1 - J0)),
                        modules_vs_accepted=max(normmax(rm - r0), normmax(stacked - J0)),
                        residual_only_vs_accepted=normmax(global_graph.residual_only(x) - r0))
        for repeat in range(cfg['repeats']):
            for name in rng.permutation(EVALUATOR_ROUTES):
                name = str(name)
                start = perf_counter()
                for _ in range(10):
                    if name == 'accepted_cutgraph':
                        accepted_graph.residual(x)
                    elif name == 'generated_global':
                        global_graph.residual(x)
                    elif name == 'generated_modules':
                        for g in module_graphs:
                            g.residual(x[g.columns])
                    else:
                        global_graph.residual_only(x)
                elapsed = (perf_counter() - start) * 100.
                ok = True if name == 'accepted_cutgraph' else bool(max(equality.values()) <= 1e-12)
                raw.append(dict(method=name, case=case, repeat=repeat, source_index=int(k), time_ms=elapsed,
                                success=ok, reference=name == 'accepted_cutgraph', **equality))
    save_csv(out / 'evaluator_raw.csv', raw)
    s = timing_summary(raw)
    save_csv(out / 'evaluator_summary.csv', s)
    return dict(attempts=len(raw), passed=sum(bool(r['success']) for r in raw), methods=s,
                max_generated_vs_accepted=max(r['generated_vs_accepted'] for r in raw),
                max_modules_vs_accepted=max(r['modules_vs_accepted'] for r in raw),
                note='Mean of ten consecutive calls per record; residual and 144 x 140 Jacobian unless residual-only.')


# ---------------------------------------------------------------------------
# Stage 8: end-to-end delivered rollout with accepted vs generated evaluator
# ---------------------------------------------------------------------------
def rollout_stage(comp, cfg, out, acc):
    from .rollout import run_rollouts
    c = run_rollouts(comp, acc, supplied_cmg(), cfg['rollout_duration_s'], out)
    ok = bool(c['identical_step_count'] and c['max_stored_state_difference'] is not None
              and c['max_stored_state_difference'] <= 1e-6 and c['repeat_runs_bitwise_identical'])
    return dict(attempts=1, passed=int(ok), comparison=c)


# ---------------------------------------------------------------------------
# Stage 9: ideal-support dynamics witnesses and curvature ablation
# ---------------------------------------------------------------------------
def dynamics_stage(comp, cfg, out, native, acc):
    ref = supplied_reference()
    layout = Layout(comp, acc)
    chart_ref = NumpyReference(comp.chart_cmg())
    indices = np.linspace(8, 392, cfg['dynamics_states']).astype(int)
    rng = np.random.default_rng(SEEDS['dynamics'])
    raw, states, ablation, avinputs = [], [], [], []
    for case, k in enumerate(indices):
        sites = MODES[case % 4]
        x = chart_base_state(ref, k, layout(ref['qaug'][k]))
        plan = comp.support_plan(sites)
        va = rng.uniform(-.12, .12, len(plan['active']))
        motors = rng.uniform(-200., 200., 12)
        wrench = rng.uniform([-15, -15, -10, -2, -2, -2], [15, 15, 10, 2, 2, 2])
        row = dict(case=case, source_index=int(k), time_s=float(ref['t'][k]), mode=mode_name(sites),
                   support_count=len(sites), success=False, error='')
        try:
            result, state = dynamics_witness(comp, x, sites, va, motors, wrench, native=native, chart_ref=chart_ref)
            row.update(result)
            state.update(case=case, source_index=int(k))
            states.append(state)
            if case < cfg['curvature_states']:
                aactive = rng.uniform(-.3, .3, len(plan['active']))
                for speed in (.5, 1., 2., 4.):
                    full, omitted = curvature_ablation(comp, x, sites, va * speed, aactive, chart_ref)
                    ablation.append(dict(case=case, mode=mode_name(sites), support_count=len(sites),
                                         speed_scale=speed, full_residual=full, omitted_residual=omitted,
                                         success=bool(full < 2e-6)))
                avinputs.append(dict(case=case, x=x, sites=sites, active_velocity=va, active_acceleration=aactive))
        except Exception as e:
            row['error'] = type(e).__name__ + ': ' + str(e)
        raw.append(row)
        print(f'dynamics witness {case + 1}/{len(indices)} {mode_name(sites)} success={row["success"]}', flush=True)
    save_csv(out / 'dynamics_raw.csv', raw)
    save_json(out / 'dynamics_states.json', states)
    save_csv(out / 'curvature_ablation.csv', ablation)
    save_json(out / 'curvature_inputs.json', avinputs)
    cols = ['relative_acceleration_difference', 'absolute_acceleration_difference',
            'constraint_acceleration_residual', 'mapping_discrepancy', 'tangent_residual', 'virtual_power_defect_W',
            'reduced_inertia_relative', 'kinematic_curvature_constraint', 'kkt_force_balance', 'closure_gap']
    if native:
        cols += ['native_acceleration_relative', 'native_mass_inf', 'native_bias_inf',
                 'native_constraint_jacobian_inf', 'native_constraint_bias_inf']
    maxima = {c: max([r[c] for r in raw if r.get(c) is not None], default=None) for c in cols}
    ds = []
    for sites in MODES:
        rr = [r for r in raw if r['mode'] == mode_name(sites)]
        ds.append(dict(mode=mode_name(sites), support_count=len(sites), attempts=len(rr),
                       passed=sum(bool(r['success']) for r in rr),
                       expected_rank=comp.plan['physical_closure_rank_expected'] + 6 * len(sites),
                       expected_mobility=6 + len(comp.plan['motor_indices']) - 6 * len(sites),
                       native_acceleration_relative=max((r.get('native_acceleration_relative') or 0 for r in rr),
                                                        default=0),
                       acceleration_relative=max((r.get('relative_acceleration_difference', 0) for r in rr), default=0),
                       constraint_residual=max((r.get('constraint_acceleration_residual', 0) for r in rr), default=0),
                       min_reduced_inertia_eigenvalue=min((r.get('min_reduced_inertia_eigenvalue', 0) for r in rr),
                                                          default=0)))
    save_csv(out / 'support_dynamics_summary.csv', ds)
    cur = []
    for speed in (.5, 1., 2., 4.):
        rr = [r for r in ablation if r['speed_scale'] == speed]
        cur.append(dict(speed_scale=speed, configurations=len(rr),
                        full_max=max((r['full_residual'] for r in rr), default=0),
                        omitted_max=max((r['omitted_residual'] for r in rr), default=0),
                        omitted_min=min((r['omitted_residual'] for r in rr), default=0),
                        passed=sum(bool(r['success']) for r in rr)))
    save_csv(out / 'curvature_summary.csv', cur)
    status = dict(status='completed' if native else 'not_run',
                  version=states[0]['native_version'] if native and states else None,
                  reason=None if native else 'Native Pinocchio not enabled or not importable; no native values.')
    save_json(out / 'native_status.json', status)
    return dict(attempts=len(raw), passed=sum(bool(r['success']) for r in raw), maxima=maxima, support_modes=ds,
                curvature_attempts=len(ablation), curvature_passed=sum(bool(r['success']) for r in ablation),
                native=status,
                note='Rigid-body plus armature, ideal bilateral sole welds placed at the current sole frames; no '
                     'contact transitions, friction, impacts or rollouts.')


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=list(PROFILES), default='full')
    p.add_argument('--out', default='results_local')
    p.add_argument('--stages', nargs='+', default=['all'],
                   choices=['all', 'pipeline', 'checks', 'evaluator', 'warm', 'derivatives', 'incremental',
                            'acquisition', 'dynamics', 'native', 'rollout'])
    p.add_argument('--native', choices=['off', 'auto', 'required'], default='required')
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Output directory is not empty. Use a fresh --out name; previous results are never overwritten.')
    native, import_error = False, None
    if a.native != 'off':
        try:
            import pinocchio as pin
            if not hasattr(pin, 'constraintDynamics'):
                raise ImportError('Imported module lacks constraintDynamics (install the robotics package "pin")')
            native = True
        except ImportError as e:
            import_error = str(e)
            if a.native == 'required':
                p.error('Native Pinocchio required but unavailable: ' + str(e))
    stages = ['pipeline', 'checks', 'evaluator', 'warm', 'derivatives', 'incremental', 'acquisition', 'dynamics',
              'rollout'] if 'all' in a.stages else a.stages
    if 'rollout' in stages and not native:
        p.error('--stages rollout runs the delivered Pinocchio simulation and requires Pinocchio')
    if 'native' in stages and not native:
        p.error('--stages native requires Pinocchio (--native required or auto)')
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        p.error('threadpoolctl is required to fix the BLAS thread count (pip install threadpoolctl)')
    out.mkdir(parents=True, exist_ok=True)
    cfg = PROFILES[a.profile].copy()
    source = json.loads((ROOT / 'inputs/physical_graph.json').read_text())
    comp = compile_graph(source)
    acc = accepted()
    protocol = dict(profile=a.profile, configuration=cfg, stages=stages, native_requested=a.native,
                    native_import_error=import_error, methods=METHODS, derivative_methods=DERIVATIVE_METHODS,
                    incremental_methods=INCREMENTAL_METHODS, acquisition_methods=ACQUISITION_METHODS, seeds=SEEDS,
                    physical_gap_gate_m=1e-8, rank_rcond_gate=1e-10, independent_tangent_gate=TANGENT_GATE,
                    motor_input_gate=1e-12, known_branch_gate=2e-6, pacdm_tolerances='unchanged core: selected rows '
                    '1e-9, all rows 1e-8, corrector maxiter 8 in continuation', trf_tolerances=1e-11,
                    trf_max_nfev=1500, derivative_fresh_evaluation_budget=DERIVATIVE_BUDGET,
                    continuation_fresh_evaluation_budget=1500, blas_threads=1, derivative_lookahead_s=.05,
                    route='supplied v22 contact_reference motor inputs u(t), PCHIP at the route samples',
                    curvature_fd_step='1e-5 / max(1, max(abs(v)))',
                    timing_scope='warm/incremental: correction + output map; derivative: corrector only; '
                                 'acquisition: homotopy + mapping; setup and independent validation excluded equally',
                    note='Predefined finite suite; all attempts and unfavourable baselines retained. New compiler/'
                         'evaluator/scheduler extension; PACDM original bytes unchanged.')
    save_json(out / 'protocol.json', protocol)
    save_json(out / 'compiled_cmg.json', comp.cmg)
    save_json(out / 'compiled_plan.json', comp.plan)
    save_json(out / 'source_hashes.json', {
        str(f.relative_to(ROOT)).replace('\\', '/'): sha(f) for f in sorted(ROOT.rglob('*'))
        if f.is_file() and (f.suffix in ('.py', '.json', '.npz') and ('original' in f.parts or 'src' in f.parts
                                                                          or f.parent == ROOT / 'inputs'))
        and '__pycache__' not in f.parts and out not in f.parents})
    summary = dict(status='running', profile=a.profile, stages={}, protocol_sha256=sha(out / 'protocol.json'))
    with threadpool_limits(limits=1):
        save_json(out / 'environment.json', environment())
        start_all = perf_counter()
        for stage in stages:
            print('STAGE', stage, flush=True)
            start = perf_counter()
            try:
                if stage == 'pipeline':
                    r = pipeline_stage(source, cfg, out, acc)
                elif stage == 'checks':
                    r = fault_stage(source, out, acc)
                elif stage == 'warm':
                    r = warm_stage(comp, cfg, out, acc)
                elif stage == 'derivatives':
                    r = derivative_stage(comp, cfg, out, acc)
                elif stage == 'incremental':
                    r = incremental_stage(comp, cfg, out, acc)
                elif stage == 'acquisition':
                    r = acquisition_stage(comp, cfg, out, acc)
                elif stage == 'evaluator':
                    r = evaluator_stage(comp, cfg, out, acc)
                elif stage == 'rollout':
                    r = rollout_stage(comp, cfg, out, acc)
                else:
                    r = dynamics_stage(comp, cfg, out, native, acc)
                r['stage_wall_s'] = perf_counter() - start
                summary['stages'][stage] = r
                save_json(out / 'summary.json', summary)
                print('DONE', stage, 'seconds', round(r['stage_wall_s'], 2), flush=True)
            except Exception as e:
                summary.update(status='error', error=type(e).__name__ + ': ' + str(e),
                               traceback=traceback.format_exc())
                save_json(out / 'summary.json', summary)
                raise
        summary['total_wall_s'] = perf_counter() - start_all
    failures = []
    for name, r in summary['stages'].items():
        denom = r.get('attempts', r.get('checks', r.get('tests', 0)))
        if r.get('passed', 0) != denom:
            failures.append(name)
        if r.get('curvature_passed', 0) != r.get('curvature_attempts', 0):
            failures.append(name + '_curvature')
    summary.update(status='completed' if not failures else 'completed_with_failures', failed_stages=failures)
    save_json(out / 'summary.json', summary)
    save_json(out / 'RUN_MANIFEST.json', {str(f.relative_to(out)).replace('\\', '/'): sha(f)
                                          for f in sorted(out.rglob('*')) if f.is_file()
                                          and f.name != 'RUN_MANIFEST.json'})
    print('RESULT:', summary['status'], 'OUTPUT:', out, flush=True)
    return 0 if not failures else 2
