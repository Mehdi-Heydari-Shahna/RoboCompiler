"""Reproducible excavator CMG/PACDM benefit experiments. All attempts are kept."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import math
import traceback

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from . import bootstrap  # noqa: F401
from .bootstrap import ROOT
from .compiler import compile_graph, ModelError, GROUND
from .context import Context, PARTITIONS
from .io_utils import save_json, save_csv, sha, environment, timing_summary
from .loop_graph import GlobalLoopGraph, ModuleLoopGraph
from .solvers import (create_solver, trf_solve, polish, CompiledModular, command_drivers, loop_input_positions,
                      METHOD_TABLE)
from .physics import Dynamics, mapping_curvature, normmax, relative
from .variants import variant_inputs, opaque
from pacdm import PACDM
from plant import ReducedPlant, ExcavatorPinModel
from excavator_pacdm import compile_pacdm

# ------------------------------------------------------------------ protocol
REFERENCE_METHODS = ['original_global', 'compiled_global', 'compiled_global_reuse', 'compiled_global_loopfree',
                     'compiled_modular', 'modular_no_reuse', 'modular_no_predictor', 'trf_global', 'trf_global_loopfree',
                     'trf_modular', 'native_newton', 'native_newton_loopfree', 'native_modular']
NOHOLD_METHODS = ['compiled_global_reuse', 'compiled_global_loopfree', 'compiled_modular', 'trf_global',
                  'trf_global_loopfree', 'trf_modular', 'native_newton', 'native_newton_loopfree', 'native_modular']
MEASURED_METHODS = ['original_global', 'compiled_global', 'compiled_modular', 'trf_global', 'trf_modular',
                    'native_newton', 'native_modular']
# signal: 'desired' = original mission reference (planned commands); 'measured' = the simulated
# independent coordinates of the validated nominal run. commands: 'joint' = the signal itself;
# 'hold' = cylinder strokes held bit-exactly while their derived joint-space drivers are unchanged;
# 'nohold' = strokes read from the closed state at every sample.
ROUTES = {
    'joint_reference': dict(partition='joint_space', signal='desired', commands='joint', methods=REFERENCE_METHODS),
    'cylinder_reference': dict(partition='cylinder_space', signal='desired', commands='hold', methods=REFERENCE_METHODS),
    'cylinder_reference_nohold': dict(partition='cylinder_space', signal='desired', commands='nohold',
                                      methods=NOHOLD_METHODS),
    'joint_measured': dict(partition='joint_space', signal='measured', commands='joint', methods=MEASURED_METHODS),
    'cylinder_measured': dict(partition='cylinder_space', signal='measured', commands='nohold', methods=MEASURED_METHODS),
}
LOCAL_METHODS = ['original_global', 'compiled_global', 'compiled_global_loopfree', 'compiled_modular', 'modular_no_reuse',
                 'trf_global', 'trf_global_loopfree', 'trf_modular', 'native_newton', 'native_newton_loopfree',
                 'native_modular']
DERIVATIVE_METHODS = ['analytic_dense', 'fd2_dense', 'fd3_dense', 'analytic_sparse', 'fd2_colored', 'fd3_colored']
# Local edits are defined on the joint-space coordinates (always feasible); the
# cylinder-space targets are read from the resulting closed configuration, so
# both partitions receive the same physical edit sequence. The amplitude keeps
# every edit below the original controller's 0.02 rad subdivision step, which
# the benchmark does not reproduce.
LOCAL_AMPLITUDE = 0.018
LOCAL_CONDITIONS = [('slew', ['q23']), ('boom', ['q7']), ('stick', ['q4']), ('bucket', ['q0']), ('tilt', ['q1']),
                    ('rotator', ['q21']), ('pin', ['q22']), ('dig', ['q7', 'q4', 'q0']),
                    ('all', ['q23', 'q7', 'q4', 'q0', 'q1', 'q21', 'q22'])]
PHASE_GROUPS = {'hold': (0, 5, 8), 'dig': (1, 2, 3, 4, 7, 9), 'slew': (6, 10)}
PROFILES = {
    'smoke': dict(route_stride=40, repeats=1, variants=3, witnesses=2, derivative_cases=4, local_starts=1,
                  local_edits=3, dynamics_states=4, curvature_states=2, evaluator_calls=100, evaluator_repeats=2,
                  state_stride=10),
    'full': dict(route_stride=1, repeats=3, variants=12, witnesses=5, derivative_cases=48, local_starts=4,
                 local_edits=12, dynamics_states=48, curvature_states=12, evaluator_calls=2000,
                 evaluator_repeats=5, state_stride=10)}
# Effort amplitudes (actuator order p0 p1 p2 p3 p4 p5 q21 q23, N or N m) of the order of the
# nominal cycle's peak port efforts (2 N, 1.3 kN, 14 kN, 144 kN, 144 kN, 496 kN, 72 N m, 20.7 kN m).
EFFORT_RANGE = np.array([5e3, 5e3, 3e4, 1.5e5, 1.5e5, 5e5, 1e3, 2.5e4])
SEEDS = dict(pipeline=2026092601, warm=2026092602, derivatives=2026092603, local=2026092604,
             dynamics=2026092605, evaluators=2026092606)
# Error metrics are recorded rounded UP to four significant digits (a recorded value never
# understates the computed one); acceptance itself uses the full-precision values.
ERROR_FIELDS = ('max_gap_m', 'target_error', 'tangent_residual_ref', 'map_error', 'known_branch_error')


def digest(*arrays):
    """Short state identifier (SHA-1 over the float64 bytes, 48 bits)."""
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()[:12]


def up4(x):
    """Round a nonnegative error metric up to four significant digits (never below x)."""
    x = float(x)
    if x == 0. or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x))) - 3
    m = x / 10. ** e
    r = round(m) if abs(m - round(m)) <= 1e-9 * abs(m) else math.ceil(m)
    v = float(f'{r}e{e}')
    while v < x:
        v = float(np.nextafter(v, math.inf))
    return v


def sig4(x):
    return None if x is None else float(f'{float(x):.4g}')


def record_check(row, check):
    for k, v in check.items():
        row[k] = up4(v) if k in ERROR_FIELDS else v


def record_time(t0):
    return round((perf_counter() - t0) * 1e3, 4)


# ---------------------------------------------------------------- route truth and commands
def route_truth(ctx, route, partition='joint_space'):
    """Closed reference states along a route: validated native projection, 1e-12 m."""
    plant = ReducedPlant(ctx.cmg, ctx.mapping['cut_joint_ids'], ctx.independent_ids(partition),
                         projection_tolerance_m=1e-12)
    plant.initialize(np.asarray(ctx.route['tree'][0], float))
    states = np.zeros((len(route), len(ctx.ref_ids)))
    for i, u in enumerate(route):
        states[i] = plant.project(u)[0]
    return states


def cylinder_commands(ctx, joint_targets, truth_states, previous=None, previous_joint=None, hold=True):
    """Cylinder-space targets for a sequence of joint-space targets.

    A stroke's exact value is read from the closed truth state. With ``hold``, the stroke
    command is instead held bit-exactly while every joint-space input of the joint-space
    module that contains the stroke (``command_drivers``, derived from the compiled
    modules) is unchanged: the stroke is then physically unchanged and only the global
    truth projection's round-off would move it. Returns the targets and the largest
    deviation of a held value from the truth.
    """
    drivers = command_drivers(ctx)
    jids, cids = ctx.independent_ids('joint_space'), ctx.independent_ids('cylinder_space')
    out = np.zeros((len(joint_targets), len(cids)))
    deviation = 0.
    for i, (u, state) in enumerate(zip(joint_targets, truth_states)):
        prev_u = joint_targets[i - 1] if i else previous_joint
        prev_c = out[i - 1] if i else previous
        for c, name in enumerate(cids):
            if name in jids:
                out[i, c] = u[jids.index(name)]
                continue
            exact = state[ctx.ref_ids.index(name)]
            held = hold and prev_c is not None and all(u[jids.index(d)] == prev_u[jids.index(d)] for d in drivers[name])
            out[i, c] = prev_c[c] if held else exact
            deviation = max(deviation, abs(out[i, c] - exact))
    return out, deviation


def route_inputs(ctx):
    """Targets and truth of every benchmark route (built once, saved with the results)."""
    signals = dict(desired=ctx.joint_route(), measured=np.asarray(ctx.route['measured'], float))
    truth = dict(desired=ctx._truth, measured=ctx._truth_measured)
    targets, deviations = {}, {}
    for name, spec in ROUTES.items():
        joint = signals[spec['signal']]
        if spec['commands'] == 'joint':
            targets[name], deviations[name] = joint, 0.
        else:
            targets[name], deviations[name] = cylinder_commands(ctx, joint, truth[spec['signal']],
                                                                hold=spec['commands'] == 'hold')
    return targets, truth, deviations


# ---------------------------------------------------------------- pipeline
def native_cmg_from_physical(physical):
    """Native-model records written directly from the physical input (field renaming only;
    the compiler's tree, cuts and coordinate order are not used)."""
    return dict(root_body=physical.get('ground', GROUND),
                bodies=[dict(id=physical.get('ground', GROUND), kind='reference_frame', mass_kg=0.,
                             com_m=[0., 0., 0.], inertia_com_kg_m2=np.zeros((3, 3)).tolist())]
                + [dict(deepcopy(b), kind='rigid_body') for b in physical['bodies']],
                joints=[dict(id=j['id'], type=j['type'], base_body=j['body_a'], follower_body=j['body_b'],
                             T_BJ=deepcopy(j['T_AJ']), T_FJ=deepcopy(j['T_BJ']), axis=deepcopy(j['axis']))
                        for j in physical['joints']])


def _original_names(physical, maps):
    """Undo the opaque renaming of a physical input (bodies and joints)."""
    s = deepcopy(physical)
    bmap = {v: k for k, v in maps['bodies'].items()}
    jmap = {v: k for k, v in maps['joints'].items()}
    for b in s['bodies']:
        b['id'] = bmap.get(b['id'], b['id'])
    for j in s['joints']:
        j['id'] = jmap.get(j['id'], j['id'])
        j['body_a'] = bmap.get(j['body_a'], j['body_a'])
        j['body_b'] = bmap.get(j['body_b'], j['body_b'])
    return s


def pipeline_stage(ctx, cfg, out):
    rows, compiles = [], []
    rng = np.random.default_rng(SEEDS['pipeline'])
    original_cuts = list(ctx.mapping['cut_joint_ids'])
    for variant, physical in variant_inputs(ctx.physical)[:cfg['variants']]:
        rename, maps = None, None
        if variant == 'opaque_body_joint_names':
            _, maps = opaque(ctx.physical)
            rename = {v: k for k, v in maps['joints'].items()}      # opaque -> original joint id
        # Independent reference: native Pinocchio model built DIRECTLY from the variant's
        # physical records (not from the compiler output) with the ORIGINAL cut set.
        ref = ExcavatorPinModel(native_cmg_from_physical(_original_names(physical, maps) if maps else physical),
                                original_cuts)
        ref_ids = list(ref.tree_ids)
        for partition in PARTITIONS:
            start = perf_counter()
            comp = compile_graph(physical, partition)
            ms = (perf_counter() - start) * 1e3
            directory = out / 'generated' / variant / partition
            save_json(directory / 'compiled_cmg.json', comp.cmg)
            save_json(directory / 'plan.json', comp.plan)
            if partition == 'joint_space':
                save_json(out / 'generated' / variant / 'physical_graph.json', physical)
            compiles.append(dict(variant=variant, partition=partition, compilation_ms=ms,
                                 modules=comp.plan['modules_count'], largest_block=comp.plan['largest_dependent_block'],
                                 mobility=comp.plan['mobility'], seed_acquired=comp.plan['seed_acquisition'] is not None,
                                 cut_joint_ids=' '.join(comp.plan['cut_joint_ids'])))
            s = comp.structure
            ids = list(s['coordinate_ids'])
            names = [rename.get(k, k) for k in ids] if rename else ids
            take = np.array([names.index(k) for k in ref_ids])
            graph = GlobalLoopGraph(comp)
            pac = PACDM(graph)
            seed = np.asarray(s['seed_closed'])
            modules = [ModuleLoopGraph(comp, m) for m in s['modules']]
            sparsity = np.asarray(s['sparsity'])
            dependent = np.array(s['dependent'], int)
            active = np.array(s['independent'], int)
            kind = {j['id']: j['type'] for j in physical['joints']}
            spans = np.array([.02 if kind[c] == 'prismatic' else .06 for c in ids])
            for k in range(cfg['witnesses']):
                target = seed[active] + rng.uniform(-1, 1, len(active)) * spans[active]
                q, info = pac.acquire(target, seed)
                row = dict(variant=variant, partition=partition, witness=k, success=False, error='')
                try:
                    if not info['success']:
                        raise RuntimeError(f'acquisition failed: {info.get("message")}')
                    N, mi = pac.mapping(q)
                    if N is None:
                        raise RuntimeError(f'mapping failed: {mi.get("message")}')
                    r, J, _ = graph.residual(q)
                    # finite-difference Jacobian of the compiled residual
                    h = 1e-6
                    Jfd = np.zeros_like(J)
                    for c in range(len(q)):
                        d = np.zeros(len(q))
                        d[c] = h
                        Jfd[:, c] = (graph.residual(q + d)[0] - graph.residual(q - d)[0]) / (2 * h)
                    fd_error = normmax(J - Jfd) / max(1., normmax(J))
                    zero_error = normmax(J[:, dependent][sparsity == 0])
                    # modular map vs global map
                    Nmod = np.zeros_like(N)
                    Nmod[active, np.arange(len(active))] = 1.
                    col = {c: i for i, c in enumerate(active)}
                    mod_ok = True
                    for mg in modules:
                        Nm, im = PACDM(mg).mapping(q[mg.global_index])
                        mod_ok = mod_ok and Nm is not None
                        if Nm is not None:
                            Nmod[np.ix_(mg.dependents, [col[c] for c in mg.inputs])] = Nm[mg.passive]
                    modular_error = normmax(Nmod - N)
                    # native reference at the compiled configuration
                    qref = q[take]
                    gap = normmax(ref.closure_residual(qref))
                    Jn = ref.closure_jacobian(qref)
                    a_ref = np.array([ref_ids.index(names[c]) for c in active])
                    p_ref = np.setdiff1d(np.arange(len(ref_ids)), a_ref)
                    Nn = np.zeros((len(ref_ids), len(active)))
                    Nn[a_ref, np.arange(len(active))] = 1.
                    Nn[p_ref] = np.linalg.lstsq(Jn[:, p_ref], -Jn[:, a_ref], rcond=None)[0]
                    map_error = normmax(N[take] - Nn)
                    tangent = normmax(Jn @ N[take])
                    M, _ = ref.mass_bias(qref, np.zeros(len(ref_ids)))
                    mr_min = float(np.linalg.eigvalsh(N[take].T @ M @ N[take]).min())
                    row.update(native_closure_gap_m=gap, compiled_residual=normmax(r), fd_jacobian_error=fd_error,
                               sparsity_zero_error=zero_error, modular_vs_global_map=modular_error,
                               map_vs_native=map_error, tangent_residual_native=tangent,
                               min_reduced_inertia_eigenvalue=mr_min, rank_passive=mi['rank_passive'],
                               expected_rank=len(dependent), rcond=mi['rcond'])
                    row['success'] = bool(gap < 1e-9 and fd_error < 1e-6 and zero_error == 0. and modular_error < 1e-9
                                          and map_error < 1e-9 and tangent < 1e-9 and mr_min > 0 and mod_ok
                                          and mi['rank_passive'] == len(dependent))
                except Exception as e:  # recorded, never hidden
                    row['error'] = type(e).__name__ + ': ' + str(e)
                rows.append(row)
    save_csv(out / 'pipeline_raw.csv', rows)
    save_csv(out / 'compilation_times.csv', compiles)
    return dict(variants=len({c['variant'] for c in compiles}), partitions=len(PARTITIONS),
                compilations=len(compiles), checks=len(rows), passed=sum(r['success'] for r in rows),
                median_compilation_ms=float(np.median([c['compilation_ms'] for c in compiles])),
                note='Closed witness configurations per variant and partition, assembled by the compiled PACDM and '
                     'checked against a native Pinocchio model written directly from the same physical records with '
                     'the original cut set. Not cold-start trials.')


# ---------------------------------------------------------------- checks
def checks_stage(ctx, out):
    raw = []
    source = ctx.physical

    def reject(name, mutate, expected, partition='joint_space', **kw):
        s = deepcopy(source)
        mutate(s)
        message, detected = '', False
        try:
            compile_graph(s, partition, **kw)
        except ModelError as e:
            detected, message = True, str(e)
        raw.append(dict(test=name, expected=f'reject: {expected}', success=detected and expected in message,
                        message=message))

    joint = lambda s, jid: next(j for j in s['joints'] if j['id'] == jid)  # noqa: E731

    def drop_joint_everywhere(s, jid):
        s['joints'].remove(joint(s, jid))
        s['seed']['joints'].pop(jid)
        s['actuators'] = [a for a in s['actuators'] if a['joint_id'] != jid]
        s['partition_requests'] = {k: [x for x in v if x != jid] for k, v in s['partition_requests'].items()}

    def make_fixed(s, jids):
        for jid in jids:
            joint(s, jid).update(type='fixed')
            s['seed']['joints'].pop(jid)

    reject('duplicate_body', lambda s: s['bodies'].append(deepcopy(s['bodies'][0])), 'Unique physical bodies required')
    reject('duplicate_joint', lambda s: s['joints'].append(deepcopy(s['joints'][0])), 'Unique physical joint IDs required')
    reject('negative_mass', lambda s: s['bodies'][0].update(mass_kg=-1.), 'finite positive mass')
    reject('invalid_inertia', lambda s: s['bodies'][0].update(inertia_com_kg_m2=np.diag([1., 1., -1.]).tolist()),
           'inertia must be symmetric positive definite')
    reject('missing_joint_endpoint', lambda s: s['joints'][2].update(body_a='missing'), 'invalid joint endpoints')
    reject('disconnected_graph', lambda s: drop_joint_everywhere(s, 'q21'), 'Disconnected physical graph')
    reject('nonunit_axis', lambda s: joint(s, 'q7').update(axis=[0., 0., 2.]), 'axis must be a finite unit vector')
    reject('improper_transform', lambda s: joint(s, 'q7').update(T_AJ=np.zeros((4, 4)).tolist()),
           'expected a finite homogeneous transform')
    reject('unsupported_joint_type', lambda s: joint(s, 'q7').update(type='spherical'), 'unsupported joint type')
    reject('preauthored_cut_list', lambda s: s.update(cut_joint_ids=['q5']), 'must not pre-author computational structure')
    reject('preauthored_modules', lambda s: s.update(modules=[['q5']]), 'must not pre-author computational structure')
    reject('seed_missing_joint', lambda s: s['seed']['joints'].pop('q9'), 'Named seed must cover exactly')
    reject('unknown_requested_coordinate', lambda s: None, 'must be distinct moving physical joints',
           requested=['q23', 'q7', 'q4', 'q0', 'q1', 'q21', 'qX'])
    reject('redundant_parallel_cylinders_requested', lambda s: None, 'kinematically redundant',
           requested=['q23', 'p3', 'p4', 'p5', 'p2', 'q21', 'q22'])
    reject('more_requests_than_mobility', lambda s: None, 'exceed the mobility',
           requested=['q23', 'q7', 'q4', 'q0', 'q1', 'q21', 'q22', 'q24'])
    reject('fixed_joint_requested', lambda s: None, 'must be distinct moving physical joints',
           requested=['fixed_world_base', 'q7', 'q4', 'q0', 'q1', 'q21', 'q22'])
    reject('incomplete_partition_without_completion', lambda s: None, 'completion disabled',
           requested=['q23', 'q7', 'q4', 'q0', 'q1', 'q21'], complete=False)
    reject('loop_only_cuttable_at_prismatic', lambda s: make_fixed(s, ('q24', 'q5')),
           'loop would have to be cut at a prismatic')

    # component checks
    comp = ctx.compiled['joint_space']
    auto = compile_graph(source, 'joint_space', requested=['q23', 'q7', 'q4', 'q0', 'q1', 'q21'])
    completion = auto.plan['completion_ids']
    raw.append(dict(test='completion_finds_internal_pivot_mode', expected='one of q15, q18, q22',
                    success=len(completion) == 1 and completion[0] in ('q15', 'q18', 'q22'), message=str(completion)))
    drivers = command_drivers(ctx)
    raw.append(dict(test='command_drivers_derived_from_modules',
                    expected="p3<-q7, p5<-q4, p2<-(q0,q22), p1<-q1 (the machine's cylinder functions)",
                    success=drivers == {'p3': ('q7',), 'p5': ('q4',), 'p2': ('q0', 'q22'), 'p1': ('q1',)},
                    message=str(drivers)))
    loopfree = {p: [ctx.independent_ids(p)[i] for i in range(7) if i not in loop_input_positions(ctx.compiled[p])]
                for p in PARTITIONS}
    raw.append(dict(test='loop_free_coordinates', expected='slew q23 and rotator q21 in both partitions',
                    success=all(v == ['q23', 'q21'] for v in loopfree.values()), message=str(loopfree)))
    q32 = ctx.full_state(ctx.route['tree'][0])
    solver = CompiledModular(ctx, q32, 'joint_space')
    u0 = np.array([q32[k] for k in ctx.independent_ids('joint_space')])
    _, _, i1 = solver.step(u0)
    raw.append(dict(test='unchanged_inputs_reuse_every_module', expected='0 solved, 6 skipped',
                    success=i1['solved_modules'] == 0 and i1['skipped_modules'] == 6, message=str(i1)))
    u = u0.copy()
    for label, coord, expected in (('slew_change_solves_no_module', 'q23', 0), ('rotator_change_solves_no_module', 'q21', 0),
                                   ('boom_change_solves_two_modules', 'q7', 2), ('tilt_change_solves_two_modules', 'q1', 2),
                                   ('stick_change_solves_one_module', 'q4', 1), ('bucket_change_solves_linkage_module', 'q0', 1),
                                   ('pin_change_solves_linkage_module', 'q22', 1)):
        u = u.copy()
        u[ctx.independent_ids('joint_space').index(coord)] += .01
        q, N, info = solver.step(u)
        chk = ctx.reference.check(q, N, u, 'joint_space')
        raw.append(dict(test=label, expected=f'{expected} solved and accepted',
                        success=info['solved_modules'] == expected and chk['success'], message=str(info)))
    cyl = CompiledModular(ctx, q32, 'cylinder_space')
    uc = np.array([q32[k] for k in ctx.independent_ids('cylinder_space')])
    uc[1] += .005
    q, N, info = cyl.step(uc)
    chk = ctx.reference.check(q, N, uc, 'cylinder_space')
    raw.append(dict(test='boom_cylinder_change_solves_one_merged_module', expected='1 solved (boom pair) and accepted',
                    success=info['solved_modules'] == 1 and chk['success'], message=str(info)))
    # loop-free exclusion is exact: the compiled loop residual does not depend on slew/rotator
    lf = create_solver('compiled_global_loopfree', ctx, q32, 'joint_space')
    x_before = lf.x.copy()
    u = u0.copy()
    u[0] += .3
    u[5] -= .2
    q, N, info = lf.step(u)
    N_fresh, _ = lf.solver.mapping(lf.x)
    exact = (info['solved_modules'] == 0 and np.array_equal(lf.g.residual(x_before)[0], lf.g.residual(lf.x)[0])
             and np.array_equal(N_fresh[lf.out.take], N))
    raw.append(dict(test='loop_free_reuse_bit_exact', expected='no solve; residual and fresh PACDM map bit-identical',
                    success=bool(exact and ctx.reference.check(q, N, u, 'joint_space')['success']), message=str(info)))
    # commit after success: a failing step leaves the modular solvers unchanged
    for name in ('compiled_modular', 'native_modular'):
        s = create_solver(name, ctx, q32, 'cylinder_space')
        before = (s.q.copy(), s.N.copy())
        u = np.array([q32[k] for k in ctx.independent_ids('cylinder_space')])
        u[4] += .005          # feasible tilt-cylinder change (first module)
        u[1] += 1.0           # infeasible boom-cylinder stroke
        failed = False
        try:
            s.step(u)
        except RuntimeError:
            failed = True
        unchanged = np.array_equal(before[0], s.q) and np.array_equal(before[1], s.N)
        raw.append(dict(test=f'{name}_failed_step_commits_nothing', expected='RuntimeError and unchanged state',
                        success=failed and unchanged, message=f'failed={failed} unchanged={unchanged}'))
    # compiled cut choice vs original cut set: identical physical solutions
    alt = compile_graph(source, 'joint_space', cut_override=ctx.mapping['cut_joint_ids'])
    ga, gb = GlobalLoopGraph(comp), GlobalLoopGraph(alt)
    route = ctx.joint_route()
    worst, all_ok, samples = 0., True, 0
    xa = np.array([q32[k] for k in ga.ids])
    xb = np.array([q32[k] for k in gb.ids])
    pa, pb = PACDM(ga), PACDM(gb)
    Na, ia = pa.mapping(xa)
    Nb, ib = pb.mapping(xb)
    for i in range(0, len(route), 84):
        samples += 1
        for g, p, holder in ((ga, pa, 'a'), (gb, pb, 'b')):
            x = xa if holder == 'a' else xb
            N_ = Na if holder == 'a' else Nb
            info_ = ia if holder == 'a' else ib
            pred = x[g.passive] + N_[g.passive] @ (route[i] - x[g.active])
            y, ci = p.correct(route[i], pred, None, np.array(info_['rows']), maxiter=12)
            all_ok = all_ok and bool(ci['success'])
            y = polish(g, y, np.array(info_['rows']))
            Nn, inn = p.mapping(y)
            all_ok = all_ok and bool(inn['success'])
            if holder == 'a':
                xa, Na, ia = y, Nn, inn
            else:
                xb, Nb, ib = y, Nn, inn
        va = dict(zip(ga.ids, xa))
        vb = dict(zip(gb.ids, xb))
        worst = max(worst, max(abs(va[k] - vb[k]) for k in va))
    raw.append(dict(test='generated_cuts_equal_original_cuts_physically',
                    expected=f'every correction succeeds; max joint difference < 1e-10 over {samples} route samples',
                    success=all_ok and worst < 1e-10, message=f'{worst:.3e} (all corrections successful: {all_ok})'))
    # reversed joint storage preserves kinematics
    s = deepcopy(source)
    for j in s['joints']:
        j['body_a'], j['body_b'] = j['body_b'], j['body_a']
        j['T_AJ'], j['T_BJ'] = j['T_BJ'], j['T_AJ']
        if j['type'] != 'fixed':
            j['axis'] = (-np.array(j['axis'])).tolist()
    rev = compile_graph(s, 'joint_space')
    gr = GlobalLoopGraph(rev)
    xr = np.array([q32[k] for k in gr.ids])
    raw.append(dict(test='reversed_joint_storage_same_closed_seed', expected='residual < 1e-12 at the same joint values',
                    success=normmax(gr.residual(xr)[0]) < 1e-12, message=f'{normmax(gr.residual(xr)[0]):.3e}'))
    save_csv(out / 'verification_tests.csv', raw)
    return dict(tests=len(raw), passed=sum(r['success'] for r in raw),
                note='Designed compiler/precondition/component checks with asserted messages; not a real-world '
                     'failure-detection probability.')


# ---------------------------------------------------------------- evaluators
def evaluators_stage(ctx, cfg, out):
    rng = np.random.default_rng(SEEDS['evaluators'])
    comp = ctx.compiled['joint_space']
    alt = compile_graph(ctx.physical, 'joint_space', cut_override=ctx.mapping['cut_joint_ids'])
    go, _ = compile_pacdm(ctx.cmg, ctx.mapping, ctx.mapping['tree_joint_ids'])
    gg, ga = GlobalLoopGraph(comp), GlobalLoopGraph(alt)
    native = ExcavatorPinModel(ctx.cmg, ctx.mapping['cut_joint_ids'])
    truth = ctx._truth
    samples = truth[rng.integers(0, len(truth), cfg['evaluator_calls'])]
    states = [ctx.full_state(t) for t in samples]
    X = {'original': [np.array([s[k] for k in go.ids]) for s in states],
         'compiled': [np.array([s[k] for k in gg.ids]) for s in states],
         'compiled_original_cuts': [np.array([s[k] for k in ga.ids]) for s in states],
         'native': [np.array([s[k] for k in ctx.ref_ids]) for s in states]}
    # agreement before timing
    perm = [go.ids.index(k) for k in ga.ids]
    ocut, acut = ctx.mapping['cut_joint_ids'], alt.plan['cut_joint_ids']
    agree = 0.
    for xo, xa in zip(X['original'][:50], X['compiled_original_cuts'][:50]):
        ro, Jo, _ = go.residual(xo)
        ra, Ja, _ = ga.residual(xa)
        for i, cid in enumerate(acut):
            j = ocut.index(cid)
            agree = max(agree, normmax(ra[6 * i:6 * i + 6] - ro[6 * j:6 * j + 6]),
                        normmax(Ja[6 * i:6 * i + 6] - Jo[6 * j:6 * j + 6][:, perm]))
    modules = [(m, ModuleLoopGraph(comp, m)) for m in comp.structure['modules']]
    tasks = [('original_residual_jacobian', go.residual, X['original']),
             ('compiled_residual_jacobian', gg.residual, X['compiled']),
             ('compiled_original_cuts_residual_jacobian', ga.residual, X['compiled_original_cuts']),
             ('compiled_residual_only', gg.residual_only, X['compiled']),
             ('native_residual_jacobian', lambda x: (native.closure_residual(x), native.closure_jacobian(x)), X['native']),
             ('native_residual_only', native.closure_residual, X['native'])]
    for m, mg in modules:
        label = 'module_' + '_'.join(m['cut_ids'])
        tasks.append((label + '_residual_jacobian', mg.residual, [x[mg.global_index] for x in X['compiled']]))
    rows = []
    for rep in range(cfg['evaluator_repeats']):
        for name, fn, xs in [tasks[i] for i in rng.permutation(len(tasks))]:
            start = perf_counter()
            for x in xs:
                fn(x)
            rows.append(dict(evaluator=name, repeat=rep, calls=len(xs), us_per_call=(perf_counter() - start) / len(xs) * 1e6))
    summary = []
    for name in dict.fromkeys(t[0] for t in tasks):
        v = [r['us_per_call'] for r in rows if r['evaluator'] == name]
        summary.append(dict(evaluator=name, repeats=len(v), median_us=float(np.median(v)), min_us=float(min(v)),
                            max_us=float(max(v))))
    mods = [s for s in summary if s['evaluator'].startswith('module_')]
    summary.append(dict(evaluator='sum_of_all_modules_residual_jacobian', repeats=cfg['evaluator_repeats'],
                        median_us=float(sum(s['median_us'] for s in mods)), min_us=None, max_us=None))
    save_csv(out / 'evaluator_raw.csv', rows)
    save_csv(out / 'evaluator_summary.csv', summary)
    return dict(evaluators=len(tasks), repeats=cfg['evaluator_repeats'], calls=cfg['evaluator_calls'],
                original_vs_compiled_same_cuts_max_difference=agree, passed=int(agree < 1e-12), attempts=1,
                summary=summary,
                note='Per-call cost on closed states sampled from the route (an implementation comparison of two '
                     'Python evaluators of the same residual). Native = the validated Pinocchio backend (C++ kinematics '
                     'through Python bindings, 18 point pairs).')


# ---------------------------------------------------------------- warm routes
def _phase_group(phase):
    return next(g for g, ph in PHASE_GROUPS.items() if phase in ph)


def warm_stage(ctx, cfg, out):
    rng = np.random.default_rng(SEEDS['warm'])
    targets, truth, deviations = route_inputs(ctx)
    idx = np.arange(0, len(ctx._truth), cfg['route_stride'])
    phase = np.asarray(ctx.route['phase'])
    q0 = ctx.full_state(ctx._truth[0])
    store, qs, Ns = {}, {}, {}
    summaries, phase_rows, repeat_rows, totals, attempts, passed = [], [], [], [], 0, 0
    loop_changes = {}
    for route_name, spec in ROUTES.items():
        partition = spec['partition']
        route = targets[route_name]
        known = truth[spec['signal']]
        methods = [m for m in spec['methods'] if not (m == 'original_global' and partition != 'joint_space')]
        loop_pos = loop_input_positions(ctx.compiled[partition])
        loop_changes[route_name] = int(sum(1 for n in range(1, len(idx))
                                           if not np.array_equal(route[idx[n]][loop_pos], route[idx[n - 1]][loop_pos])))
        start_state = q0 if spec['signal'] == 'desired' else ctx.full_state(known[0])
        raw = []
        for repeat in range(cfg['repeats']):
            solvers = {name: create_solver(name, ctx, start_state, partition) for name in methods}
            for n, i in enumerate(idx):
                moving = bool(n and not np.array_equal(route[i], route[idx[n - 1]]))
                for name in rng.permutation(methods):
                    row = dict(route=route_name, partition=partition, method=str(name), repeat=repeat, sample=int(i),
                               phase=int(phase[i]), moving=moving, success=False, error='')
                    t0 = perf_counter()
                    try:
                        q, N, info = solvers[name].step(route[i])
                        row['time_ms'] = record_time(t0)
                        check = ctx.reference.check(q, N, route[i], partition, known=known[i])
                        row.update(fallback=info['fallback'], solved_modules=info['solved_modules'],
                                   skipped_modules=info['skipped_modules'], rcond=sig4(info['rcond']))
                        record_check(row, check)
                        key = digest(q, N)
                        row['state_digest'] = key
                        if repeat == 0 and n % cfg['state_stride'] == 0:
                            qk, Nk = q.tobytes(), N.tobytes()
                            qs.setdefault(qk, (len(qs), q))
                            Ns.setdefault(Nk, (len(Ns), N))
                            store[key] = (qs[qk][0], Ns[Nk][0])
                    except Exception as e:
                        row['time_ms'] = record_time(t0)
                        row['error'] = type(e).__name__ + ': ' + str(e)
                    raw.append(row)
                if n % 1000 == 0:
                    print(f'warm {route_name} repeat {repeat + 1}/{cfg["repeats"]} sample {i}/{len(idx)}', flush=True)
        save_csv(out / 'warm_raw' / f'{route_name}.csv.gz', raw)
        for r in raw:
            r['phase_group'] = _phase_group(r['phase'])
        summaries += timing_summary(raw, ('route', 'partition', 'method'))
        phase_rows += timing_summary(raw, ('route', 'partition', 'phase_group', 'method'))
        repeat_rows += timing_summary(raw, ('route', 'partition', 'method', 'repeat'))
        sums, no_solve_count = {}, {}
        for r in raw:
            key = (r['method'], r['repeat'])
            sums[key] = sums.get(key, 0.) + r['time_ms']
            if r['repeat'] == 0 and r.get('solved_modules') == 0:
                no_solve_count[r['method']] = no_solve_count.get(r['method'], 0) + 1
        for name in methods:
            per = [sums[(name, rep)] / 1e3 for rep in range(cfg['repeats'])]
            no_solve = no_solve_count.get(name, 0)
            totals.append(dict(route=route_name, partition=partition, method=name, repeats=cfg['repeats'],
                               median_total_s=float(np.median(per)), min_total_s=min(per), max_total_s=max(per),
                               samples_per_repeat=len(idx), samples_without_solve=no_solve))
        attempts += len(raw)
        passed += sum(r['success'] for r in raw)
        del raw
    keys = sorted(store)
    np.savez_compressed(out / 'warm_states.npz', state_digest=np.array(keys),
                        q_index=np.array([store[k][0] for k in keys], int),
                        N_index=np.array([store[k][1] for k in keys], int),
                        q=np.array([v for _, v in sorted(qs.values(), key=lambda t: t[0])]),
                        N=np.array([v for _, v in sorted(Ns.values(), key=lambda t: t[0])]))
    arrays = {f'targets_{k}': v for k, v in targets.items()}
    np.savez_compressed(out / 'route_inputs.npz', truth_desired=truth['desired'], truth_measured=truth['measured'],
                        sample_index=idx, phase=phase, ref_ids=np.array(ctx.ref_ids), **arrays)
    save_csv(out / 'warm_summary.csv', summaries)
    save_csv(out / 'warm_phase_summary.csv', phase_rows)
    save_csv(out / 'warm_per_repeat.csv', repeat_rows)
    save_csv(out / 'warm_route_totals.csv', totals)
    return dict(route_samples=len(idx), repeats=cfg['repeats'], duration_s=float(ctx.route['time'][idx[-1]]),
                routes={k: dict(partition=v['partition'], signal=v['signal'], commands=v['commands'],
                                methods=[m for m in v['methods'] if not (m == 'original_global'
                                                                         and v['partition'] != 'joint_space')],
                                held_command_max_deviation=deviations[k],
                                samples_with_changed_loop_inputs=loop_changes[k]) for k, v in ROUTES.items()},
                measured_truth_vs_recorded_tree_max=float(np.max(np.abs(truth['measured'] - ctx.route['tree']))),
                attempts=attempts, passed=passed, methods=summaries, totals=totals,
                stored_states=dict(states=len(keys), unique_q=len(qs), unique_N=len(Ns)))


# ---------------------------------------------------------------- local edits
def local_stage(ctx, cfg, out):
    rng = np.random.default_rng(SEEDS['local'])
    truth = ctx._truth
    starts = np.linspace(0, 4000, cfg['local_starts']).astype(int)
    joint_ids = ctx.independent_ids('joint_space')
    ref_plant = ReducedPlant(ctx.cmg, ctx.mapping['cut_joint_ids'], joint_ids, projection_tolerance_m=1e-12)
    raw, inputs = [], []
    deviation_max = 0.
    for label, coords in LOCAL_CONDITIONS:
        for block, k in enumerate(starts):
            ref_plant.initialize(truth[k])
            q0 = ctx.full_state(truth[k])
            u = np.array([q0[c] for c in joint_ids])
            edits = []
            joint_targets, knowns = [], []
            for step in range(cfg['local_edits']):
                for n_c, c in enumerate(coords):
                    u[joint_ids.index(c)] += LOCAL_AMPLITUDE * np.sin(.65 * (step + 1) + .3 * n_c)
                joint_targets.append(u.copy())
                knowns.append(ref_plant.project(u.copy())[0])
            start_joint = np.array([q0[c] for c in joint_ids])
            start_cyl = np.array([q0[c] for c in ctx.independent_ids('cylinder_space')])
            cyl, deviation = cylinder_commands(ctx, joint_targets, knowns, previous=start_cyl, previous_joint=start_joint)
            deviation_max = max(deviation_max, deviation)
            for step in range(cfg['local_edits']):
                targets = dict(joint_space=joint_targets[step], cylinder_space=cyl[step])
                edits.append((targets, knowns[step]))
                inputs.append(dict(condition=label, block=block, step=step, joint_target=joint_targets[step],
                                   cylinder_target=cyl[step], truth=knowns[step]))
            for partition in PARTITIONS:
                methods = [m for m in LOCAL_METHODS if not (m == 'original_global' and partition != 'joint_space')]
                for repeat in range(cfg['repeats']):
                    solvers = {m: create_solver(m, ctx, q0, partition) for m in methods}
                    for step, (targets, known) in enumerate(edits):
                        target = targets[partition]
                        for name in rng.permutation(methods):
                            row = dict(partition=partition, condition=label, changed=len(coords), method=str(name),
                                       block=block, step=step, repeat=repeat, success=False, error='')
                            t0 = perf_counter()
                            try:
                                q, N, info = solvers[name].step(target)
                                row['time_ms'] = record_time(t0)
                                row.update(fallback=info['fallback'], solved_modules=info['solved_modules'],
                                           skipped_modules=info['skipped_modules'], rcond=sig4(info['rcond']))
                                record_check(row, ctx.reference.check(q, N, target, partition, known=known))
                            except Exception as e:
                                row['time_ms'] = record_time(t0)
                                row['error'] = type(e).__name__ + ': ' + str(e)
                            raw.append(row)
    save_json(out / 'local_inputs.json', inputs)
    save_csv(out / 'local_raw.csv.gz', raw)
    summary = timing_summary(raw, ('partition', 'condition', 'method'))
    save_csv(out / 'local_summary.csv', summary)
    return dict(attempts=len(raw), passed=sum(r['success'] for r in raw), methods=summary,
                amplitude_rad=LOCAL_AMPLITUDE, held_command_max_deviation=deviation_max,
                note='Joint-space edits of the named coordinates (at most 0.018 rad per edit) with all other inputs held '
                     'exactly; cylinder-space targets are the strokes of the same closed configurations (held while their '
                     'derived drivers are unchanged). The truth (validated native projection, 1e-12 m) is used only by '
                     'the branch check.')


# ---------------------------------------------------------------- derivatives
def derivative_stage(ctx, cfg, out):
    rng = np.random.default_rng(SEEDS['derivatives'])
    truth = ctx._truth
    targets, _, _ = route_inputs(ctx)
    routes = {'joint_space': targets['joint_reference'], 'cylinder_space': targets['cylinder_reference']}
    raw, inputs = [], []
    for partition in PARTITIONS:
        comp = ctx.compiled[partition]
        route = routes[partition]
        eligible = [k for k in range(len(route) - 3) if not np.array_equal(route[k + 3], route[k])
                    and np.max(np.abs(np.delete(route[k + 3] - route[k], [0, 5]))) > 0]
        chosen = np.array(eligible)[np.linspace(0, len(eligible) - 1, cfg['derivative_cases']).astype(int)]
        pattern = csr_matrix(np.asarray(comp.structure['sparsity']))
        for case, k in enumerate(chosen):
            g = GlobalLoopGraph(comp)
            q32 = ctx.full_state(truth[k])
            x = np.array([q32[c] for c in g.ids])
            N, info = PACDM(g).mapping(x)
            qa = route[k + 3]
            pred = x[g.passive] + N[g.passive] @ (qa - x[g.active])
            clipped = bool(np.any(pred <= g.lower[g.passive]) or np.any(pred >= g.upper[g.passive]))
            pred = np.clip(pred, g.lower[g.passive] + 1e-10, g.upper[g.passive] - 1e-10)
            inputs.append(dict(partition=partition, case=case, source_index=int(k), target=qa, prediction=pred,
                               prediction_clipped=clipped))
            for repeat in range(cfg['repeats']):
                for method in rng.permutation(DERIVATIVE_METHODS):
                    analytic = method.startswith('analytic')
                    sparse = method.endswith('sparse') or method.endswith('colored')
                    fd = None if analytic else ('2-point' if method.startswith('fd2') else '3-point')
                    graph = GlobalLoopGraph(comp)
                    row = dict(partition=partition, method=str(method), case=case, repeat=repeat, source_index=int(k),
                               success=False, error='')
                    start = perf_counter()
                    try:
                        full, res, oracle = trf_solve(graph, qa, pred.copy(), analytic=analytic, sparse=sparse, fd=fd,
                                                      jac_sparsity=pattern if (sparse and not analytic) else None)
                        row['time_ms'] = record_time(start)
                        row['evaluations'] = oracle.evaluations
                        row['scipy_nfev'] = int(res.nfev)
                        Nn, mi = PACDM(graph).mapping(full)
                        take = np.array([graph.ids.index(c) for c in ctx.ref_ids])
                        chk = ctx.reference.check(full[take], Nn[take], qa, partition, known=truth[k + 3]) if Nn is not None \
                            else dict(success=False)
                        record_check(row, chk)
                        row['success'] = bool(chk['success'] and res.success and mi['success'])
                    except Exception as e:
                        row.setdefault('time_ms', record_time(start))
                        row['error'] = type(e).__name__ + ': ' + str(e)
                    raw.append(row)
    save_json(out / 'derivative_inputs.json', inputs)
    save_csv(out / 'derivative_raw.csv', raw)
    summary = timing_summary(raw, ('partition', 'method'))
    save_csv(out / 'derivative_summary.csv', summary)
    return dict(cases=cfg['derivative_cases'], repeats=cfg['repeats'], attempts=len(raw),
                passed=sum(r['success'] for r in raw), methods=summary,
                note='Corrector-only timing (TRF), 0.03 s look-ahead, identical formulation, bounds, tolerances and '
                     'prediction within each pair; colored differences receive the generated sparsity pattern.')


# ---------------------------------------------------------------- dynamics
def dynamics_stage(ctx, cfg, out):
    rng = np.random.default_rng(SEEDS['dynamics'])
    truth = ctx._truth
    dyn = Dynamics(ctx)
    peaks = EFFORT_RANGE
    indices = np.linspace(20, len(truth) - 20, cfg['dynamics_states']).astype(int)
    raw, states, curvature = [], [], []
    for case, k in enumerate(indices):
        q32 = ctx.full_state(truth[k])
        ua = rng.uniform(-.3, .3, 7)
        motors = rng.uniform(-1., 1., 8) * peaks
        wrench = np.r_[rng.uniform(-5e3, 5e3, 3), rng.uniform(-2e3, 2e3, 3)]
        results = {}
        for partition in PARTITIONS:
            row = dict(case=case, source_index=int(k), phase=int(ctx.route['phase'][k]), partition=partition,
                       success=False, error='')
            try:
                if partition == 'joint_space':
                    data, st = dyn.witness(partition, q32, ua, motors, wrench)
                    v_joint = st['v_all']
                    ids_joint = ctx.compiled['joint_space'].structure['coordinate_ids']
                else:
                    ids_c = ctx.compiled['cylinder_space'].structure['coordinate_ids']
                    v_named = dict(zip(ids_joint, v_joint))
                    v_c = np.array([v_named[c] for c in ids_c])
                    data, st = dyn.witness(partition, q32, None, motors, wrench, joint_velocity=v_c)
                results[partition] = st
                row.update(data)
                row['success'] = bool(data['pacdm_vs_kkt_force'] < 1e-10 and data['pacdm_vs_native_force'] < 1e-10
                                      and data['pacdm_constraint_acceleration_m_s2'] < 2e-6
                                      and data['pacdm_tangent_dynamics_relative'] < 1e-9
                                      and data['numpy_vs_native_mass'] < 1e-8 and data['numpy_vs_native_jacobian'] < 1e-9
                                      and data['tangent_residual_native'] < 1e-9 and data['closure_gap_native_m'] < 1e-9
                                      and data['min_reduced_inertia_eigenvalue'] > 0)
            except Exception as e:
                row['error'] = type(e).__name__ + ': ' + str(e) + ' ' + traceback.format_exc(limit=2)
            raw.append(row)
        if len(results) == 2:
            j, c = results['joint_space'], results['cylinder_space']
            inv_force = normmax(j['M_native'] @ (c['a_ref'] - j['a_ref'])) / j['scale']
            inv = relative(c['a_ref'], j['a_ref'])
            vel = relative(c['v_ref'], j['v_ref'])
            raw.append(dict(case=case, source_index=int(k), phase=int(ctx.route['phase'][k]), partition='invariance',
                            partition_acceleration_force=inv_force, partition_acceleration_relative=inv,
                            partition_velocity_relative=vel, success=bool(inv_force < 1e-10 and vel < 1e-10), error=''))
        states.append(dict(case=case, source_index=int(k), tree=truth[k], ua=ua, motors=motors, wrench=wrench))
        if case < cfg['curvature_states']:
            comp = ctx.compiled['joint_space']
            g = GlobalLoopGraph(comp)
            x = np.array([q32[c] for c in g.ids])
            ref_rows = np.array([g.ids.index(c) for c in ctx.ref_ids])
            aa = rng.uniform(-.3, .3, 7)
            for speed in (.5, 1., 2., 4.):
                N, v, c, info = mapping_curvature(g, x, ua * speed)
                gamma = dyn.native.gamma(x[ref_rows], v[ref_rows])
                Jn = dyn.native.closure_jacobian(x[ref_rows])
                full = normmax(Jn @ (N[ref_rows] @ aa + c[ref_rows]) + gamma)
                omitted = normmax(Jn @ (N[ref_rows] @ aa) + gamma)
                curvature.append(dict(case=case, speed_scale=speed, velocity_product_term_max=normmax(c),
                                      with_term_residual_m_s2=full, omitted_term_residual_m_s2=omitted,
                                      success=bool(full < 2e-6)))
    save_csv(out / 'dynamics_raw.csv', raw)
    save_json(out / 'dynamics_states.json', states)
    save_csv(out / 'curvature_magnitude.csv', curvature)
    rows = [r for r in raw if r['partition'] in PARTITIONS]
    inv = [r for r in raw if r['partition'] == 'invariance']
    cols = ['pacdm_vs_kkt_force', 'pacdm_vs_native_force', 'kkt_vs_native_force', 'reduced_inertia_condition',
            'pacdm_vs_kkt_relative', 'pacdm_vs_native_relative', 'kkt_vs_native_relative',
            'map_pacdm_vs_native', 'native_map_reduction_vs_native_relative', 'native_map_reduction_vs_native_force',
            'pacdm_constraint_acceleration_m_s2', 'pacdm_tangent_dynamics_relative',
            'native_constraint_acceleration_m_s2', 'native_tangent_dynamics_relative', 'numpy_vs_native_mass',
            'numpy_vs_native_bias', 'numpy_vs_native_jacobian', 'numpy_vs_native_gamma', 'tangent_residual_native',
            'closure_gap_native_m', 'identity_virtual_power_W', 'identity_kkt_solve_residual']
    maxima = {p: {c: max((r[c] for r in rows if r['partition'] == p and r.get(c) is not None), default=None) for c in cols}
              for p in PARTITIONS}
    return dict(attempts=len(raw), passed=sum(r['success'] for r in raw), maxima=maxima,
                invariance_force_max=max((r['partition_acceleration_force'] for r in inv), default=None),
                invariance_relative_max=max((r['partition_acceleration_relative'] for r in inv), default=None),
                invariance_velocity_max=max((r['partition_velocity_relative'] for r in inv), default=None),
                curvature_attempts=len(curvature), curvature_passed=sum(r['success'] for r in curvature),
                note='Instantaneous rigid-body constrained dynamics at closed route states; no hydraulics, soil, '
                     'friction or armature on any route.')


# ---------------------------------------------------------------- main
STAGES = ['pipeline', 'checks', 'evaluators', 'warm', 'local', 'derivatives', 'dynamics']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=list(PROFILES), default='full')
    p.add_argument('--out', default='results_local')
    p.add_argument('--stages', nargs='+', default=['all'], choices=['all'] + STAGES)
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Output directory is not empty. Use a fresh --out; previous results are never overwritten.')
    stages = list(STAGES) if 'all' in a.stages else a.stages
    out.mkdir(parents=True, exist_ok=True)
    cfg = PROFILES[a.profile].copy()
    protocol = dict(profile=a.profile, configuration=cfg, stages=stages, routes=ROUTES, method_table=METHOD_TABLE,
                    local_methods=LOCAL_METHODS, derivative_methods=DERIVATIVE_METHODS, local_conditions=LOCAL_CONDITIONS,
                    local_amplitude_rad=LOCAL_AMPLITUDE, phase_groups=PHASE_GROUPS, seeds=SEEDS, effort_range=EFFORT_RANGE,
                    acceptance=dict(closure_gap_m=1e-8, target_error=1e-12, tangent_residual=1e-8, map_error=1e-8,
                                    branch_error=1e-6, passive_rank='full (16)', branch_window='physical input'),
                    recording=dict(time_ms='rounded to 1e-4 ms at record time; all summaries use the recorded value',
                                   error_metrics='rounded up to four significant digits; acceptance uses full precision',
                                   state_digest='SHA-1 over float64 q and N, first 12 hex digits'),
                    pacdm=dict(correct_maxiter=12, polish='original 5e-13 polish, 6 iterations',
                               stopping='unchanged PACDM rules'),
                    trf=dict(ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=1500, x_scale=1.),
                    native=dict(tolerance_m=1e-12, source='validated backend ReducedPlant.project'),
                    blas_threads=1,
                    timing_scope='route/local: correction + output map; derivatives: TRF corrector only; setup, truth and '
                                 'acceptance excluded equally',
                    note='Predefined finite suite; all attempts and unfavourable baselines retained.')
    save_json(out / 'protocol.json', protocol)
    save_json(out / 'source_hashes.json', {str(f.relative_to(ROOT)).replace('\\', '/'): sha(f) for f in ROOT.rglob('*')
                                            if f.is_file() and (f.suffix == '.py' or f.parent == ROOT / 'inputs'
                                                                or f.parent == ROOT / 'original/v21/data')
                                            and '__pycache__' not in str(f) and not f.is_relative_to(out)
                                            and not str(f.relative_to(ROOT)).startswith('results')})
    summary = dict(status='running', profile=a.profile, stages={}, protocol_sha256=sha(out / 'protocol.json'))
    with threadpool_limits(limits=1):
        save_json(out / 'environment.json', environment())
        start_all = perf_counter()
        ctx = Context()
        save_json(out / 'compiled_cmg_joint_space.json', ctx.compiled['joint_space'].cmg)
        save_json(out / 'compiled_plan_joint_space.json', ctx.compiled['joint_space'].plan)
        save_json(out / 'compiled_cmg_cylinder_space.json', ctx.compiled['cylinder_space'].cmg)
        save_json(out / 'compiled_plan_cylinder_space.json', ctx.compiled['cylinder_space'].plan)
        t0 = perf_counter()
        ctx._truth = route_truth(ctx, ctx.joint_route())
        ctx._truth_measured = route_truth(ctx, np.asarray(ctx.route['measured'], float))
        print('route truth', round(perf_counter() - t0, 1), 's', flush=True)
        for stage in stages:
            print('STAGE', stage, flush=True)
            start = perf_counter()
            try:
                r = dict(pipeline=lambda: pipeline_stage(ctx, cfg, out), checks=lambda: checks_stage(ctx, out),
                         evaluators=lambda: evaluators_stage(ctx, cfg, out), warm=lambda: warm_stage(ctx, cfg, out),
                         local=lambda: local_stage(ctx, cfg, out), derivatives=lambda: derivative_stage(ctx, cfg, out),
                         dynamics=lambda: dynamics_stage(ctx, cfg, out))[stage]()
                r['stage_wall_s'] = perf_counter() - start
                summary['stages'][stage] = r
                save_json(out / 'summary.json', summary)
                print('DONE', stage, 'seconds', round(r['stage_wall_s'], 1), 'passed', r.get('passed'),
                      '/', r.get('attempts', r.get('checks', r.get('tests'))), flush=True)
            except Exception as e:
                summary.update(status='error', error=type(e).__name__ + ': ' + str(e), traceback=traceback.format_exc())
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
    from .reporting import generate_report
    generate_report(out)
    save_json(out / 'RUN_MANIFEST.json', {str(f.relative_to(out)).replace('\\', '/'): sha(f) for f in out.rglob('*')
                                           if f.is_file() and f.name != 'RUN_MANIFEST.json'})
    print('RESULT:', summary['status'], 'OUTPUT:', out, flush=True)
    return 0 if not failures else 2
