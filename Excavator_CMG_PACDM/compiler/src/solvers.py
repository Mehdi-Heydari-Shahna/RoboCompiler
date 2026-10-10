"""Closure solvers compared in the benchmark, behind one interface.

``step(target)`` receives the partition's independent coordinates and returns

* ``q``: the 23 coordinates of the reference tree (the validated Pinocchio
  backend's depth-first order, ``REF_IDS``),
* ``N``: the 23 x 7 differential map of those coordinates with respect to the
  independent coordinates (partition order),
* ``info``: solved/skipped modules, fallback flag, rcond, tangent residual.

All nonlinear PACDM correct/acquire/mapping calls use the unchanged original
core (``original/v21/pacdm.py``). No inverse-kinematics formula, learned or
precomputed solution is used.

Reuse modes (bit-exact comparisons only; no tolerance):

* ``whole``: return the previous result when the whole input vector is
  unchanged (the cheapest possible reuse, available to every solver);
* ``loopfree``: return the previous loop solution when only independent
  coordinates that are inputs of no loop module changed (slew, rotator). This
  uses the compiled structure but no decomposition, so the benchmark can
  credit loop-free exclusion and loop decomposition separately;
* modular solvers reuse a module when its own inputs are unchanged.

Every solver commits its new state only after the whole step succeeded.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from . import bootstrap  # noqa: F401
from pacdm import PACDM
from excavator_pacdm import compile_pacdm
from plant import ReducedPlant  # validated Pinocchio backend (unchanged)
from .loop_graph import GlobalLoopGraph, ModuleLoopGraph

CORRECT_MAXITER = 12      # original online controller (_assemble) setting
REUSE_MODES = (None, 'whole', 'loopfree')


def polish(graph, q, rows):
    """Verbatim logic of the original ``TrackedArmController._polish``
    (v26/tracked_arm.py, "Original accepted 5e-13 precision after PACDM physical
    correction"); that module imports MuJoCo, so the ten lines are reproduced
    here and compared with the original in tests/test_excavator.py."""
    q = q.copy()
    for _ in range(6):
        residual, jacobian, _ = graph.residual(q)
        if np.max(np.abs(residual)) < 5e-13:
            break
        q[graph.passive] -= np.linalg.solve(jacobian[np.ix_(rows, graph.passive)], residual[rows])
    if np.any(q < graph.lower) or np.any(q > graph.upper):
        raise ValueError("PACDM polish left the original numerical branch guards")
    return q


def loop_input_positions(comp):
    """Partition positions of the independent coordinates that drive at least one loop module.

    The complement (slew and rotator here) lies in no fundamental cycle; the
    compiled loop residuals do not depend on it at all (LCA-relative evaluation).
    """
    s = comp.structure
    col_of = {k: j for j, k in enumerate(s['independent'])}
    return np.array(sorted({col_of[k] for m in s['modules'] for k in m['input_indices']}), int)


def command_drivers(ctx):
    """For each cylinder-space stroke command: the joint-space inputs of the joint-space
    module that contains the stroke as a dependent coordinate (derived, not declared)."""
    s = ctx.compiled['joint_space'].structure
    ids = list(s['coordinate_ids'])
    drivers = {}
    for name in ctx.independent_ids('cylinder_space'):
        if name in ctx.independent_ids('joint_space'):
            continue
        k = ids.index(name)
        module = [m for m in s['modules'] if k in m['dependent_indices']]
        if len(module) != 1:
            raise RuntimeError(f'Stroke {name} must be dependent in exactly one joint-space module')
        drivers[name] = tuple(ids[i] for i in module[0]['input_indices'])
    return drivers


def _unchanged(mode, new, old, loop_pos):
    if mode == 'whole':
        return bool(np.array_equal(new, old))
    if mode == 'loopfree':
        return bool(np.array_equal(new[loop_pos], old[loop_pos]))
    return False


class _Output:
    """Row extraction from a solver's coordinate order to the reference order."""

    def __init__(self, ids, ref_ids, independent_ids):
        self.take = np.array([ids.index(k) for k in ref_ids], int)
        self.ncol = len(independent_ids)


def _skip_info(info):
    return dict(fallback=False, solved_modules=0, skipped_modules=1, rcond=info.get('rcond'),
                tangent_residual=info.get('tangent_residual'))


class OriginalGlobal:
    """Original revolute-cut adapter + unchanged PACDM (joint space only, no reuse)."""
    name = 'original_global'

    def __init__(self, ctx, q32, partition):
        if partition != 'joint_space':
            raise ValueError('The original adapter hard-codes the joint-space independent set')
        self.g, self.solver = compile_pacdm(ctx.cmg, ctx.mapping, ctx.mapping['tree_joint_ids'])
        self.out = _Output(self.g.ids, ctx.ref_ids, ctx.independent_ids(partition))
        self.x = np.array([q32[k] for k in self.g.ids])
        self.N, self.info = self.solver.mapping(self.x)
        if self.N is None:
            raise RuntimeError(self.info)

    def step(self, active):
        g = self.g
        active = np.asarray(active, float)
        pred = self.x[g.passive] + self.N[g.passive] @ (active - self.x[g.active])
        rows = np.array(self.info['rows'])
        x, ci = self.solver.correct(active, pred, None, rows, maxiter=CORRECT_MAXITER)
        fallback = False
        if not ci['success']:
            x, ci = self.solver.acquire(active, self.x)
            fallback = True
        if ci['success']:
            x = polish(g, x, rows)
        N, info = self.solver.mapping(x)
        if not ci['success'] or not info['success']:
            raise RuntimeError(str((ci, info)))
        self.x, self.N, self.info = x, N, info
        return x[self.out.take].copy(), N[self.out.take].copy(), dict(
            fallback=fallback, solved_modules=1, skipped_modules=0, rcond=info['rcond'],
            tangent_residual=info['tangent_residual'])


class TRFOracle:
    """Counts EVERY fresh evaluation, including finite-difference perturbations.

    An analytic residual/Jacobian pair is computed once at an identical x and
    shared by ``fun`` and ``jac``; the FD residual path never forms a Jacobian.
    """

    def __init__(self, graph, active, analytic=True, sparse=False, budget=1500):
        self.g = graph
        self.active = np.asarray(active, float).copy()
        self.analytic, self.sparse, self.budget = analytic, sparse, budget
        self.last = None
        self.r = self.J = None
        self.evaluations = 0

    def full(self, p):
        x = np.zeros(self.g.n)
        x[self.g.active] = self.active
        x[self.g.passive] = p
        return x

    def _evaluate(self, p):
        if self.last is not None and np.array_equal(p, self.last):
            return
        if self.evaluations >= self.budget:
            raise RuntimeError('Common fresh-evaluation budget exceeded')
        self.last = np.asarray(p, float).copy()
        self.evaluations += 1
        x = self.full(p)
        if self.analytic:
            self.r, J, _ = self.g.residual(x)
            self.J = J[:, self.g.passive]
        else:
            self.r = self.g.residual_only(x)
            self.J = None

    def fun(self, p):
        self._evaluate(p)
        return self.r.copy()

    def jac(self, p):
        self._evaluate(p)
        return csr_matrix(self.J) if self.sparse else self.J.copy()


def trf_solve(graph, active, pred, *, analytic=True, sparse=False, fd=None, jac_sparsity=None):
    oracle = TRFOracle(graph, active, analytic=analytic, sparse=sparse)
    kwargs = {}
    if jac_sparsity is not None:
        kwargs['jac_sparsity'] = jac_sparsity
    res = least_squares(oracle.fun, pred, jac=oracle.jac if analytic else fd, method='trf',
                        bounds=(graph.lower[graph.passive], graph.upper[graph.passive]),
                        tr_solver='lsmr' if sparse else 'exact', ftol=1e-11, xtol=1e-11, gtol=1e-11,
                        max_nfev=1500, x_scale=1., **kwargs)
    return oracle.full(res.x), res, oracle


def _clip_prediction(graph, pred):
    return np.clip(pred, graph.lower[graph.passive] + 1e-10, graph.upper[graph.passive] - 1e-10)


class CompiledGlobal:
    """Generated evaluator, all loops as one PACDM (or TRF) problem."""

    def __init__(self, ctx, q32, partition, *, corrector='pacdm', reuse=None, name=None):
        if reuse not in REUSE_MODES:
            raise ValueError(reuse)
        comp = ctx.compiled[partition]
        self.name = name or ('compiled_global' if corrector == 'pacdm' else 'trf_global')
        self.g = GlobalLoopGraph(comp)
        self.solver = PACDM(self.g)
        self.corrector, self.reuse = corrector, reuse
        self.loop_pos = loop_input_positions(comp)
        self.out = _Output(self.g.ids, ctx.ref_ids, ctx.independent_ids(partition))
        self.x = np.array([q32[k] for k in self.g.ids])
        self.N, self.info = self.solver.mapping(self.x)
        if self.N is None:
            raise RuntimeError(self.info)

    def step(self, active):
        g = self.g
        active = np.asarray(active, float)
        if _unchanged(self.reuse, active, self.x[g.active], self.loop_pos):
            # Loop residuals do not depend on loop-free coordinates, so the dependent
            # coordinates and the map are those of the previous solution (bit-exact).
            x = self.x.copy()
            x[g.active] = active
            self.x = x
            return x[self.out.take].copy(), self.N[self.out.take].copy(), _skip_info(self.info)
        pred = self.x[g.passive] + self.N[g.passive] @ (active - self.x[g.active])
        fallback = False
        if self.corrector == 'trf':
            x, res, _ = trf_solve(g, active, _clip_prediction(g, pred))
            ci = dict(success=bool(res.success))
        else:
            rows = np.array(self.info['rows'])
            x, ci = self.solver.correct(active, pred, None, rows, maxiter=CORRECT_MAXITER)
            if not ci['success']:
                x, ci = self.solver.acquire(active, self.x)
                fallback = True
            if ci['success']:
                x = polish(g, x, rows)
        N, info = self.solver.mapping(x)
        if not ci['success'] or not info['success']:
            raise RuntimeError(str((ci, info)))
        self.x, self.N, self.info = x, N, info
        return x[self.out.take].copy(), N[self.out.take].copy(), dict(
            fallback=fallback, solved_modules=1, skipped_modules=0, rcond=info['rcond'],
            tangent_residual=info['tangent_residual'])


class CompiledModular:
    """One PACDM (or TRF) problem per generated loop module, exact per-module reuse."""

    def __init__(self, ctx, q32, partition, *, predictor=True, reuse=True, corrector='pacdm', name=None):
        comp = ctx.compiled[partition]
        s = comp.structure
        self.name = name or 'compiled_modular'
        self.predictor, self.reuse, self.corrector = predictor, reuse, corrector
        ids = list(s['coordinate_ids'])
        self.n = len(ids)
        self.independent = np.array(s['independent'], int)
        self.col_of = {k: j for j, k in enumerate(self.independent)}
        self.out = _Output(ids, ctx.ref_ids, ctx.independent_ids(partition))
        q = np.array([q32[k] for k in ids])
        self.q = q
        self.modules = []
        for m in s['modules']:
            g = ModuleLoopGraph(comp, m)
            x = q[g.global_index].copy()
            solver = PACDM(g)
            N, info = solver.mapping(x)
            if N is None:
                raise RuntimeError(info)
            cols = np.array([self.col_of[k] for k in m['input_indices']], int)
            self.modules.append(dict(g=g, solver=solver, x=x, N=N, info=info, cols=cols,
                                     dep=np.array(m['dependent_indices'], int)))
        self.N = np.zeros((self.n, len(self.independent)))
        self.N[self.independent, np.arange(len(self.independent))] = 1.
        for m in self.modules:
            self.N[np.ix_(m['dep'], m['cols'])] = m['N'][m['g'].passive]

    def step(self, active):
        active = np.asarray(active, float)
        q = self.q.copy()
        N = self.N.copy()
        q[self.independent] = active
        updates = []
        fallback = False
        for m in self.modules:
            g = m['g']
            qa = active[m['cols']]
            if self.reuse and np.array_equal(qa, m['x'][g.active]):
                continue
            pred = m['x'][g.passive].copy()
            if self.predictor:
                pred += m['N'][g.passive] @ (qa - m['x'][g.active])
            if self.corrector == 'trf':
                x, res, _ = trf_solve(g, qa, _clip_prediction(g, pred))
                ci = dict(success=bool(res.success))
            else:
                rows = np.array(m['info']['rows'])
                x, ci = m['solver'].correct(qa, pred, None, rows, maxiter=CORRECT_MAXITER)
                if not ci['success']:
                    x, ci = m['solver'].acquire(qa, m['x'])
                    fallback = True
                if ci['success']:
                    x = polish(g, x, rows)
            Nm, info = m['solver'].mapping(x)
            if not ci['success'] or not info['success']:
                raise RuntimeError(str((ci, info)))
            updates.append((m, x, Nm, info))
            q[m['dep']] = x[g.passive]
            N[np.ix_(m['dep'], m['cols'])] = Nm[g.passive]
        for m, x, Nm, info in updates:          # commit only after every module succeeded
            m.update(x=x, N=Nm, info=info)
        self.q, self.N = q, N
        rc = min(m['info']['rcond'] for m in self.modules)
        tr = max(m['info']['tangent_residual'] for m in self.modules)
        return q[self.out.take].copy(), N[self.out.take].copy(), dict(
            fallback=fallback, solved_modules=len(updates), skipped_modules=len(self.modules) - len(updates),
            rcond=rc, tangent_residual=tr)


class NativeNewton:
    """Validated Pinocchio backend: native two-point closures, Newton least squares.

    ``ReducedPlant.project`` of the original backend, unchanged: first-order
    predictor along its last tangent map, Newton steps on the 54 native point
    rows, passive-rank check and least-squares tangent map; stopping tolerance
    1e-12 m, as in the original backend. The harness adds the same bit-exact reuse modes as
    the compiled global route and returns cached values without calling the
    plant (the plant's own identical-input path refreshes the joint Jacobians,
    about 0.4 ms, and is therefore bypassed). For ``loopfree`` the native closure
    is invariant to slew and rotator in exact arithmetic (the world-frame point
    residual rotates rigidly with the arm); the acceptance gate verifies the
    reused state and map at the new coordinates.
    """

    def __init__(self, ctx, q32, partition, tolerance=1e-12, reuse='whole', name=None):
        if reuse not in REUSE_MODES:
            raise ValueError(reuse)
        self.name = name or 'native_newton'
        self.reuse = reuse
        self.loop_pos = loop_input_positions(ctx.compiled[partition])
        self.plant = ReducedPlant(ctx.cmg, ctx.mapping['cut_joint_ids'], ctx.independent_ids(partition),
                                  projection_tolerance_m=tolerance)
        if list(self.plant.model.tree_ids) != list(ctx.ref_ids):
            raise RuntimeError('Reference order must be the native plant order')
        self.plant.initialize(np.array([q32[k] for k in ctx.ref_ids]))
        self.q = self.plant.q_cache.copy()
        self.N = self.plant.N_cache.copy()
        self.u = self.plant.u_cache.copy()

    def step(self, active):
        u = np.asarray(active, float)
        if _unchanged(self.reuse, u, self.u, self.loop_pos):
            q = self.q.copy()
            q[self.plant.active] = u
            self.q, self.u = q, u.copy()
            return q.copy(), self.N.copy(), _skip_info({})
        before = self.plant.stats['projections']
        q, J, N, err = self.plant.project(u)
        solved = self.plant.stats['projections'] - before
        self.q, self.N, self.u = q.copy(), N.copy(), u.copy()
        return q.copy(), N.copy(), dict(fallback=False, solved_modules=solved, skipped_modules=1 - solved,
                                        rcond=None, tangent_residual=None)


class NativeModular:
    """The compiler's loop-module schedule applied to the native Pinocchio closure.

    Same native model, point rows, Newton least squares and tolerance as
    ``NativeNewton``; only the rows of native cuts belonging to modules whose
    inputs changed (bit-exact comparison) and those modules' dependent tree
    coordinates enter the solve. Unchanged modules keep their coordinates and
    map rows. A native cut belongs to the compiled module that contains its
    joint as a dependent coordinate (cycle membership is independent of which
    joint of a loop is cut). The native constraint Jacobian is evaluated for all
    54 rows and the changed modules' rows are selected (the plant has no
    per-constraint Jacobian call), so the schedule's native gain is not
    inflated by a cheaper kernel.
    """
    name = 'native_modular'

    def __init__(self, ctx, q32, partition, tolerance=1e-12):
        import pinocchio as pin
        self.pin = pin
        self.plant = ReducedPlant(ctx.cmg, ctx.mapping['cut_joint_ids'], ctx.independent_ids(partition),
                                  projection_tolerance_m=tolerance)
        pm = self.plant.model
        if list(pm.tree_ids) != list(ctx.ref_ids):
            raise RuntimeError('Reference order must be the native plant order')
        self.pm, self.tol = pm, tolerance
        ids = list(ctx.ref_ids)
        self.active = self.plant.active
        comp = ctx.compiled[partition]
        cids = comp.structure['coordinate_ids']
        self.modules = []
        cut_ids = list(ctx.mapping['cut_joint_ids'])
        for m in comp.structure['modules']:
            dep_names = [cids[i] for i in m['dependent_indices']]
            inp_names = [cids[i] for i in m['input_indices']]
            cuts = [c for c, jid in enumerate(cut_ids) if jid in dep_names]
            rows = np.concatenate([np.arange(6 * c, 6 * c + 6) for c in cuts]) if cuts else np.array([], int)
            dep = np.array([ids.index(k) for k in dep_names if k in ids], int)
            inputs = np.array([ids.index(k) for k in inp_names], int)
            frames = [pm.endpoint_frames[2 * c + k] for c in cuts for k in (0, 1)]
            input_pos = np.array([list(self.active).index(i) for i in inputs], int)
            self.modules.append(dict(rows=rows, dep=dep, inputs=inputs, input_pos=input_pos, frames=frames, cuts=cuts))
        covered = sorted(int(c) for m in self.modules for c in m['cuts'])
        if covered != list(range(len(cut_ids))):
            raise RuntimeError('Every native cut must belong to exactly one compiled module')
        q = np.array([q32[k] for k in ids])
        if np.max(np.abs(pm.closure_residual(q))) > 1e-9:
            raise RuntimeError('Initial state is not closed')
        self.q = q
        J = pm.closure_jacobian(q)
        self.N = np.zeros((len(ids), len(self.active)))
        self.N[self.active, np.arange(len(self.active))] = 1.
        for m in self.modules:
            self.N[m['dep']] = self._map(m, J)

    def _map(self, m, J):
        block = J[np.ix_(m['rows'], m['dep'])]
        if np.linalg.matrix_rank(block, tol=1e-9) != len(m['dep']):   # same test as the validated plant
            raise RuntimeError('Native module lost passive rank')
        return np.linalg.lstsq(block, -J[np.ix_(m['rows'], self.active)], rcond=None)[0]

    def _residual(self, q, frames):
        pin, pm = self.pin, self.pm
        pin.forwardKinematics(pm.model, pm.data, q)
        pin.updateFramePlacements(pm.model, pm.data)
        oMf = pm.data.oMf
        return np.concatenate([oMf[a].translation - oMf[b].translation for a, b in frames])

    def step(self, active):
        active = np.asarray(active, float)
        changed = [m for m in self.modules if not np.array_equal(active[m['input_pos']], self.q[m['inputs']])]
        previous = self.q[self.active].copy()
        q = self.q.copy()
        N = self.N.copy()
        q[self.active] = active
        if changed:
            rows = np.concatenate([m['rows'] for m in changed])
            dep = np.concatenate([m['dep'] for m in changed])
            frames = [f for m in changed for f in m['frames']]
            q[dep] += self.N[dep] @ (active - previous)
            for iteration in range(26):
                residual = self._residual(q, frames)
                if float(np.max(np.abs(residual))) <= self.tol:
                    break
                if iteration == 25:
                    raise RuntimeError('Native modular projection failed')
                J = self.pm.closure_jacobian(q)
                q[dep] -= np.linalg.lstsq(J[np.ix_(rows, dep)], residual, rcond=None)[0]
            J = self.pm.closure_jacobian(q)
            for m in changed:
                N[m['dep']] = self._map(m, J)
        self.q, self.N = q, N          # commit only after the whole step succeeded
        return q.copy(), N.copy(), dict(fallback=False, solved_modules=len(changed),
                                        skipped_modules=len(self.modules) - len(changed),
                                        rcond=None, tangent_residual=None)


SOLVERS = {
    'original_global': lambda c, q, p: OriginalGlobal(c, q, p),
    'compiled_global': lambda c, q, p: CompiledGlobal(c, q, p, reuse=None, name='compiled_global'),
    'compiled_global_reuse': lambda c, q, p: CompiledGlobal(c, q, p, reuse='whole', name='compiled_global_reuse'),
    'compiled_global_loopfree': lambda c, q, p: CompiledGlobal(c, q, p, reuse='loopfree',
                                                               name='compiled_global_loopfree'),
    'compiled_modular': lambda c, q, p: CompiledModular(c, q, p),
    'modular_no_reuse': lambda c, q, p: CompiledModular(c, q, p, reuse=False, name='modular_no_reuse'),
    'modular_no_predictor': lambda c, q, p: CompiledModular(c, q, p, predictor=False, name='modular_no_predictor'),
    'trf_global': lambda c, q, p: CompiledGlobal(c, q, p, corrector='trf', reuse='whole', name='trf_global'),
    'trf_global_loopfree': lambda c, q, p: CompiledGlobal(c, q, p, corrector='trf', reuse='loopfree',
                                                          name='trf_global_loopfree'),
    'trf_modular': lambda c, q, p: CompiledModular(c, q, p, corrector='trf', name='trf_modular'),
    'native_newton': lambda c, q, p: NativeNewton(c, q, p, reuse='whole', name='native_newton'),
    'native_newton_loopfree': lambda c, q, p: NativeNewton(c, q, p, reuse='loopfree', name='native_newton_loopfree'),
    'native_modular': lambda c, q, p: NativeModular(c, q, p),
}

# What each route uses (for the method table): corrector, evaluator, reuse / schedule.
METHOD_TABLE = {
    'original_global': ('PACDM', 'original adapter', 'none (one global problem)'),
    'compiled_global': ('PACDM', 'generated', 'none (one global problem)'),
    'compiled_global_reuse': ('PACDM', 'generated', 'whole-input reuse'),
    'compiled_global_loopfree': ('PACDM', 'generated', 'loop-free exclusion (one global loop problem)'),
    'compiled_modular': ('PACDM', 'generated', 'loop modules, per-module reuse'),
    'modular_no_reuse': ('PACDM', 'generated', 'loop modules, every module solved'),
    'modular_no_predictor': ('PACDM', 'generated', 'loop modules, reuse, no tangent predictor'),
    'trf_global': ('TRF (analytic Jacobian)', 'generated', 'whole-input reuse'),
    'trf_global_loopfree': ('TRF (analytic Jacobian)', 'generated', 'loop-free exclusion'),
    'trf_modular': ('TRF (analytic Jacobian)', 'generated', 'loop modules, per-module reuse'),
    'native_newton': ('Newton (native)', 'Pinocchio 3.8.0', 'whole-input reuse'),
    'native_newton_loopfree': ('Newton (native)', 'Pinocchio 3.8.0', 'loop-free exclusion'),
    'native_modular': ('Newton (native)', 'Pinocchio 3.8.0', 'loop modules, per-module reuse'),
}


def create_solver(name, ctx, q32, partition):
    try:
        factory = SOLVERS[name]
    except KeyError:
        raise ValueError(name) from None
    return factory(ctx, q32, partition)
