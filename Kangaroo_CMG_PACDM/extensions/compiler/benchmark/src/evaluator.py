"""Generated batched SE(3) cut evaluator, loop modules and PACDM task scheduling.

New extension; not functionality attributed to the unchanged PACDM core.

Every nonlinear PACDM correct/acquire/mapping call uses the original unchanged
core (``original/original_v22/pacdm.py``).  The generated evaluator reproduces
the residual convention of the accepted v22 ``CutGraph``: angular-first SE(3)
logarithm of ``inv(minus @ D) @ plus`` with free chart rotations about the cut
frame axes (three for point cuts, two for universal cuts) and Jacobian rows
``inv_left(r) adj(reverse) (E_plus - E_minus)``.

What is generated from the compiled plan (and not authored by hand):

* for every cut, the two directed joint chains below its lowest common
  ancestor (LCA), so joints above the LCA are never visited and the
  common-prefix columns are structurally zero;
* a position-major step table for all chains of the requested cut set, so one
  batched NumPy operation advances every chain by one joint (chains are
  left-padded with exact identity steps);
* a scatter table that places each joint twist, with sign, directly into the
  six rows of its cut and the column of its coordinate (the passive sparsity
  pattern of the plan).

The accepted evaluator instead propagates all 78 body poses from the root and
carries a dense 6 x 140 twist matrix per body.  Equality with the accepted
``CutGraph`` residual and Jacobian is a tested property (tests/ and the
pipeline stage), not an assumption.

Module reuse is exact: a module is skipped only if its motor inputs are
bit-identical to its last accepted inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from . import bootstrap  # noqa: F401
from pacdm import PACDM, inv_left as accepted_inv_left

_I4 = np.eye(4)
# flat positions (in a (3, 4, 4) block) of [c, -s, s, c] for rotations about x, y, z
_CHART_POS = np.array([16 * i + 4 * a + b for i in range(3)
                       for a, b in (((i + 1) % 3, (i + 1) % 3), ((i + 1) % 3, (i + 2) % 3),
                                    ((i + 2) % 3, (i + 1) % 3), ((i + 2) % 3, (i + 2) % 3))])


def _skew(a):
    return np.array([[0., -a[2], a[1]], [a[2], 0., -a[0]], [-a[1], a[0], 0.]])


_SKEW_POS = np.array([1, 2, 3, 5, 6, 7])
_SKEW_SRC = np.array([2, 1, 2, 0, 1, 0])
_SKEW_SIGN = np.array([-1., 1., 1., -1., -1., 1.])


def _skew_batch(v):
    out = np.zeros(v.shape[:-1] + (9,))
    out[..., _SKEW_POS] = v[..., _SKEW_SRC] * _SKEW_SIGN
    return out.reshape(v.shape[:-1] + (3, 3))


def _inv_batch(T):
    out = np.zeros_like(T)
    Rt = np.swapaxes(T[:, :3, :3], 1, 2)
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum('cij,cj->ci', Rt, T[:, :3, 3])
    out[:, 3, 3] = 1.
    return out


_P1, _P2 = np.array([1, 2, 0]), np.array([2, 0, 1])


def _cross(a, b):
    return a[..., _P1] * b[..., _P2] - a[..., _P2] * b[..., _P1]


def _log_batch(T):
    """Batched copy of pacdm.log (angular-first SE(3) logarithm, same branches).

    Uses (I - K/2 + beta K^2) p = p - (w x p)/2 + beta w x (w x p).
    """
    R, p = T[:, :3, :3], T[:, :3, 3]
    u = (R[:, _P2, _P1] - R[:, _P1, _P2]) / 2
    s = np.sqrt((u * u).sum(1))
    theta = np.arctan2(s, np.clip((np.einsum('cii->c', R) - 1) / 2, -1, 1))
    if (np.pi - theta < 1e-6).any():
        raise ValueError('SE3 logarithm near pi branch')
    scale = theta / np.maximum(s, 1e-300)
    scale[s < 1e-14] = 1.
    w = u * scale[:, None]
    th2 = theta * theta
    beta = 1 / 12 + th2 / 720 + th2 * th2 / 30240
    large = theta >= 1e-4
    if large.any():
        t = theta[large]
        beta[large] = (1 - t / 2 / np.tan(t / 2)) / t ** 2
    wp = _cross(w, p)
    return np.concatenate([w, p - wp / 2 + beta[:, None] * _cross(w, wp)], axis=1)


def _inv_left_batch(r):
    """Batched copy of pacdm.inv_left: series branch; exact accepted call otherwise."""
    W, V = _skew_batch(r[:, :3]), _skew_batch(r[:, 3:])
    Z = np.zeros((len(r), 6, 6))
    Z[:, :3, :3] = W
    Z[:, 3:, 3:] = W
    Z[:, 3:, :3] = V
    norm = np.sqrt(np.einsum('cij,cij->c', Z, Z))
    Z2 = Z @ Z
    out = np.eye(6) - Z / 2 + Z2 / 12 - (Z2 @ Z2) / 720
    for c in np.flatnonzero(norm > 1e-3):
        out[c] = accepted_inv_left(r[c])
    return out


def _adj_batch(T):
    R = T[:, :3, :3]
    out = np.zeros((len(T), 6, 6))
    out[:, :3, :3] = R
    out[:, 3:, 3:] = R
    out[:, 3:, :3] = _skew_batch(T[:, :3, 3]) @ R
    return out


class CutSetEvaluator:
    """Generated, batched evaluator for a set of cut records over a column layout.

    ``cuts`` is a list of dictionaries with ``chain1``/``chain2`` (lists of
    directed joint records), ``A``/``B`` (4x4 cut frames on body1/body2),
    ``columns`` (joint id -> column) and ``chart_columns`` (0, 2 or 3 columns).
    """

    def __init__(self, cuts, ncols):
        self.nc = len(cuts)
        self.n = ncols
        chains = [c['chain1'] for c in cuts] + [c['chain2'] for c in cuts]
        self.depth = max([len(ch) for ch in chains] + [1])
        C2 = len(chains)
        D = self.depth
        E = np.tile(_I4, (C2, D, 1, 1))
        L = np.tile(_I4, (C2, D, 1, 1))
        axis = np.zeros((C2, D, 3))
        axis[..., 2] = 1.
        rev = np.zeros((C2, D))
        pri = np.zeros((C2, D))
        col = np.full((C2, D), ncols, dtype=int)  # index ncols -> appended zero
        for k, chain in enumerate(chains):
            offset = D - len(chain)  # right-align: identity padding first
            cols = cuts[k % self.nc]['columns']
            for i, j in enumerate(chain):
                E[k, offset + i] = np.asarray(j['T_BJ'], float)
                F = np.asarray(j['T_FJ'], float)
                Li = np.eye(4)
                Li[:3, :3] = F[:3, :3].T
                Li[:3, 3] = -F[:3, :3].T @ F[:3, 3]
                L[k, offset + i] = Li
                if j['type'] == 'fixed':
                    continue
                axis[k, offset + i] = np.asarray(j['axis'], float)
                if j['type'] == 'revolute':
                    rev[k, offset + i] = 1.
                else:
                    pri[k, offset + i] = 1.
                col[k, offset + i] = cols[j['id']]
        self.E, self.L, self.axis, self.rev, self.pri, self.col = E, L, axis, rev, pri, col
        self.K = _skew_batch(axis)
        self.K2 = self.K @ self.K
        self.A = np.array([c['A'] for c in cuts], float)
        self.B = np.array([c['B'] for c in cuts], float)
        chart = np.full((self.nc, 3), ncols, dtype=int)
        for k, c in enumerate(cuts):
            chart[k, :len(c['chart_columns'])] = c['chart_columns']
        self.chart = chart
        # scatter table (cut, column, sign, source) for the Jacobian difference
        cut_idx, col_idx, sign, src_chain, src_pos = [], [], [], [], []
        for k in range(C2):
            for pos in range(D):
                if col[k, pos] < ncols:
                    cut_idx.append(k % self.nc)
                    col_idx.append(col[k, pos])
                    sign.append(1. if k >= self.nc else -1.)
                    src_chain.append(k)
                    src_pos.append(pos)
        self.s_cut = np.array(cut_idx, int)
        self.s_col = np.array(col_idx, int)
        self.s_sign = np.array(sign)[:, None]
        self.s_src = (np.array(src_chain, int), np.array(src_pos, int))
        cc, ci = np.nonzero(chart < ncols)
        self.c_cut, self.c_axis, self.c_col = cc, ci, chart[cc, ci]
        if len(set(zip(self.s_cut.tolist(), self.s_col.tolist()))) != len(self.s_cut):
            raise ValueError('A joint column appears twice in one cut (paths must be LCA-pruned)')
        self.sparsity = np.zeros((6 * self.nc, ncols), dtype=int)
        for k, c in zip(self.s_cut, self.s_col):
            self.sparsity[6 * k:6 * k + 6, c] = 1
        for k, c in zip(self.c_cut, self.c_col):
            self.sparsity[6 * k:6 * k + 6, c] = 1
        self._eye_chains = np.tile(_I4, (C2, 1, 1))
        self._eye_steps = np.tile(_I4, (C2, D, 1, 1))
        self._eye_charts = np.tile(_I4, (self.nc, 3, 1, 1))
        self._chart_axes, self._ones, self._zeros = np.eye(3), np.ones(3), np.zeros(3)

    def _chains(self, xz, need):
        """Advance every chain one joint per batched step; keep joint frames for twists."""
        C2, D = self.col.shape
        T = self._eye_chains.copy()
        frames = np.empty((C2, D, 4, 4)) if need else None
        v = xz[self.col]
        s, omc = np.sin(v) * self.rev, (1. - np.cos(v)) * self.rev
        M = self._eye_steps.copy()
        M[:, :, :3, :3] += s[..., None, None] * self.K + omc[..., None, None] * self.K2
        M[:, :, :3, 3] = self.axis * (v * self.pri)[..., None]
        ML = M @ self.L
        for pos in range(D):
            T = T @ self.E[:, pos]
            if need:
                frames[:, pos] = T
            T = T @ ML[:, pos]
        return T, frames

    @staticmethod
    def _twists(frames, axis, rev, pri):
        """Spatial twists [R a; p x R a] (revolute) or [0; R a] (prismatic) of stacked joint frames."""
        a = (frames[..., :3, :3] @ axis[..., None])[..., 0]
        return np.concatenate([a * rev[..., None], _cross(frames[..., :3, 3], a) * rev[..., None]
                               + a * pri[..., None]], axis=-1)

    def evaluate(self, x, defects=None, need=True):
        xz = np.append(np.asarray(x, float), 0.)
        T, frames = self._chains(xz, need)
        nc = self.nc
        minus = T[:nc] @ self.A
        chart_frames = np.empty((nc, 3, 4, 4)) if need else None
        th = xz[self.chart]
        c, s = np.cos(th), np.sin(th)
        Rk = self._eye_charts.copy()
        Rk.reshape(nc, 48)[:, _CHART_POS] = np.stack([c, -s, s, c], axis=2).reshape(nc, 12)
        for i in range(3):
            if need:
                chart_frames[:, i] = minus
            minus = minus @ Rk[:, i]
        plus = T[nc:] @ self.B
        if defects is not None:
            minus = minus @ np.asarray(defects, float)
        reverse = _inv_batch(minus)
        delta = reverse @ plus
        r = _log_batch(delta)
        if not need:
            return r.ravel(), None, delta
        diff = np.zeros((nc, 6, self.n))
        src = self.s_src
        diff[self.s_cut, :, self.s_col] = self.s_sign * self._twists(
            frames[src], self.axis[src], self.rev[src], self.pri[src])
        diff[self.c_cut, :, self.c_col] = -self._twists(
            chart_frames[self.c_cut, self.c_axis], self._chart_axes[self.c_axis], self._ones[self.c_axis],
            self._zeros[self.c_axis])
        J = (_inv_left_batch(r) @ _adj_batch(reverse)) @ diff
        return r.ravel(), J.reshape(6 * nc, self.n), delta


def _cut_frames(cut, plan_cut):
    A, B = np.eye(4), np.eye(4)
    A[:3, 3] = cut['point1_m']
    B[:3, 3] = cut['point2_m']
    if cut['type'] == 'universal':
        A[:3, :3] = cut['frame1_R']
        B[:3, :3] = cut['frame2_R']
    else:
        A[:3, :3] = plan_cut['chart_reference']
    return A, B


class _GraphBase:
    """PACDM-compatible graph protocol: n, active, passive, lower, upper, residual."""

    def residual(self, x, defects=None):
        self.calls += 1
        return self.evaluator.evaluate(x, defects, True)

    def residual_only(self, x):
        self.calls += 1
        return self.evaluator.evaluate(x, None, False)[0]


class GeneratedLoopGraph(_GraphBase):
    """Loop cuts over the augmented closure vector (global layout) or a module-local vector.

    Global layout matches the accepted CutGraph: x = [physical (nt), cut charts];
    active = motor coordinates. Module layout: x = [module motors, module passive].
    """

    def __init__(self, comp, cut_indices=None, module=None):
        plan, cmg = comp.plan, comp.cmg
        self.comp = comp
        cuts = list(range(plan['loop_cuts'])) if cut_indices is None else list(cut_indices)
        n_aug = plan['augmented_coordinates']
        jrec = {j['id']: j for j in cmg['joints']}
        ids = cmg['coordinate_ids']
        lower_all = np.full(n_aug, -np.pi)
        upper_all = np.full(n_aug, np.pi)
        for i, k in enumerate(ids):
            lower_all[i] = jrec[k]['limits']['lower']
            upper_all[i] = jrec[k]['limits']['upper']
        if module is None:
            self.columns = np.arange(n_aug)
            self.active = np.asarray(plan['motor_indices'], int)
            self.passive = np.setdiff1d(np.arange(n_aug), self.active)
        else:
            self.columns = np.r_[module['motors'], module['passive']].astype(int)
            self.active = np.arange(len(module['motors']))
            self.passive = np.arange(len(module['motors']), len(self.columns))
        self.n = len(self.columns)
        self.lower, self.upper = lower_all[self.columns], upper_all[self.columns]
        local = {int(g): i for i, g in enumerate(self.columns)}
        records = []
        for k in cuts:
            pc = plan['cut_plans'][k]
            cut = cmg['closures'][k]
            A, B = _cut_frames(cut, pc)
            records.append(dict(chain1=[jrec[j] for j in pc['path1']], chain2=[jrec[j] for j in pc['path2']],
                                A=A, B=B, chart_columns=[local[c] for c in pc['chart_columns']],
                                columns={j: local[ids.index(j)] for j in pc['path1'] + pc['path2']
                                         if jrec[j]['type'] != 'fixed'}))
        self.evaluator = CutSetEvaluator(records, self.n)
        self.cut_indices = cuts
        self.nc = len(cuts)
        self.calls = 0


class GeneratedSupportGraph(_GraphBase):
    """Chart-base graph: [base6, physical, cut charts] with loop cuts plus sole welds to world anchors."""

    def __init__(self, comp, sites, anchors):
        plan = comp.support_plan(sites)
        self.plan = plan
        chart = comp.chart_cmg()
        jrec = {j['id']: j for j in chart['joints']}
        ids = chart['coordinate_ids']
        n = plan['coordinates']
        self.n = n
        self.columns = np.arange(n)
        self.active = np.asarray(plan['active'], int)
        self.passive = np.asarray(plan['passive'], int)
        self.lower = np.full(n, -np.pi)
        self.upper = np.full(n, np.pi)
        for i, k in enumerate(ids):
            self.lower[i] = jrec[k]['limits']['lower']
            self.upper[i] = jrec[k]['limits']['upper']
        records = []
        for k, pc in enumerate(comp.plan['cut_plans']):
            cut = comp.cmg['closures'][k]
            A, B = _cut_frames(cut, pc)
            records.append(dict(chain1=[jrec[j] for j in pc['path1']], chain2=[jrec[j] for j in pc['path2']],
                                A=A, B=B, chart_columns=[6 + c for c in pc['chart_columns']],
                                columns={j: ids.index(j) for j in pc['path1'] + pc['path2']
                                         if jrec[j]['type'] != 'fixed'}))
        base_path = [jrec[b] for b in ('base_x', 'base_y', 'base_z', 'base_yaw', 'base_pitch', 'base_roll')]
        self.welds = []
        self.site_B = []
        self.anchors = [np.asarray(a, float) for a in anchors]
        for name, anchor in zip(plan['sites'], self.anchors):
            site = next(s for s in comp.cmg['sole_sites'] if s['id'] == name)
            path = base_path + [jrec[j] for j in comp.plan['root_paths'][site['body']]['joint_path']]
            F = np.eye(4)
            F[:3, :3] = site['frame_rotation']
            F[:3, 3] = site['frame_translation_m']
            records.append(dict(chain1=[], chain2=path, A=anchor, B=F, chart_columns=[],
                                columns={j['id']: ids.index(j['id']) for j in path if j['type'] != 'fixed'}))
            self.welds.append(name)
            self.site_B.append(F)
        self.weld_records = records[len(comp.plan['cut_plans']):]
        self.evaluator = CutSetEvaluator(records, n)
        self.nc = len(records)
        self.calls = 0

    def site_frames(self, x):
        """Current world poses of the welded sole frames (for anchors and checks)."""
        probe = CutSetEvaluator([dict(r, A=np.eye(4)) for r in self.weld_records], self.n)
        xz = np.append(np.asarray(x, float), 0.)
        T, _ = probe._chains(xz, False)
        return [T[probe.nc + k] @ self.site_B[k] for k in range(len(self.welds))]


class SourceLayoutGraph:
    """The generated global evaluator presented in the original v22 CutGraph layout.

    Columns follow the original coordinate order and chart columns, rows follow
    the original cut order, so the unchanged original rollout code
    (``kangaroo_pin.pin_simulation``) can use it in place of the accepted
    ``CutGraph``.  Like the original ``make_graph`` wrapper it keeps an exact
    single-entry cache for a byte-identical configuration without defects.
    """

    def __init__(self, comp, accepted, cmg0):
        layout = accepted.CutGraph(cmg0, np.asarray(cmg0['initial_seed']))
        compiled = accepted.CutGraph(comp.cmg, np.asarray(comp.cmg['initial_seed']))
        self.inner = GeneratedLoopGraph(comp)
        self.n, self.nt, self.nc = layout.n, layout.nt, layout.nc
        self.active, self.passive = layout.active.copy(), layout.passive.copy()
        self.lower, self.upper = layout.lower.copy(), layout.upper.copy()
        self.ids, self.cuts, self.chart_columns = layout.ids, layout.cuts, layout.chart_columns
        src = np.empty(self.n, int)  # compiled column i <- original column src[i]
        src[:self.nt] = [layout.ids.index(k) for k in comp.cmg['coordinate_ids']]
        supplied_cut = {c['id']: k for k, c in enumerate(cmg0['closures'])}
        for c, cols in zip(comp.cmg['closures'], compiled.chart_columns):
            src[cols] = layout.chart_columns[supplied_cut[c['id']]]
        self.src = src
        self.column_of = np.argsort(src)  # original column j -> compiled column
        compiled_cut = {c['id']: k for k, c in enumerate(comp.cmg['closures'])}
        self.cut_order = np.array([compiled_cut[c['id']] for c in cmg0['closures']], int)  # original k -> compiled
        self.cut_inverse = np.argsort(self.cut_order)
        if not np.array_equal(np.sort(self.active), np.sort(src[compiled.active])):
            raise ValueError('Motor columns differ between the original and compiled layouts')
        self._key = self._value = None
        self.evaluations = self.cache_hits = 0

    def residual(self, q, defects=None):
        q = np.asarray(q, dtype=float)
        if defects is None:
            key = q.tobytes()
            if key == self._key:
                self.cache_hits += 1
                return self._value
        self.evaluations += 1
        D = None if defects is None else np.asarray(defects)[self.cut_inverse]
        r, J, delta = self.inner.residual(q[self.src], D)
        r = r.reshape(self.nc, 6)[self.cut_order].ravel()
        J = J.reshape(self.nc, 6, self.n)[self.cut_order].reshape(6 * self.nc, self.n)[:, self.column_of]
        value = (r, J, delta[self.cut_order])
        if defects is None:
            for item in value:
                item.setflags(write=False)
            self._key, self._value = key, value
        return value

    def augment(self, q):
        return np.r_[q, np.zeros(self.n - self.nt)]


def sole_frames(comp, x_chart):
    """World poses of every declared sole frame at a chart-base configuration."""
    probe = GeneratedSupportGraph(comp, [s['id'] for s in comp.cmg['sole_sites']],
                                  [np.eye(4)] * len(comp.cmg['sole_sites']))
    return dict(zip(probe.welds, probe.site_frames(x_chart)))


# ---------------------------------------------------------------------------
# Solvers (continuation / local updates)
# ---------------------------------------------------------------------------
class TRFOracle:
    """Count EVERY fresh evaluation, including numerical perturbations.

    An analytical (residual, Jacobian) pair is computed once at an identical x;
    fun and jac share it. The finite-difference fun never computes an
    analytical Jacobian.
    """

    def __init__(self, graph, active, analytic=True, sparse=False, budget=1500):
        self.g = graph
        self.active = np.asarray(active, float).copy()
        self.analytic, self.sparse, self.budget = analytic, sparse, budget
        self.last = self.r = self.J = None
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
            self.r, j, _ = self.g.residual(x)
            self.J = j[:, self.g.passive]
        else:
            self.r = self.g.residual_only(x)
            self.J = None

    def fun(self, p):
        self._evaluate(p)
        return self.r.copy()

    def jac(self, p):
        self._evaluate(p)
        return csr_matrix(self.J) if self.sparse else self.J.copy()


TRF_OPTIONS = dict(method='trf', ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=1500, x_scale=1.)


def _trf(graph, active, pred, sparse=False):
    oracle = TRFOracle(graph, active, analytic=True, sparse=sparse)
    res = least_squares(oracle.fun, pred, jac=oracle.jac, tr_solver='lsmr' if sparse else 'exact',
                        bounds=(graph.lower[graph.passive], graph.upper[graph.passive]), **TRF_OPTIONS)
    return oracle.full(res.x), dict(success=bool(res.success)), oracle.evaluations


def _clip(g, pred):
    return np.clip(pred, g.lower[g.passive] + 1e-10, g.upper[g.passive] - 1e-10)


def counted(graph):
    """Attach a call counter to an accepted graph instance (the class itself is not modified)."""
    method = graph.residual
    graph.calls = 0

    def residual(q, defects=None):
        graph.calls += 1
        return method(q, defects)

    graph.residual = residual
    return graph


class GlobalSolver:
    """Monolithic assembly over all 24 cuts; original, generated or TRF route.

    ``reuse=True`` returns the stored accepted state and map when the entire motor input is
    bit-identical to the previous accepted input (whole-input reuse, no module structure).
    """

    def __init__(self, comp, x0, kind='compiled', predictor=True, accepted=None, reuse=False):
        if kind == 'original':
            self.g = counted(accepted.CutGraph(comp.cmg, np.asarray(comp.cmg['initial_seed'])))
        else:
            self.g = GeneratedLoopGraph(comp)
        self.solver = PACDM(self.g)
        self.kind, self.predictor, self.comp, self.reuse = kind, predictor, comp, reuse
        self.x = np.asarray(x0, float).copy()
        self.N, self.info = self.solver.mapping(self.x)
        if not self.info['success']:
            raise RuntimeError(self.info)
        self.nt = comp.plan['physical_coordinates']

    def step(self, active):
        g = self.g
        active = np.asarray(active, float)
        if self.reuse and np.array_equal(active, self.x[g.active]):
            return self.x[:self.nt].copy(), self.N[:self.nt].copy(), dict(
                fallback=False, solved_modules=0, skipped_modules=1, rcond=self.info['rcond'],
                tangent_residual=self.info['tangent_residual'], evaluations=0)
        pred = self.x[g.passive].copy()
        if self.predictor:
            pred += self.N[g.passive] @ (active - self.x[g.active])
        fallback = False
        evaluations = 0
        if self.kind == 'trf':
            x, ci, evaluations = _trf(g, active, _clip(g, pred))
        else:
            x, ci = self.solver.correct(active, pred, None, np.array(self.info['rows']), maxiter=8)
            if not ci['success']:
                x, ci = self.solver.acquire(active, self.x)
                fallback = True
        N, info = self.solver.mapping(x)
        if not ci['success'] or not info['success']:
            raise RuntimeError(str((ci, info)))
        self.x, self.N, self.info = x, N, info
        return x[:self.nt].copy(), N[:self.nt].copy(), dict(
            fallback=fallback, solved_modules=1, skipped_modules=0, rcond=info['rcond'],
            tangent_residual=info['tangent_residual'], evaluations=evaluations)

    def augmented(self):
        return self.x.copy()


class ModularSolver:
    """Independent loop modules generated from shared dependent coordinates, exact reuse."""

    def __init__(self, comp, x0, predictor=True, reuse=True, corrector='pacdm'):
        self.comp, self.predictor, self.reuse, self.corrector = comp, predictor, reuse, corrector
        plan = comp.plan
        self.nt = plan['physical_coordinates']
        self.motor_pos = {m: k for k, m in enumerate(plan['motor_indices'])}
        self.modules = []
        x0 = np.asarray(x0, float)
        for mod in plan['modules']:
            g = GeneratedLoopGraph(comp, mod['cuts'], mod)
            x = x0[g.columns].copy()
            solver = PACDM(g)
            N, info = solver.mapping(x)
            if not info['success']:
                raise RuntimeError(info)
            phys = np.array([i for i in g.passive if g.columns[i] < self.nt], int)
            self.modules.append(dict(g=g, solver=solver, x=x, N=N, info=info, phys=phys,
                                     phys_cols=g.columns[phys],
                                     inputs=np.array([self.motor_pos[m] for m in mod['motors']], int)))

    def step(self, active):
        active = np.asarray(active, float)
        nt = self.nt
        na = len(self.motor_pos)
        q = np.zeros(nt)
        N = np.zeros((nt, na))
        for m, k in self.motor_pos.items():
            q[m] = active[k]
            N[m, k] = 1.
        solved = skipped = evaluations = 0
        fallback = False
        rc, tr = 1., 0.
        for m in self.modules:
            g = m['g']
            qa = active[m['inputs']]
            if self.reuse and np.array_equal(qa, m['x'][g.active]):
                skipped += 1
            else:
                solved += 1
                pred = m['x'][g.passive].copy()
                if self.predictor:
                    pred += m['N'][g.passive] @ (qa - m['x'][g.active])
                if self.corrector == 'trf':
                    x, ci, ev = _trf(g, qa, _clip(g, pred))
                    evaluations += ev
                else:
                    x, ci = m['solver'].correct(qa, pred, None, np.array(m['info']['rows']), maxiter=8)
                    if not ci['success']:
                        x, ci = m['solver'].acquire(qa, m['x'])
                        fallback = True
                Nm, info = m['solver'].mapping(x)
                if not ci['success'] or not info['success']:
                    raise RuntimeError(str((ci, info)))
                m.update(x=x, N=Nm, info=info)
            q[m['phys_cols']] = m['x'][m['phys']]
            N[np.ix_(m['phys_cols'], m['inputs'])] = m['N'][m['phys']]
            rc = min(rc, m['info']['rcond'])
            tr = max(tr, m['info']['tangent_residual'])
        return q, N, dict(fallback=fallback, solved_modules=solved, skipped_modules=skipped, rcond=rc,
                          tangent_residual=tr, evaluations=evaluations)

    def augmented(self):
        """Current full augmented state assembled from the module states."""
        x = np.zeros(self.comp.plan['augmented_coordinates'])
        for m in self.modules:
            x[m['g'].columns] = m['x']
        return x


METHODS = ['original_monolithic', 'compiled_monolithic', 'compiled_monolithic_reuse', 'compiled_modular',
           'modular_no_predictor', 'modular_no_reuse', 'trf_compiled_predictor', 'trf_modular_predictor']


def create_solver(name, comp, x0, accepted):
    if name == 'original_monolithic':
        return GlobalSolver(comp, x0, 'original', accepted=accepted)
    if name == 'compiled_monolithic':
        return GlobalSolver(comp, x0, 'compiled')
    if name == 'compiled_monolithic_reuse':
        return GlobalSolver(comp, x0, 'compiled', reuse=True)
    if name == 'compiled_modular':
        return ModularSolver(comp, x0)
    if name == 'modular_no_predictor':
        return ModularSolver(comp, x0, predictor=False)
    if name == 'modular_no_reuse':
        return ModularSolver(comp, x0, reuse=False)
    if name == 'trf_compiled_predictor':
        return GlobalSolver(comp, x0, 'trf')
    if name == 'trf_modular_predictor':
        return ModularSolver(comp, x0, corrector='trf')
    raise ValueError(name)
