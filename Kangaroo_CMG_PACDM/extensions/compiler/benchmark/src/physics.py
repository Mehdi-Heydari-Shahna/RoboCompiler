"""Independent NumPy closure/support geometry, KKT dynamics and the PACDM-reduced route.

New extension code. The NumPy tree (``numpy_reference_base.NumpyTree``, taken
unchanged from the Go2 benefit study) computes body poses, world-frame
Jacobians and classical bias accelerations by an explicit recursion; it calls
neither PACDM, the generated evaluator nor Pinocchio.

Constraint rows used by the independent routes (physical records only):

* point cut: ``p1 - p2 = 0`` (three rows);
* universal cut: the three point rows plus ``(R1 f1_x) . (R2 f2_y) = 0``
  (one angular row), exactly the v22 ``UniversalMechanism`` definition;
* ideal sole weld: angular velocity of the foot body and linear velocity of
  the sole-frame origin (six rows), classical bias accelerations.

The 80 loop rows carry 16 redundant physical rows (rank 64). The KKT route
selects a maximal independent row subset by column-pivoted QR of J^T and
solves the resulting nonsingular saddle point; all rows are checked afterwards.

Dynamics terms are rigid-body inertia plus the supplied diagonal armature (no
damping, friction, contact compliance or unilateral inequalities) on every
compared route.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.linalg import qr

from . import bootstrap  # noqa: F401
from .numpy_reference_base import NumpyTree, cross_matrix
from .evaluator import GeneratedSupportGraph
from pacdm import PACDM, rank


def normmax(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


def relative(x, y):
    return normmax(np.asarray(x) - np.asarray(y)) / max(1., normmax(y))


class NumpyReference(NumpyTree):
    """NumPy tree over a compiled or chart-base CMG (independent of PACDM and Pinocchio)."""

    def __init__(self, cmg):
        c = deepcopy(cmg)
        for b in c['bodies']:
            b['inertia_kg_m2'] = b['inertia_com_kg_m2']
        c.setdefault('gravity_m_s2', [0., 0., -9.81])
        super().__init__(c)
        self.armature = np.array([float(cmg['armature'].get(k, 0.)) for k in self.ids])

    @staticmethod
    def _point(states, body, local):
        s = states[body]
        r = s['R'] @ np.asarray(local, float)
        acc = s['a'] + np.cross(s['alpha'], r) + np.cross(s['w'], np.cross(s['w'], r))
        return s['p'] + r, s['Jv'] - cross_matrix(r) @ s['Jw'], acc

    def loop_geometry(self, q, v=None):
        """Point gaps (3 per cut), universal residuals, the 80 x n rows and their bias."""
        states = self.forward(q, v, acceleration=v is not None)
        vv = np.zeros(self.n) if v is None else np.asarray(v, float)
        gaps, univ, rows, bias = [], [], [], []
        for c in self.cmg['closures']:
            p1, J1, a1 = self._point(states, c['body1'], c['point1_m'])
            p2, J2, a2 = self._point(states, c['body2'], c['point2_m'])
            gaps.append(p1 - p2)
            rows.append(J1 - J2)
            bias.append(a1 - a2)
        for c in self.cmg['closures']:
            if c['type'] != 'universal':
                continue
            s1, s2 = states[c['body1']], states[c['body2']]
            a = s1['R'] @ np.asarray(c['frame1_R'])[:, 0]
            b = s2['R'] @ np.asarray(c['frame2_R'])[:, 1]
            n = np.cross(a, b)
            w1, w2 = s1['Jw'] @ vv, s2['Jw'] @ vv
            univ.append(float(a @ b))
            rows.append((n @ (s1['Jw'] - s2['Jw']))[None, :])
            dn = np.cross(np.cross(w1, a), b) + np.cross(a, np.cross(w2, b))
            bias.append(np.array([n @ (s1['alpha'] - s2['alpha']) + dn @ (w1 - w2)]))
        return dict(point_gap=np.concatenate(gaps), universal_residual=np.asarray(univ),
                    jacobian=np.vstack(rows), bias=np.concatenate(bias), states=states)

    def weld_geometry(self, q, v, frames):
        """Six rows per welded sole frame: [foot angular; sole-origin linear] (world axes)."""
        states = self.forward(q, v, acceleration=v is not None)
        rows, bias, poses = [], [], []
        for body, F in frames:
            s = states[body]
            p, Jp, ap = self._point(states, body, np.asarray(F)[:3, 3])
            rows.append(np.vstack([s['Jw'], Jp]))
            bias.append(np.r_[s['alpha'], ap])
            T = np.eye(4)
            T[:3, :3] = s['R'] @ np.asarray(F)[:3, :3]
            T[:3, 3] = p
            poses.append(T)
        return dict(jacobian=np.vstack(rows) if rows else np.zeros((0, self.n)),
                    bias=np.concatenate(bias) if bias else np.zeros(0), poses=poses)

    def dynamics_terms(self, q, v):
        M, b, _ = self.mass_bias(q, v)
        return M + np.diag(self.armature), b

    def body_jacobian(self, q, body):
        s = self.forward(q)[body]
        return np.vstack([s['Jv'], s['Jw']])


def row_basis(J, expected=None):
    """Column-pivoted QR row selection of a (possibly redundant) constraint matrix."""
    r = rank(J)
    if expected is not None and r != expected:
        raise RuntimeError(f'Constraint rank {r} differs from the expected {expected}')
    return np.sort(qr(J.T, pivoting=True, mode='economic')[2][:r]), r


def mapping_curvature(graph, x, active_velocity):
    """PACDM tangent map and curvature c (J c = -Jdot v) by a centred directional difference."""
    N, info = PACDM(graph).mapping(x)
    if not info['success']:
        raise RuntimeError(info)
    v = N @ np.asarray(active_velocity, float)
    eps = 1e-5 / max(1., normmax(v))
    J = graph.residual(x)[1]
    Jp = graph.residual(x + eps * v)[1]
    Jm = graph.residual(x - eps * v)[1]
    gamma = ((Jp - Jm) / (2 * eps)) @ v
    c = np.zeros(graph.n)
    rows = np.array(info['rows'], int)
    c[graph.passive] = -np.linalg.solve(J[np.ix_(rows, graph.passive)], gamma[rows])
    return N, v, c, info


def sole_frame_records(comp, sites):
    out = []
    for name in sites:
        site = next(s for s in comp.cmg['sole_sites'] if s['id'] == name)
        F = np.eye(4)
        F[:3, :3] = site['frame_rotation']
        F[:3, 3] = site['frame_translation_m']
        out.append((site['body'], F))
    return out


def motor_effort(comp, ref, q_chart, motors, wrench):
    """Generalized chart-base effort of 12 motor forces plus a world wrench at the pelvis origin."""
    chart = comp.chart_cmg()
    tau = np.zeros(len(chart['coordinate_ids']))
    for a, u in zip(sorted(comp.cmg['actuators'], key=lambda a: a['id']), motors):
        tau[chart['coordinate_ids'].index(a['joint'])] += float(a['gear']) * u
    motor_tau = tau.copy()
    tau += ref.body_jacobian(q_chart, comp.cmg['root_body']).T @ np.asarray(wrench, float)
    return tau, motor_tau


def dynamics_witness(comp, x, sites, va, motors, wrench, native=False, chart_ref=None):
    """One instantaneous ideal-support witness in chart-base coordinates.

    ``x`` is the 146-vector [base6, physical 76, cut charts 64]; anchors are the
    current sole frames (ideal welds placed where the soles are).
    """
    ref = chart_ref or NumpyReference(comp.chart_cmg())
    nq = 6 + comp.plan['physical_coordinates']
    q = np.asarray(x, float)[:nq]
    frames = sole_frame_records(comp, sites)
    anchors = ref.weld_geometry(q, None, frames)['poses']
    g = GeneratedSupportGraph(comp, sites, anchors)
    N, v, c, info = mapping_curvature(g, x, va)
    Np, vp, cp = N[:nq], v[:nq], c[:nq]
    M, b = ref.dynamics_terms(q, vp)
    tau, motor_tau = motor_effort(comp, ref, q, motors, wrench)
    Mr = Np.T @ M @ Np
    aa = np.linalg.solve(Mr, Np.T @ (tau - b - M @ cp))
    a = Np @ aa + cp
    loop = ref.loop_geometry(q, vp)
    weld = ref.weld_geometry(q, vp, frames)
    J = np.vstack([loop['jacobian'], weld['jacobian']])
    gamma = np.r_[loop['bias'], weld['bias']]
    expected_rank = comp.plan['physical_closure_rank_expected'] + 6 * len(sites)
    rows, r = row_basis(J, expected_rank)
    Js, gs = J[rows], gamma[rows]
    K = np.block([[M, -Js.T], [Js, np.zeros((len(rows), len(rows)))]])
    sol = np.linalg.solve(K, np.r_[tau - b, -gs])
    ak, lam = sol[:nq], sol[nq:]
    # Independent tangent map from the NumPy rows with the generated active set.
    act = np.array([i for i in g.active if i < nq], int)
    dep = np.setdiff1d(np.arange(nq), act)
    Ni = np.zeros((nq, len(act)))
    Ni[act] = np.eye(len(act))
    Ni[dep] = -np.linalg.lstsq(J[:, dep], J[:, act], rcond=None)[0]
    closure = max(normmax(loop['point_gap']), normmax(loop['universal_residual']))
    data = dict(
        relative_acceleration_difference=relative(a, ak), absolute_acceleration_difference=normmax(a - ak),
        constraint_acceleration_residual=normmax(J @ a + gamma), tangent_residual=normmax(J @ Np),
        mapping_discrepancy=normmax(Np - Ni), virtual_power_defect_W=abs(float(tau @ vp - (Np.T @ tau) @ va)),
        reduced_inertia_relative=relative(Mr, Ni.T @ M @ Ni),
        min_reduced_inertia_eigenvalue=float(np.linalg.eigvalsh(0.5 * (Mr + Mr.T)).min()),
        kkt_force_balance=normmax(M @ ak + b - tau - Js.T @ lam), kinematic_curvature_constraint=normmax(J @ cp + gamma),
        base_motor_effort_max=normmax(motor_tau[:6]), rank=int(r), expected_rank=expected_rank,
        mobility=len(g.active), expected_mobility=len(g.active), closure_gap=closure, rcond=info['rcond'],
        native_acceleration_relative=None, native_mass_inf=None, native_bias_inf=None,
        native_constraint_jacobian_inf=None, native_constraint_bias_inf=None, native_iterations=None)
    native_version = None
    if native:
        from .native import NativeSupportOracle
        oracle = NativeSupportOracle(comp, q, sites)
        an, stats = oracle.acceleration(q, vp, tau)
        geo = oracle.geometry(q, vp)
        data.update(native_acceleration_relative=relative(a, an),
                    native_mass_inf=normmax(oracle.backend.mass(q) + np.diag(ref.armature) - M),
                    native_bias_inf=normmax(oracle.backend.bias(q, vp) - b),
                    native_constraint_jacobian_inf=normmax(geo['jacobian'] - J),
                    native_constraint_bias_inf=normmax(geo['bias'] - gamma),
                    native_iterations=stats.get('point_iterations'))
        native_version = oracle.version
    limits = dict(relative_acceleration_difference=1e-8, constraint_acceleration_residual=2e-6,
                  mapping_discrepancy=1e-9, virtual_power_defect_W=1e-9, reduced_inertia_relative=1e-9,
                  tangent_residual=1e-9, closure_gap=1e-8)
    good = all(data[k] <= v for k, v in limits.items()) and data['min_reduced_inertia_eigenvalue'] > 0 \
        and data['base_motor_effort_max'] == 0 and data['rank'] == expected_rank
    if native:
        native_limits = dict(native_acceleration_relative=1e-7, native_mass_inf=1e-8, native_bias_inf=1e-7,
                             native_constraint_jacobian_inf=1e-9, native_constraint_bias_inf=1e-8)
        good = good and all(data[k] <= v for k, v in native_limits.items())
    data['success'] = bool(good)
    return data, dict(x=np.asarray(x, float), sites=list(sites), active_velocity=np.asarray(va, float),
                      motors=np.asarray(motors, float), wrench=np.asarray(wrench, float), v=vp, acceleration=a,
                      acceleration_kkt=ak, independent_ids=[comp.chart_cmg()['coordinate_ids'][i] for i in act],
                      native_version=native_version)


def curvature_ablation(comp, x, sites, va, aactive, ref):
    """Full vs curvature-omitted mapping residual J (N a + c) + gamma with independent NumPy J, gamma."""
    nq = 6 + comp.plan['physical_coordinates']
    q = np.asarray(x, float)[:nq]
    frames = sole_frame_records(comp, sites)
    anchors = ref.weld_geometry(q, None, frames)['poses']
    g = GeneratedSupportGraph(comp, sites, anchors)
    N, v, c, info = mapping_curvature(g, x, va)
    loop = ref.loop_geometry(q, v[:nq])
    weld = ref.weld_geometry(q, v[:nq], frames)
    J = np.vstack([loop['jacobian'], weld['jacobian']])
    gamma = np.r_[loop['bias'], weld['bias']]
    full = normmax(J @ (N[:nq] @ aactive + c[:nq]) + gamma)
    omitted = normmax(J @ (N[:nq] @ aactive) + gamma)
    return full, omitted
