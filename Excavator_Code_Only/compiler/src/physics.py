"""Constrained rigid-body dynamics: generated PACDM reduction and two references.

* PACDM route: the compiled graph's map N and velocity-product term c
  (``q_dd = N u_dd + c``) with M and h from the NumPy ``SourceBiasDynamics``
  (original v21 code, no Pinocchio) evaluated on the *compiled* spanning tree.
* KKT reference: the same NumPy code on the *original* spanning tree with its
  own two-point closure Jacobian and analytic J_dot q_dot, solved in full
  coordinates with a rank-revealing row selection (54 rows, rank 16).
* Native reference: Pinocchio 3.8.0 ``constraintDynamics`` on the validated
  backend model (CONTACT_3D point pairs, proximal settings of that backend),
  with its own crba/nonLinearEffects/frame-acceleration certificates.

Rigid-body inertias only: no armature, friction, damping, hydraulics or soil;
all routes use the same physical parameters and the same generalized efforts.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr

from . import bootstrap  # noqa: F401
from pacdm import PACDM
from source_bias import SourceBiasDynamics  # original NumPy-only dynamics
from plant import ExcavatorPinModel          # validated Pinocchio backend
from .loop_graph import GlobalLoopGraph

GRAVITY = np.array([0., 0., -9.81])


def normmax(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


def relative(x, y):
    return normmax(np.asarray(x) - np.asarray(y)) / max(1., normmax(y))


def mapping_curvature(graph, q, active_velocity):
    """PACDM map N and c from a centered directional difference of the Jacobian."""
    N, info = PACDM(graph).mapping(q)
    if N is None:
        raise RuntimeError(info)
    v = N @ np.asarray(active_velocity, float)
    eps = 1e-5 / max(1., normmax(v))
    J = graph.residual(q)[1]
    Jp = graph.residual(q + eps * v)[1]
    Jm = graph.residual(q - eps * v)[1]
    gamma = ((Jp - Jm) / (2 * eps)) @ v
    c = np.zeros(graph.n)
    rows = np.array(info['rows'], int)
    c[graph.passive] = -np.linalg.solve(J[np.ix_(rows, graph.passive)], gamma[rows])
    return N, v, c, info


class Dynamics:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ref_ids = list(ctx.ref_ids)
        self.cut_ids = list(ctx.mapping['cut_joint_ids'])
        self.native = ExcavatorPinModel(ctx.cmg, self.cut_ids)
        self.numpy_ref = SourceBiasDynamics(ctx.cmg, self.ref_ids)
        self.compiled = {}
        for p, comp in ctx.compiled.items():
            tree_ids = list(comp.cmg['tree_joint_ids'])
            self.compiled[p] = dict(comp=comp, graph=GlobalLoopGraph(comp), tree_ids=tree_ids,
                                    numpy=SourceBiasDynamics(comp.cmg, tree_ids),
                                    tree_rows=np.array([comp.structure['coordinate_ids'].index(k) for k in tree_ids]),
                                    ref_rows=np.array([comp.structure['coordinate_ids'].index(k) for k in self.ref_ids]))
        self.actuated = [a['joint_id'] for a in ctx.physical['actuators']]
        import pinocchio as pin
        self.pin = pin
        self.prox = pin.ProximalSettings(1e-12, 1e-9, 8)   # validated backend settings
        self.bucket_frame = self.native.body_frame['body_56']

    def efforts(self, tree_ids, motors, wrench, q_tree):
        """Generalized efforts: actuator efforts on their joints + a body_56 wrench."""
        tau = np.zeros(len(tree_ids))
        for jid, f in zip(self.actuated, motors):
            tau[tree_ids.index(jid)] += f
        return tau

    def witness(self, partition, q32, active_velocity, motors, wrench, *, joint_velocity=None):
        """One state: PACDM (compiled tree) vs KKT (original tree) vs native Pinocchio."""
        d = self.compiled[partition]
        comp, g = d['comp'], d['graph']
        cids = comp.structure['coordinate_ids']
        q = np.array([q32[k] for k in cids])
        if joint_velocity is not None:
            active_velocity = np.asarray(joint_velocity)[comp.structure['independent']]
        N, v, c, info = mapping_curvature(g, q, active_velocity)
        # ---- PACDM reduction with NumPy M, h on the compiled tree
        tr = d['tree_rows']
        qt, vt = q[tr], v[tr]
        ev = d['numpy'].evaluate(qt, vt, GRAVITY)
        M, h = ev['mass_matrix'], ev['bias_forces']
        tau_c = self._tau(d['tree_ids'], motors, wrench, ev, qt)
        Nt, ct = N[tr], c[tr]
        Mr = Nt.T @ M @ Nt
        udd = np.linalg.solve(Mr, Nt.T @ (tau_c - h - M @ ct))
        a_all = N @ udd + c
        rr = d['ref_rows']
        a_pacdm = a_all[rr]
        q_ref, v_ref = q[rr], v[rr]
        # ---- KKT reference: NumPy on the original tree
        ev0 = self.numpy_ref.evaluate(q_ref, v_ref, GRAVITY)
        M0, h0 = ev0['mass_matrix'], ev0['bias_forces']
        tau0 = self._tau(self.ref_ids, motors, wrench, ev0, q_ref)
        res0, J0, Jd0 = self.numpy_ref.closure(ev0, self.cut_ids)
        gamma0 = Jd0 @ v_ref
        piv = qr(J0.T, pivoting=True, mode='economic')[2]
        s = np.linalg.svd(J0, compute_uv=False)
        r0 = int(np.sum(s > 1e-10 * s[0]))
        rows = np.sort(piv[:r0])
        K = np.block([[M0, -J0[rows].T], [J0[rows], np.zeros((r0, r0))]])
        sol = np.linalg.solve(K, np.r_[tau0 - h0, -gamma0[rows]])
        a_kkt = sol[:len(self.ref_ids)]
        # ---- native Pinocchio
        pin, nm = self.pin, self.native
        tau_n = self._tau_native(motors, wrench, q_ref)
        a_nat = np.array(pin.constraintDynamics(nm.model, nm.data, q_ref, v_ref, tau_n, nm.constraints,
                                                nm.constraint_datas, self.prox))
        Mn, hn = nm.mass_bias(q_ref, v_ref)
        Jn = nm.closure_jacobian(q_ref)
        gn = nm.gamma(q_ref, v_ref)
        Nref = N[rr]
        scale = max(1., normmax(tau_n), normmax(hn))
        # ---- probe: the same reduction formula with the NATIVE least-squares map and velocity term
        a_idx, p_idx = self.ctx.reference.active[partition], self.ctx.reference.passive[partition]
        N_nat = np.zeros_like(Nref)
        N_nat[a_idx, np.arange(len(a_idx))] = 1.
        N_nat[p_idx] = np.linalg.lstsq(Jn[:, p_idx], -Jn[:, a_idx], rcond=None)[0]
        c_nat = np.zeros(len(q_ref))
        c_nat[p_idx] = np.linalg.lstsq(Jn[:, p_idx], -gn, rcond=None)[0]
        udd_nat = np.linalg.solve(N_nat.T @ Mn @ N_nat, N_nat.T @ (tau_n - hn - Mn @ c_nat))
        a_native_map = N_nat @ udd_nat + c_nat
        # Force-equivalent difference |M (a - a_ref)| / max(1, |tau|, |h|), as in the validated
        # backend. The normalized acceleration difference is also recorded: it is dominated by
        # the light passive pivot mode, along which a map difference of order 1e-12 produces a
        # large acceleration of a nearly massless body; the native-map probe below separates the
        # map difference from the reduction formula.
        force = lambda x, y: normmax(Mn @ (np.asarray(x) - np.asarray(y))) / scale  # noqa: E731
        data = dict(
            pacdm_vs_kkt_force=force(a_pacdm, a_kkt),
            pacdm_vs_native_force=force(a_pacdm, a_nat),
            kkt_vs_native_force=force(a_kkt, a_nat),
            reduced_inertia_condition=float(np.linalg.cond(Mr)),
            map_pacdm_vs_native=normmax(Nref - N_nat),
            native_map_reduction_vs_native_relative=relative(a_native_map, a_nat),
            native_map_reduction_vs_native_force=force(a_native_map, a_nat),
            pacdm_vs_kkt_relative=relative(a_pacdm, a_kkt),
            pacdm_vs_native_relative=relative(a_pacdm, a_nat),
            kkt_vs_native_relative=relative(a_kkt, a_nat),
            pacdm_constraint_acceleration_m_s2=normmax(Jn @ a_pacdm + gn),
            pacdm_tangent_dynamics_relative=normmax(Nref.T @ (Mn @ a_pacdm + hn - tau_n)) / scale,
            native_constraint_acceleration_m_s2=normmax(Jn @ a_nat + gn),
            native_tangent_dynamics_relative=normmax(Nref.T @ (Mn @ a_nat + hn - tau_n)) / scale,
            numpy_vs_native_mass=normmax(M0 - Mn), numpy_vs_native_bias=normmax(h0 - hn),
            numpy_vs_native_jacobian=normmax(J0 - Jn), numpy_vs_native_gamma=normmax(gamma0 - gn),
            numpy_vs_native_effort=normmax(tau0 - tau_n),
            tangent_residual_native=normmax(Jn @ Nref), closure_gap_native_m=normmax(nm.closure_residual(q_ref)),
            min_reduced_inertia_eigenvalue=float(np.linalg.eigvalsh(Mr).min()),
            velocity_product_term_max=normmax(c), kkt_rank=r0,
            identity_virtual_power_W=abs(float(tau_c @ vt - (Nt.T @ tau_c) @ np.asarray(active_velocity))),
            identity_kkt_solve_residual=normmax(K @ sol - np.r_[tau0 - h0, -gamma0[rows]]),
            rcond=info['rcond'])
        return data, dict(a_ref=a_pacdm, a_kkt=a_kkt, a_native=a_nat, v_ref=v_ref, q_ref=q_ref, v_all=v, a_all=a_all,
                          M_native=Mn, scale=scale)

    def _tau(self, tree_ids, motors, wrench, ev, q_tree):
        tau = self.efforts(tree_ids, motors, wrench, q_tree)
        J = ev['jacobians']['body_56']          # linear-first, world-aligned at the body origin
        return tau + J.T @ np.asarray(wrench)

    def _tau_native(self, motors, wrench, q_ref):
        pin, nm = self.pin, self.native
        tau = self.efforts(self.ref_ids, motors, wrench, q_ref)
        pin.computeJointJacobians(nm.model, nm.data, q_ref)
        pin.updateFramePlacements(nm.model, nm.data)
        J = np.array(pin.getFrameJacobian(nm.model, nm.data, self.bucket_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        return tau + J.T @ np.asarray(wrench)
