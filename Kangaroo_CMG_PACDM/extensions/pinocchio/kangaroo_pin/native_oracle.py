"""Native Pinocchio closed-loop oracle for the Kangaroo cuts (independent of PACDM).

The oracle never uses PACDM rows, charts, logarithms or tangent maps.  It is
assembled directly from the CMG closure records on the same CMG-compiled
Pinocchio tree:

* every cut point (16 point cuts and the 8 universal centres) becomes a
  native ``RigidConstraintModel(CONTACT_3D, ..., LOCAL_WORLD_ALIGNED)``;
* each universal cut adds its scalar angular condition
  ``(R1 f1_x) . (R2 f2_y) = 0`` with a row and acceleration bias formed from
  native frame orientations, Jacobians and frame accelerations.

``acceleration`` solves the constrained forward dynamics with Pinocchio's
native ``constraintDynamics`` for all point rows (proximal settings are
required because the physical point rows are redundant) and enforces the
eight angular rows through an exact Schur complement evaluated by further
native ``constraintDynamics`` calls.  ``acceleration_kkt`` is a second,
dense saddle-point solve of all 80 native rows (SVD pseudo-inverse), used
as a cross-check.  Stabilisation gains are zero; comparisons are made at
feasible configurations and tangent velocities.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED


def _skew(x):
    return np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])


class NativeLoopOracle:
    def __init__(self, backend, cmg, armature, mu_prox=1e-6, max_iterations=400,
                 accuracy=1e-14):
        self.backend = backend
        self.cmg = cmg
        self.model = backend.model
        self.data = self.model.createData()
        self.geometry_data = self.model.createData()
        self.v_indices = backend._v_indices
        self.armature_public = np.asarray(armature, dtype=float)
        native = np.zeros(self.model.nv)
        native[self.v_indices] = self.armature_public
        self.armature_native = native
        self.settings = (accuracy, accuracy, mu_prox, max_iterations)
        self.points = []
        self.universal = []
        self.constraints = []
        for cut in cmg['closures']:
            ends = []
            for side in (1, 2):
                body = cut[f'body{side}']
                T = backend._placements[body].copy()
                T[:3, 3] += T[:3, :3] @ np.asarray(cut[f'point{side}_m'], dtype=float)
                ends.append((backend._supports[body], pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())))
            self.points.append(ends)
            model = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, self.model,
                                             ends[0][0], ends[0][1], ends[1][0], ends[1][1], LWA)
            model.name = cut['id']
            if np.any(model.corrector.Kp) or np.any(model.corrector.Kd):
                raise RuntimeError('Native constraints must have zero stabilisation gains')
            self.constraints.append(model)
            if cut['type'] == 'universal':
                self.universal.append(dict(
                    id=cut['id'], frame1=backend.body_frame_ids[cut['body1']],
                    frame2=backend.body_frame_ids[cut['body2']],
                    axis1=np.asarray(cut['frame1_R'], dtype=float)[:, 0],
                    axis2=np.asarray(cut['frame2_R'], dtype=float)[:, 1]))
            elif cut['type'] != 'point_coincidence':
                raise ValueError(f"Unsupported cut type {cut['type']}")
        self.constraint_data = [c.createData() for c in self.constraints]
        pin.initConstraintDynamics(self.model, self.data, self.constraints)
        self.point_rows = 3 * len(self.constraints)
        self.angular_rows = len(self.universal)

    # ------------------------------------------------------------------ geometry
    def geometry(self, q, v=None):
        """Native residuals, 80 x nv rows and J-dot v bias (point rows first)."""
        b, m, d = self.backend, self.model, self.geometry_data
        v = np.zeros(m.nv) if v is None else np.asarray(v, dtype=float)
        qn, vn = b.native_q(q), b.native_v(v)
        pin.computeJointJacobians(m, d, qn)
        pin.forwardKinematics(m, d, qn, vn, np.zeros(m.nv))
        pin.updateFramePlacements(m, d)
        residual, rows, bias = [], [], []
        for ends in self.points:
            pos, js, acc = [], [], []
            for joint_id, placement in ends:
                pos.append((d.oMi[joint_id] * placement).translation.copy())
                js.append(np.asarray(pin.getFrameJacobian(m, d, joint_id, placement, LWA))[:3, self.v_indices])
                acc.append(np.asarray(pin.getFrameClassicalAcceleration(m, d, joint_id, placement, LWA).linear))
            residual.append(pos[0] - pos[1])
            rows.append(js[0] - js[1])
            bias.append(acc[0] - acc[1])
        angular = []
        for u in self.universal:
            R1 = d.oMf[u['frame1']].rotation
            R2 = d.oMf[u['frame2']].rotation
            a = R1 @ u['axis1']
            c = R2 @ u['axis2']
            Jw1 = np.asarray(pin.getFrameJacobian(m, d, u['frame1'], LWA))[3:, self.v_indices]
            Jw2 = np.asarray(pin.getFrameJacobian(m, d, u['frame2'], LWA))[3:, self.v_indices]
            w1, w2 = Jw1 @ v, Jw2 @ v
            al1 = np.asarray(pin.getFrameAcceleration(m, d, u['frame1'], LWA).angular)
            al2 = np.asarray(pin.getFrameAcceleration(m, d, u['frame2'], LWA).angular)
            n = np.cross(a, c)
            angular.append(float(a @ c))
            rows.append((n @ (Jw1 - Jw2))[None, :])
            dn = np.cross(np.cross(w1, a), c) + np.cross(a, np.cross(w2, c))
            bias.append(np.array([n @ (al1 - al2) + dn @ (w1 - w2)]))
        return dict(point_residual=np.concatenate(residual), angular_residual=np.asarray(angular),
                    jacobian=np.vstack(rows), bias=np.concatenate(bias))

    # ------------------------------------------------------------------ dynamics
    def _native_dynamics(self, qn, vn, tau_native):
        s = pin.ProximalSettings(*self.settings)
        a = pin.constraintDynamics(self.model, self.data, qn, vn, tau_native,
                                   self.constraints, self.constraint_data, s)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError('Native constraintDynamics returned nonfinite acceleration')
        return np.asarray(a).copy(), (s.iter, s.absolute_residual, s.relative_residual)

    def acceleration(self, q, v, tau):
        """Native constrained acceleration for public generalized effort ``tau``."""
        b = self.backend
        self.model.armature[:] = self.armature_native
        try:
            qn, vn = b.native_q(q), b.native_v(v)
            geo = self.geometry(q, v)
            tn = b.native_v(tau, 'tau')
            a0, stats0 = self._native_dynamics(qn, vn, tn)
            if self.angular_rows == 0:
                return b.from_native_v(a0), dict(point_iterations=stats0[0], schur_condition=1.)
            Je = geo['jacobian'][self.point_rows:]
            ge = geo['bias'][self.point_rows:]
            columns = []
            for row in Je:
                aj, _ = self._native_dynamics(qn, vn, tn + b.native_v(row, 'row'))
                columns.append(b.from_native_v(aj - a0))
            K = Je @ np.column_stack(columns)
            a0p = b.from_native_v(a0)
            lam = np.linalg.solve(K, -(ge + Je @ a0p))
            final, stats = self._native_dynamics(qn, vn, tn + b.native_v(Je.T @ lam, 'effort'))
            return b.from_native_v(final), dict(point_iterations=int(stats[0]),
                                                point_absolute_residual=float(stats[1]),
                                                schur_condition=float(np.linalg.cond(K)),
                                                angular_multipliers=lam)
        finally:
            self.model.armature[:] = 0.

    def acceleration_kkt(self, q, v, tau, rcond=1e-11):
        """Dense native saddle point with all 80 rows (SVD pseudo-inverse)."""
        b = self.backend
        M = b.mass(q) + np.diag(self.armature_public)
        h = b.bias(q, v)
        geo = self.geometry(q, v)
        J, gamma = geo['jacobian'], geo['bias']
        Minv_rhs = np.linalg.solve(M, np.asarray(tau) - h)
        Minv_JT = np.linalg.solve(M, J.T)
        delassus = J @ Minv_JT
        lam = -np.linalg.pinv(delassus, rcond=rcond, hermitian=True) @ (J @ Minv_rhs + gamma)
        return Minv_rhs + Minv_JT @ lam, J.T @ lam
