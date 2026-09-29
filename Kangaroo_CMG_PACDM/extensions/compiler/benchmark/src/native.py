"""Native Pinocchio chart-base model and ideal-support oracle (independent of PACDM).

New extension code for the benefit study.

``ChartPinBackend`` compiles the generated chart-base CMG (a world frame, six
scalar base-chart joints x, y, z, yaw, pitch, roll with massless intermediate
frames, then the physical tree) into a native Pinocchio model with one scalar
joint per coordinate, in true depth-first preorder (required by CRBA).  Public
vectors use the chart-CMG coordinate order through explicit index maps.

``NativeSupportOracle`` reuses the unchanged
``kangaroo_pin.native_oracle.NativeLoopOracle`` (native ``CONTACT_3D`` loop
constraints, native universal angular rows enforced by an exact Schur
complement, proximal ``constraintDynamics``) and appends one native
``CONTACT_6D`` weld per supporting sole frame.  Stabilization gains are zero.
It never uses PACDM rows, charts, logarithms or tangent maps.  These are ideal
bilateral welds, not the unilateral frictional contact of the original
rollouts.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pinocchio as pin

from . import bootstrap  # noqa: F401
from kangaroo_pin.native_oracle import NativeLoopOracle

LWA = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED


def _se3(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return pin.SE3(matrix[:3, :3].copy(), matrix[:3, 3].copy())


def _inverse(matrix):
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


class ChartPinBackend:
    """Scalar-joint Pinocchio model of a CMG rooted at a fixed generated ``world`` body."""

    def __init__(self, cmg, gravity=(0., 0., -9.81)):
        self.cmg = deepcopy(cmg)
        self.coordinate_ids = list(cmg['coordinate_ids'])
        bodies = {b['id']: b for b in cmg['bodies']}
        joints = {j['id']: j for j in cmg['joints']}
        if len(bodies) != len(cmg['bodies']) or len(joints) != len(cmg['joints']):
            raise ValueError('Body and joint ids must be unique')
        if {k for k, j in joints.items() if j['type'] != 'fixed'} != set(self.coordinate_ids):
            raise ValueError('coordinate_ids must list every moving joint exactly once')
        if len(joints) != len(bodies) - 1:
            raise ValueError('ChartPinBackend requires a spanning tree; closures are separate records')
        adjacency = {b: [] for b in bodies}
        for jid in sorted(joints):
            j = joints[jid]
            adjacency[j['base_body']].append((jid, j['follower_body'], 1))
            adjacency[j['follower_body']].append((jid, j['base_body'], -1))
        self.model = pin.Model()
        self.model.name = 'kangaroo_chart_base_tree'
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.asarray(gravity, float)
        self.body_frame_ids, self.joint_ids, self._supports, self._placements = {}, {}, {}, {}
        self.root_body = cmg['root_body']

        def add_body(body_id, support, placement, parent_frame):
            frame = pin.Frame(body_id, support, parent_frame, _se3(placement), pin.FrameType.BODY)
            self.body_frame_ids[body_id] = int(self.model.addFrame(frame, False))
            self._supports[body_id] = support
            self._placements[body_id] = placement.copy()
            b = bodies[body_id]
            mass = float(b.get('mass_kg', 0.))
            tensor = np.asarray(b.get('inertia_com_kg_m2', np.zeros((3, 3))), float)
            com = np.asarray(b.get('com_m', [0., 0., 0.]), float)
            if mass == 0.:
                if np.any(tensor != 0.) or b.get('kind') != 'generated_frame':
                    raise ValueError(f'Only generated frames may be massless ({body_id})')
                return
            if mass < 0 or np.min(np.linalg.eigvalsh(tensor)) < -1e-12:
                raise ValueError(f'Invalid inertia for {body_id}')
            self.model.appendBodyToJoint(support, pin.Inertia(mass, com, tensor), _se3(placement))

        add_body(self.root_body, 0, np.eye(4), 0)
        seen, used = {self.root_body}, set()

        def visit(parent):
            for jid, child, direction in adjacency[parent]:
                if jid in used:
                    continue
                if child in seen:
                    raise ValueError(f'Cycle at joint {jid}')
                seen.add(child)
                used.add(jid)
                j = joints[jid]
                base, follower = np.asarray(j['T_BJ'], float), np.asarray(j['T_FJ'], float)
                parent_attach, child_attach = (base, follower) if direction == 1 else (follower, base)
                joint_placement = self._placements[parent] @ parent_attach
                body_placement = _inverse(child_attach)
                if j['type'] == 'fixed':
                    support = self._supports[parent]
                    body_placement = joint_placement @ body_placement
                else:
                    axis = direction * np.asarray(j['axis'], float)
                    kind = pin.JointModelRevoluteUnaligned if j['type'] == 'revolute' else pin.JointModelPrismaticUnaligned
                    support = int(self.model.addJoint(self._supports[parent], kind(axis), _se3(joint_placement), jid))
                    self.joint_ids[jid] = support
                add_body(child, support, body_placement, self.body_frame_ids[parent])
                visit(child)

        visit(self.root_body)
        if seen != set(bodies) or self.model.nq != len(self.coordinate_ids) or self.model.nv != self.model.nq:
            raise ValueError('Expected one scalar native coordinate per CMG coordinate')
        self._q_indices = np.array([self.model.joints[self.joint_ids[k]].idx_q for k in self.coordinate_ids], int)
        self._v_indices = np.array([self.model.joints[self.joint_ids[k]].idx_v for k in self.coordinate_ids], int)
        self.nq = self.nv = self.model.nq
        self.data = self.model.createData()

    def native_q(self, q):
        q = np.asarray(q, float)
        if q.shape != (self.nq,) or not np.all(np.isfinite(q)):
            raise ValueError('Invalid public configuration')
        out = np.empty(self.nq)
        out[self._q_indices] = q
        return out

    def native_v(self, value, name='v'):
        value = np.asarray(value, float)
        if value.shape != (self.nv,) or not np.all(np.isfinite(value)):
            raise ValueError(f'{name} must contain {self.nv} finite values')
        out = np.empty(self.nv)
        out[self._v_indices] = value
        return out

    def from_native_v(self, native):
        return np.asarray(native)[self._v_indices].copy()

    def mass(self, q):
        M = np.asarray(pin.crba(self.model, self.data, self.native_q(q))).copy()
        M = np.triu(M) + np.triu(M, 1).T
        return M[np.ix_(self._v_indices, self._v_indices)].copy()

    def bias(self, q, v):
        return self.from_native_v(pin.nonLinearEffects(self.model, self.data, self.native_q(q), self.native_v(v)))


class NativeSupportOracle:
    """Native loops (unchanged delivered oracle) plus native CONTACT_6D sole welds."""

    def __init__(self, comp, q, sites, mu_prox=1e-6, max_iterations=400, accuracy=1e-14):
        self.version = pin.__version__
        chart = comp.chart_cmg()
        self.backend = b = ChartPinBackend(chart)
        armature = [float(chart['armature'][k]) for k in chart['coordinate_ids']]
        self.loops = NativeLoopOracle(b, chart, armature, mu_prox=mu_prox, max_iterations=max_iterations,
                                      accuracy=accuracy)
        data = b.model.createData()
        pin.forwardKinematics(b.model, data, b.native_q(q))
        self.welds = []
        for name in sites:
            site = next(s for s in comp.cmg['sole_sites'] if s['id'] == name)
            F = np.eye(4)
            F[:3, :3] = site['frame_rotation']
            F[:3, 3] = site['frame_translation_m']
            support = b._supports[site['body']]
            placement = _se3(b._placements[site['body']] @ F)
            target = data.oMi[support] * placement
            c = pin.RigidConstraintModel(pin.ContactType.CONTACT_6D, b.model, support, placement, 0, target,
                                         pin.ReferenceFrame.LOCAL)
            c.name = 'weld_' + name
            if np.any(c.corrector.Kp) or np.any(c.corrector.Kd):
                raise RuntimeError('Native welds must have zero stabilization gains')
            self.loops.constraints.append(c)
            self.welds.append((support, placement))
        self.loops.constraint_data = [c.createData() for c in self.loops.constraints]
        pin.initConstraintDynamics(b.model, self.loops.data, self.loops.constraints)

    def acceleration(self, q, v, tau):
        return self.loops.acceleration(q, v, tau)

    def geometry(self, q, v):
        """Native rows in the NumPy order: 72 point, 8 universal, then [angular; linear] per weld."""
        geo = self.loops.geometry(q, v)
        m, d, idx = self.backend.model, self.loops.geometry_data, self.backend._v_indices
        rows, bias = [geo['jacobian']], [geo['bias']]
        for support, placement in self.welds:
            J = np.asarray(pin.getFrameJacobian(m, d, support, placement, LWA))[:, idx]
            acc = pin.getFrameClassicalAcceleration(m, d, support, placement, LWA)
            rows.append(np.vstack([J[3:], J[:3]]))
            bias.append(np.r_[np.asarray(acc.angular), np.asarray(acc.linear)])
        return dict(jacobian=np.vstack(rows), bias=np.concatenate(bias),
                    point_residual=geo['point_residual'], angular_residual=geo['angular_residual'])
