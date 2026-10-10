"""Independent CMG-to-Pinocchio scalar-tree compiler.

This module never reads or imports MJCF/MuJoCo.  Body inertias and joint
attachments are compiled directly from CMG records.  A source joint obeys
``T_WF = T_WB @ T_BJ @ motion(axis, q) @ inv(T_FJ)``.  Point closures are
handled by PACDM outside this tree backend.

Public q, v, a and effort vectors use ``cmg['coordinate_ids']`` order, even
when the graph traversal creates a different native Pinocchio ordering.
Every moving joint has one scalar configuration and one scalar velocity.
Pinocchio spatial motions/Jacobians are linear-first, angular-last; no
spatial six-vector is exposed by this module.  Body poses map body to world.
Inertias are given in body axes about their center of mass.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pinocchio as pin


def _se3(matrix: np.ndarray) -> pin.SE3:
    return pin.SE3(matrix[:3, :3].copy(), matrix[:3, 3].copy())


def _inverse(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def _transform(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}: expected a finite 4 x 4 transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9, rtol=0)):
        raise ValueError(f"{name}: transform must be a proper rigid transform")
    return matrix.copy()


class PinBackend:
    """Compile a connected CMG tree and evaluate independent rigid dynamics.

    Fixed edges preserve their full attachment transform and aggregate every
    body's inertia at its supporting moving joint.  The inertial placement
    therefore includes fixed chains, COM offsets, and body-frame rotations.
    There are no friction, damping, armature, or contact terms in these
    rigid-body dynamics.  Constraint forces are computed by PACDM.
    """

    def __init__(self, cmg: dict):
        self.cmg = deepcopy(cmg)
        self.coordinate_ids = list(cmg['coordinate_ids'])
        self.nq = len(self.coordinate_ids)
        self.nv = self.nq
        if len(set(self.coordinate_ids)) != self.nq:
            raise ValueError("coordinate_ids must be unique")
        bodies = {body['id']: body for body in cmg['bodies']}
        joints = {joint['id']: joint for joint in cmg['joints']}
        if len(bodies) != len(cmg['bodies']) or len(joints) != len(cmg['joints']):
            raise ValueError("Body and joint IDs must each be unique")
        self.root_body = cmg.get('root_body', 'world')
        if self.root_body not in bodies:
            raise ValueError("root_body is not present in CMG bodies")
        moving = {jid for jid, joint in joints.items() if joint['type'] != 'fixed'}
        if moving != set(self.coordinate_ids):
            raise ValueError("coordinate_ids must contain each moving tree joint exactly once")
        if len(joints) != len(bodies) - 1:
            raise ValueError("PinBackend requires a tree; closures must be separate CMG records")

        adjacency = {body_id: [] for body_id in bodies}
        for jid, joint in joints.items():
            if joint['type'] not in ('fixed', 'revolute', 'prismatic'):
                raise ValueError(f"Unsupported joint type at {jid}: {joint['type']}")
            base, follower = joint['base_body'], joint['follower_body']
            if base not in bodies or follower not in bodies or base == follower:
                raise ValueError(f"Invalid body endpoints for joint {jid}")
            adjacency[base].append((jid, follower, 1))
            adjacency[follower].append((jid, base, -1))

        self.model = pin.Model()
        self.model.name = cmg.get('name', 'cmg_scalar_tree')
        gravity = np.asarray(cmg.get('gravity_m_s2', [0., 0., -9.81]), dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity_m_s2 must have three finite components")
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = gravity
        self.body_frame_ids = {}
        self.joint_ids = {}
        self._supports = {}
        self._placements = {}

        def add_body(body_id, support, placement, parent_frame):
            frame = pin.Frame(body_id, support, parent_frame, _se3(placement),
                              pin.FrameType.BODY)
            self.body_frame_ids[body_id] = int(self.model.addFrame(frame, False))
            self._supports[body_id] = support
            self._placements[body_id] = placement.copy()
            body = bodies[body_id]
            mass = float(body.get('mass_kg', 0.))
            com = np.asarray(body.get('com_m', [0., 0., 0.]), dtype=float)
            tensor = np.asarray(body.get('inertia_kg_m2', np.zeros((3, 3))), dtype=float)
            if (not np.isfinite(mass) or mass < 0. or com.shape != (3,)
                    or tensor.shape != (3, 3) or not np.all(np.isfinite(com))
                    or not np.all(np.isfinite(tensor))
                    or not np.allclose(tensor, tensor.T, atol=1e-12, rtol=0)
                    or np.min(np.linalg.eigvalsh(tensor)) < -1e-12):
                raise ValueError(f"Invalid mass, COM, or inertia tensor for body {body_id}")
            if mass == 0.:
                if np.any(tensor != 0.):
                    raise ValueError(f"Zero-mass body {body_id} must have zero inertia")
                return  # Massless intermediate scalar stages are intentional.
            self.model.appendBodyToJoint(support, pin.Inertia(mass, com, tensor),
                                         _se3(placement))

        add_body(self.root_body, 0, np.eye(4), 0)
        seen = {self.root_body}
        used_edges = set()

        def visit(parent):
            # Pinocchio's CRBA requires each subtree's joints to occupy one
            # contiguous interval.  Breadth-first numbering is valid for FK
            # and RNEA but silently loses CRBA couplings on branched trees.
            # True depth-first preorder is therefore a dynamics requirement.
            for jid, child, direction in adjacency[parent]:
                if jid in used_edges:
                    continue
                if child in seen:
                    raise ValueError(f"CMG tree contains a cycle at joint {jid}")
                seen.add(child)
                used_edges.add(jid)
                joint = joints[jid]
                base = _transform(joint['T_BJ'], f"{jid}.T_BJ")
                follower = _transform(joint['T_FJ'], f"{jid}.T_FJ")
                parent_attach, child_attach = ((base, follower) if direction == 1
                                                else (follower, base))
                parent_support = self._supports[parent]
                joint_placement = self._placements[parent] @ parent_attach
                body_placement = _inverse(child_attach)
                if joint['type'] == 'fixed':
                    support = parent_support
                    body_placement = joint_placement @ body_placement
                else:
                    axis = np.asarray(joint['axis'], dtype=float)
                    if (axis.shape != (3,) or not np.all(np.isfinite(axis))
                            or not np.isclose(np.linalg.norm(axis), 1., atol=1e-9, rtol=0)):
                        raise ValueError(f"{jid}.axis must be a unit three-vector")
                    axis = direction * axis
                    kind = (pin.JointModelRevoluteUnaligned if joint['type'] == 'revolute'
                            else pin.JointModelPrismaticUnaligned)
                    support = int(self.model.addJoint(parent_support, kind(axis),
                                                      _se3(joint_placement), jid))
                    self.joint_ids[jid] = support
                add_body(child, support, body_placement, self.body_frame_ids[parent])
                visit(child)

        visit(self.root_body)
        if seen != set(bodies):
            raise ValueError("CMG tree is disconnected from root_body")
        if self.model.nq != self.nq or self.model.nv != self.nv:
            raise ValueError("Expected one scalar configuration and velocity per moving joint")

        self._q_indices = np.asarray([
            self.model.joints[self.joint_ids[jid]].idx_q for jid in self.coordinate_ids
        ], dtype=int)
        self._v_indices = np.asarray([
            self.model.joints[self.joint_ids[jid]].idx_v for jid in self.coordinate_ids
        ], dtype=int)
        self._mass_indices = np.ix_(self._v_indices, self._v_indices)
        self.data = self.model.createData()

    def _vector(self, value, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (self.nq,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {self.nq} finite scalar values")
        return result

    def _native_q(self, q):
        native = np.empty(self.nq)
        native[self._q_indices] = self._vector(q, 'q')
        return native

    def _native_v(self, value, name):
        native = np.empty(self.nv)
        native[self._v_indices] = self._vector(value, name)
        return native

    def mass(self, q):
        """Return the symmetric unconstrained tree mass matrix in CMG order."""
        matrix = np.asarray(pin.crba(self.model, self.data, self._native_q(q))).copy()
        # CRBA specifies its upper triangle; never depend on stale lower data.
        matrix = np.triu(matrix) + np.triu(matrix, 1).T
        return matrix[self._mass_indices].copy()

    def inverse(self, q, v, a):
        """Return M(q) a + C(q,v) v + g(q), without loop/contact forces."""
        effort = pin.rnea(self.model, self.data, self._native_q(q),
                          self._native_v(v, 'v'), self._native_v(a, 'a'))
        return np.asarray(effort)[self._v_indices].copy()

    def bias(self, q, v):
        """Return Coriolis, centrifugal, and gravity terms in CMG order."""
        effort = pin.nonLinearEffects(self.model, self.data, self._native_q(q),
                                      self._native_v(v, 'v'))
        return np.asarray(effort)[self._v_indices].copy()

    def poses(self, q):
        """Return body-to-world homogeneous matrices for all source bodies."""
        pin.forwardKinematics(self.model, self.data, self._native_q(q))
        pin.updateFramePlacements(self.model, self.data)
        return {body: np.asarray(self.data.oMf[frame_id].homogeneous).copy()
                for body, frame_id in self.body_frame_ids.items()}

    def energy(self, q, v):
        """Return kinetic, potential and total rigid-body energy, in joules."""
        native_q = self._native_q(q)
        native_v = self._native_v(v, 'v')
        kinetic = float(pin.computeKineticEnergy(self.model, self.data, native_q, native_v))
        potential = float(pin.computePotentialEnergy(self.model, self.data, native_q))
        return {'kinetic_J': kinetic, 'potential_J': potential,
                'total_J': kinetic + potential}
