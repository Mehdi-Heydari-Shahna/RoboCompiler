"""Independent CMG-to-Pinocchio plant with native loop-closure constraints.

This compiler is written for the Pinocchio backend and is separate from the
preserved ``pinocchio_backend.py`` used by the original controller:

* joints are added in true depth-first preorder, the subtree-contiguity
  condition required by Pinocchio CRBA;
* every source body's COM inertia is appended to its supporting joint through
  the complete fixed-edge placement chain;
* each of the nine cut revolute joints is represented by two native
  ``pin.RigidConstraintModel`` 3-D point constraints, at the joint origin and at
  a point 0.1 m along the joint axis (54 redundant rows, rank 16);
* constrained accelerations come from Pinocchio's own proximal
  ``constraintDynamics``.

The integrated plant state is the seven independent source coordinates
``q23, q7, q4, q0, q1, q21, q22`` and their rates.  The sixteen dependent tree
coordinates are never integrated: they are solved from the native closure
equations at every evaluation by Newton's method, and the tree velocity is the
exact null-space map of the native constraint Jacobian.  No integrated value is
replaced by a reference or controller value.

Spatial conventions: Pinocchio six-vectors are linear-first; frame Jacobians
and classical accelerations are LOCAL_WORLD_ALIGNED (world axes at the frame
origin).  The undercarriage ``body_53`` is fixed to the world by the CMG
``fixed_world_base`` record, exactly as in the source graph.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pinocchio as pin

INDEPENDENT_IDS = ['q23', 'q7', 'q4', 'q0', 'q1', 'q21', 'q22']
LIP_LOCAL = np.array([-.108, -1.96733, 1.44613])      # digging_path.LIP (source)
HEEL_LOCAL = np.array([-.108, -2.13268, .45688])      # digging_path.HEEL (source)


def _se3(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return pin.SE3(matrix[:3, :3].copy(), matrix[:3, 3].copy())


def _inverse(matrix):
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


def _rigid(matrix, label):
    matrix = np.asarray(matrix, dtype=float)
    rotation = matrix[:3, :3]
    if (matrix.shape != (4, 4) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0)
            or abs(np.linalg.det(rotation) - 1.) > 1e-9):
        raise ValueError(f'{label}: not a proper rigid transform')
    return matrix


def scaled_body_cmg(cmg, body_id, factor):
    """Plant-only inertial perturbation (mass and COM inertia scaled together)."""
    out = deepcopy(cmg)
    for body in out['bodies']:
        if body['id'] == body_id:
            body['mass_kg'] = float(body['mass_kg']) * factor
            body['inertia_com_kg_m2'] = (np.asarray(body['inertia_com_kg_m2']) * factor).tolist()
            return out
    raise KeyError(body_id)


class ExcavatorPinModel:
    """Depth-first CMG compiler, native loop constraints and point frames."""

    def __init__(self, cmg, cut_ids, lever=0.1, gravity=(0., 0., -9.81)):
        self.cmg = cmg
        self.bodies = {b['id']: b for b in cmg['bodies']}
        self.joints = {j['id']: j for j in cmg['joints']}
        if len(self.bodies) != len(cmg['bodies']) or len(self.joints) != len(cmg['joints']):
            raise ValueError('Duplicate CMG body or joint id')
        self.cut_ids = list(cut_ids)
        if any(self.joints[j]['type'] != 'revolute' for j in self.cut_ids):
            raise ValueError('Only revolute cut joints are supported')
        tree = [jid for jid in self.joints if jid not in self.cut_ids]
        if len(tree) != len(self.bodies) - 1:
            raise ValueError('Tree edges and bodies do not form a spanning tree')
        adjacency = {b: [] for b in self.bodies}
        for jid in sorted(tree):
            j = self.joints[jid]
            adjacency[j['base_body']].append((jid, j['follower_body'], 1))
            adjacency[j['follower_body']].append((jid, j['base_body'], -1))
        self.model = pin.Model()
        self.model.name = 'excavator_pinocchio_plant_dfs'
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = np.asarray(gravity, dtype=float)
        root = cmg['root_body']
        self.body_frame, self.support, self.placement = {}, {}, {}
        self.joint_index, self.direction = {}, {}
        self.support[root] = 0
        self.placement[root] = np.eye(4)
        self.body_frame[root] = int(self.model.addFrame(
            pin.Frame(root, 0, 0, pin.SE3.Identity(), pin.FrameType.BODY), False))
        seen = {root}

        def visit(parent):
            for jid, child, direction in adjacency[parent]:
                if child in seen:
                    continue
                seen.add(child)
                joint = self.joints[jid]
                base = _rigid(joint['T_BJ'], jid + '.T_BJ')
                follower = _rigid(joint['T_FJ'], jid + '.T_FJ')
                parent_attach, child_attach = (base, follower) if direction == 1 else (follower, base)
                joint_placement = self.placement[parent] @ parent_attach
                body_placement = _inverse(child_attach)
                if joint['type'] == 'fixed':
                    support = self.support[parent]
                    body_placement = joint_placement @ body_placement
                else:
                    axis = direction * np.asarray(joint['axis'], dtype=float)
                    if abs(np.linalg.norm(axis) - 1.) > 1e-12:
                        raise ValueError(f'{jid}: axis must be a unit vector')
                    kind = (pin.JointModelRevoluteUnaligned(axis) if joint['type'] == 'revolute'
                            else pin.JointModelPrismaticUnaligned(axis))
                    support = int(self.model.addJoint(self.support[parent], kind,
                                                      _se3(joint_placement), jid))
                    self.joint_index[jid] = support
                    self.direction[jid] = direction
                record = self.bodies[child]
                self.body_frame[child] = int(self.model.addFrame(pin.Frame(
                    child, support, self.body_frame[parent], _se3(body_placement),
                    pin.FrameType.BODY), False))
                if record['kind'] == 'rigid_body':
                    self.model.appendBodyToJoint(support, pin.Inertia(
                        float(record['mass_kg']), np.asarray(record['com_m'], dtype=float),
                        np.asarray(record['inertia_com_kg_m2'], dtype=float)), _se3(body_placement))
                self.support[child] = support
                self.placement[child] = body_placement
                visit(child)

        visit(root)
        if seen != set(self.bodies):
            raise ValueError('CMG graph is disconnected')
        self.tree_ids = [self.model.names[i] for i in range(1, self.model.njoints)]
        if self.model.nq != len(self.tree_ids) or self.model.nv != len(self.tree_ids):
            raise ValueError('Expected one scalar coordinate per tree joint')
        self.lever = float(lever)
        self.constraints = []
        self.endpoint_frames = []
        for jid in self.cut_ids:
            joint = self.joints[jid]
            axis = np.asarray(joint['axis'], dtype=float)
            for k, distance in enumerate((0., self.lever)):
                ends = []
                for body_key, transform_key in (('base_body', 'T_BJ'), ('follower_body', 'T_FJ')):
                    transform = np.asarray(joint[transform_key], dtype=float)
                    local = transform[:3, 3] + transform[:3, :3] @ (distance * axis)
                    body = joint[body_key]
                    placement = self.placement[body]
                    at_joint = placement[:3, 3] + placement[:3, :3] @ local
                    ends.append((self.support[body], pin.SE3(np.eye(3), at_joint), body))
                model = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, self.model,
                                                 ends[0][0], ends[0][1], ends[1][0], ends[1][1],
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                model.name = f'closure_{jid}_{k}'
                self.constraints.append(model)
                frames = []
                for side, (support, place, body) in zip('ab', ends):
                    frames.append(int(self.model.addFrame(pin.Frame(
                        f'{model.name}_{side}', support, self.body_frame[body], place,
                        pin.FrameType.OP_FRAME), False)))
                self.endpoint_frames.append(tuple(frames))
        self.point_frames = {}
        for name, local in (('lip', LIP_LOCAL), ('heel', HEEL_LOCAL),
                            ('mouth', .5 * (LIP_LOCAL + HEEL_LOCAL))):
            self.add_bucket_point(name, local)
        self.index = {jid: i for i, jid in enumerate(self.tree_ids)}
        self.data = self.model.createData()
        self.constraint_datas = [c.createData() for c in self.constraints]
        pin.initConstraintDynamics(self.model, self.data, self.constraints)

    def add_bucket_point(self, name, local, body='body_56'):
        placement = self.placement[body]
        at_joint = placement[:3, 3] + placement[:3, :3] @ np.asarray(local, dtype=float)
        self.point_frames[name] = int(self.model.addFrame(pin.Frame(
            'point_' + name, self.support[body], self.body_frame[body],
            pin.SE3(np.eye(3), at_joint), pin.FrameType.OP_FRAME), False))
        if hasattr(self, 'data'):
            self.data = self.model.createData()
            self.constraint_datas = [c.createData() for c in self.constraints]
            pin.initConstraintDynamics(self.model, self.data, self.constraints)
        return self.point_frames[name]

    # ------------------------------------------------------------------ kinematics
    def closure_residual(self, q):
        """Stacked base-minus-follower endpoint positions (world axes, metres)."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placements = self.data.oMf
        return np.concatenate([placements[a].translation - placements[b].translation
                               for a, b in self.endpoint_frames])

    def closure_jacobian(self, q):
        """Native constraint Jacobian (54 x 23) at q."""
        pin.computeJointJacobians(self.model, self.data, q)
        return np.array(pin.getConstraintsJacobian(self.model, self.data, self.constraints,
                                                   self.constraint_datas))

    def gamma(self, q, v):
        """Constraint acceleration drift J(q)qdot evaluated as classical accelerations at a=0."""
        pin.forwardKinematics(self.model, self.data, q, v, np.zeros(self.model.nv))
        frames = []
        for a, b in self.endpoint_frames:
            frames.append(pin.getFrameClassicalAcceleration(
                self.model, self.data, a, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
                - pin.getFrameClassicalAcceleration(
                self.model, self.data, b, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear)
        return np.concatenate(frames)

    def body_poses(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return {body: np.asarray(self.data.oMf[fid].homogeneous).copy()
                for body, fid in self.body_frame.items()}

    def points(self, q, names, *, refresh=True):
        """World positions, linear Jacobians and bucket rotation for named bucket points.

        If ``refresh`` is False the caller guarantees computeJointJacobians(q)
        was the latest kinematic call on ``self.data``.
        """
        if refresh:
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        out = {}
        for name in names:
            fid = self.point_frames[name]
            out[name] = (np.array(self.data.oMf[fid].translation),
                         np.array(pin.getFrameJacobian(self.model, self.data, fid,
                                                       pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))[:3])
        rotation = np.array(self.data.oMf[self.body_frame['body_56']].rotation)
        return out, rotation

    # -------------------------------------------------------------------- dynamics
    def mass_bias(self, q, v):
        mass = np.array(pin.crba(self.model, self.data, q))
        mass = np.triu(mass) + np.triu(mass, 1).T
        bias = np.array(pin.nonLinearEffects(self.model, self.data, q, v))
        return mass, bias

    def energy(self, q, v):
        kinetic = float(pin.computeKineticEnergy(self.model, self.data, q, v))
        potential = float(pin.computePotentialEnergy(self.model, self.data, q))
        return kinetic, potential


class ReducedPlant:
    """Seven-coordinate closed-chain plant using native Pinocchio constraint dynamics."""

    def __init__(self, cmg, cut_ids, independent_ids=INDEPENDENT_IDS, *, lever=0.1,
                 projection_tolerance_m=1e-12, proximal=(1e-12, 1e-9, 8)):
        self.model = ExcavatorPinModel(cmg, cut_ids, lever)
        self.n = self.model.model.nq
        self.independent_ids = list(independent_ids)
        self.active = np.array([self.model.index[j] for j in self.independent_ids], dtype=int)
        self.passive = np.setdiff1d(np.arange(self.n), self.active)
        self.tolerance = float(projection_tolerance_m)
        self.prox = pin.ProximalSettings(*proximal)
        self.q_cache = None
        self.u_cache = None
        self.N_cache = None
        self.J_cache = None
        self.error_cache = None
        self.stats = dict(projections=0, newton_iterations=0, max_newton_iterations=0,
                          max_projection_residual_m=0., proximal_iterations_max=0)

    def initialize(self, q_closed):
        q = np.asarray(q_closed, dtype=float).copy()
        residual = self.model.closure_residual(q)
        if np.max(np.abs(residual)) > 1e-9:
            raise ValueError('Initial plant configuration does not satisfy native closure')
        self.q_cache = q
        self.u_cache = None
        self.J_cache = None
        self.error_cache = float(np.max(np.abs(residual)))
        self.N_cache = self.tangent(self.model.closure_jacobian(q))
        self.u_cache = q[self.active].copy()
        self.J_cache = self.model.closure_jacobian(q)

    def project(self, u):
        """Solve the 16 dependent tree coordinates for independent coordinates u."""
        u = np.asarray(u, dtype=float)
        if self.u_cache is not None and np.array_equal(u, self.u_cache) and self.J_cache is not None:
            # Identical independent state: reuse the exact previous solution.
            self.model.closure_jacobian(self.q_cache)  # refresh joint Jacobians in data
            return self.q_cache.copy(), self.J_cache.copy(), self.N_cache.copy(), self.error_cache
        q = self.q_cache.copy()
        # First-order predictor along the latest exact tangent map.
        q[self.passive] += self.N_cache[self.passive] @ (u - self.u_cache)
        q[self.active] = u
        iterations = 0
        while True:
            residual = self.model.closure_residual(q)
            error = float(np.max(np.abs(residual)))
            if error <= self.tolerance:
                break
            if iterations >= 25:
                raise RuntimeError(f'Native closure projection failed ({error:.3e} m)')
            jacobian = self.model.closure_jacobian(q)
            q[self.passive] -= np.linalg.lstsq(jacobian[:, self.passive], residual, rcond=None)[0]
            iterations += 1
        jacobian = self.model.closure_jacobian(q)
        rank = np.linalg.matrix_rank(jacobian[:, self.passive], tol=1e-9)
        if rank != self.passive.size:
            raise RuntimeError(f'Native closure lost passive rank ({rank})')
        tangent = self.tangent(jacobian)
        self.q_cache, self.u_cache, self.N_cache = q.copy(), u.copy(), tangent
        self.J_cache, self.error_cache = jacobian.copy(), error
        s = self.stats
        s['projections'] += 1
        s['newton_iterations'] += iterations
        s['max_newton_iterations'] = max(s['max_newton_iterations'], iterations)
        s['max_projection_residual_m'] = max(s['max_projection_residual_m'], error)
        return q, jacobian, tangent, error

    def tangent(self, jacobian):
        tangent = np.zeros((self.n, self.active.size))
        tangent[self.active] = np.eye(self.active.size)
        tangent[self.passive] = np.linalg.lstsq(jacobian[:, self.passive],
                                                -jacobian[:, self.active], rcond=None)[0]
        return tangent

    def forward(self, q, v, tau):
        acceleration = np.array(pin.constraintDynamics(
            self.model.model, self.model.data, q, v, tau, self.model.constraints,
            self.model.constraint_datas, self.prox))
        self.stats['proximal_iterations_max'] = max(self.stats['proximal_iterations_max'],
                                                    int(self.prox.iter))
        return acceleration

    def reduced(self, q, v, tau, jacobian, tangent, gamma):
        """Independent reduced-coordinate solve used only as a cross-check."""
        mass, bias = self.model.mass_bias(q, v)
        curvature = np.zeros(self.n)
        curvature[self.passive] = np.linalg.lstsq(jacobian[:, self.passive], -gamma, rcond=None)[0]
        reduced_mass = tangent.T @ mass @ tangent
        udd = np.linalg.solve(reduced_mass, tangent.T @ (tau - bias - mass @ curvature))
        return tangent @ udd + curvature
