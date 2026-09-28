"""Independent CMG-to-Pinocchio compiler for the floating Kangaroo tree.

Same design as the Stewart/Panda ``PinBackend``: bodies and joint attachments
are compiled directly from CMG records (no URDF, MJCF or MuJoCo), a source
joint obeys ``T_WF = T_WB @ T_BJ @ motion(axis, q) @ inv(T_FJ)``, joints are
inserted in true depth-first preorder (required by CRBA), fixed bodies are
merged onto their supporting moving joint, and every public vector uses the
CMG coordinate order through explicit native index maps.  Loop closures are
not part of this tree; PACDM handles them.

Floating base extension (Pinocchio free-flyer convention):

* configuration ``q = [p_world(3), quat_xyzw(4), q_int(76, CMG order)]``;
* velocity ``v = [v_base_local(3), omega_base_local(3), qd_int(76, CMG order)]``
  where the first six entries are the base twist in the base frame;
* efforts and accelerations use the velocity order.

Pinocchio six-vectors are linear-first.  Body poses map body to world and
inertias are expressed in body axes about the body COM.  There are no
friction, damping, armature, contact or closure terms in this tree; the
simulation adds armature and viscous damping explicitly from the CMG.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pinocchio as pin

GRAVITY = np.array([0., 0., -9.81])


def _se3(matrix: np.ndarray) -> pin.SE3:
    matrix = np.asarray(matrix, dtype=float)
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


class FloatingPinBackend:
    """Compile a connected CMG tree with a free-flyer root joint.

    ``extra_frames`` maps a new frame name to ``(body_id, T_body_frame)``;
    such frames are rigidly attached operational frames (for example foot
    corners or a disturbance application point) and add no inertia.
    """

    def __init__(self, cmg: dict, extra_frames: dict | None = None, gravity=GRAVITY):
        self.cmg = deepcopy(cmg)
        self.coordinate_ids = list(cmg['coordinate_ids'])
        self.n_int = len(self.coordinate_ids)
        if len(set(self.coordinate_ids)) != self.n_int:
            raise ValueError("coordinate_ids must be unique")
        bodies = {body['id']: body for body in cmg['bodies']}
        joints = {joint['id']: joint for joint in cmg['joints']}
        if len(bodies) != len(cmg['bodies']) or len(joints) != len(cmg['joints']):
            raise ValueError("Body and joint IDs must each be unique")
        self.root_body = cmg['root_body']
        if self.root_body not in bodies:
            raise ValueError("root_body is not present in CMG bodies")
        moving = {jid for jid, joint in joints.items() if joint['type'] != 'fixed'}
        if moving != set(self.coordinate_ids):
            raise ValueError("coordinate_ids must contain each moving tree joint exactly once")
        if len(joints) != len(bodies) - 1:
            raise ValueError("Backend requires a spanning tree; closures are separate CMG records")

        adjacency = {body_id: [] for body_id in bodies}
        for jid in sorted(joints):
            joint = joints[jid]
            if joint['type'] not in ('fixed', 'revolute', 'prismatic'):
                raise ValueError(f"Unsupported joint type at {jid}: {joint['type']}")
            base, follower = joint['base_body'], joint['follower_body']
            if base not in bodies or follower not in bodies or base == follower:
                raise ValueError(f"Invalid body endpoints for joint {jid}")
            adjacency[base].append((jid, follower, 1))
            adjacency[follower].append((jid, base, -1))

        self.model = pin.Model()
        self.model.name = 'kangaroo_cmg_floating_tree'
        gravity = np.asarray(gravity, dtype=float)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            raise ValueError("gravity must have three finite components")
        self.model.gravity = pin.Motion.Zero()
        self.model.gravity.linear = gravity
        self.body_frame_ids = {}
        self.joint_ids = {}
        self._supports = {}
        self._placements = {}
        self.body_mass = {}

        def add_body(body_id, support, placement, parent_frame):
            frame = pin.Frame(body_id, support, parent_frame, _se3(placement), pin.FrameType.BODY)
            self.body_frame_ids[body_id] = int(self.model.addFrame(frame, False))
            self._supports[body_id] = support
            self._placements[body_id] = placement.copy()
            body = bodies[body_id]
            if body.get('kind', 'rigid_body') != 'rigid_body':
                raise ValueError(f"{body_id}: floating backend expects physical rigid bodies")
            mass = float(body['mass_kg'])
            com = np.asarray(body['com_m'], dtype=float)
            tensor = np.asarray(body['inertia_com_kg_m2'], dtype=float)
            if (not np.isfinite(mass) or mass <= 0. or com.shape != (3,)
                    or tensor.shape != (3, 3) or not np.all(np.isfinite(com))
                    or not np.all(np.isfinite(tensor))
                    or not np.allclose(tensor, tensor.T, atol=1e-12, rtol=0)
                    or np.min(np.linalg.eigvalsh(tensor)) < -1e-12):
                raise ValueError(f"Invalid mass, COM, or inertia tensor for body {body_id}")
            self.body_mass[body_id] = mass
            self.model.appendBodyToJoint(support, pin.Inertia(mass, com, tensor), _se3(placement))

        root_joint = int(self.model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(),
                                             'floating_base'))
        self.root_joint = root_joint
        add_body(self.root_body, root_joint, np.eye(4), 0)
        seen = {self.root_body}
        used_edges = set()

        def visit(parent):
            # True depth-first preorder: CRBA requires contiguous subtrees.
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
        if self.model.nq != 7 + self.n_int or self.model.nv != 6 + self.n_int:
            raise ValueError("Expected a free-flyer root plus one scalar coordinate per joint")
        self.extra_frame_ids = {}
        for name, (body_id, placement) in (extra_frames or {}).items():
            T = _transform(placement, f'extra frame {name}')
            frame = pin.Frame(name, self._supports[body_id], self.body_frame_ids[body_id],
                              _se3(self._placements[body_id] @ T), pin.FrameType.OP_FRAME)
            self.extra_frame_ids[name] = int(self.model.addFrame(frame, False))

        q_int = np.asarray([self.model.joints[self.joint_ids[j]].idx_q for j in self.coordinate_ids])
        v_int = np.asarray([self.model.joints[self.joint_ids[j]].idx_v for j in self.coordinate_ids])
        root = self.model.joints[root_joint]
        if root.idx_q != 0 or root.idx_v != 0:
            raise ValueError("Free-flyer must occupy the first native coordinates")
        self._q_indices = np.r_[np.arange(7), q_int].astype(int)
        self._v_indices = np.r_[np.arange(6), v_int].astype(int)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.total_mass = float(sum(self.body_mass.values()))
        self.data = self.model.createData()

    # ----- ordering helpers -------------------------------------------------
    def _vector(self, value, size, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain {size} finite values")
        return result

    def native_q(self, q):
        q = self._vector(q, self.nq, 'q')
        quat = q[3:7]
        if not np.isclose(np.linalg.norm(quat), 1., atol=1e-9, rtol=0):
            raise ValueError("Base quaternion must be unit length")
        native = np.empty(self.nq)
        native[self._q_indices] = q
        return native

    def native_v(self, value, name='v'):
        value = self._vector(value, self.nv, name)
        native = np.empty(self.nv)
        native[self._v_indices] = value
        return native

    def from_native_v(self, native):
        return np.asarray(native)[self._v_indices].copy()

    def from_native_q(self, native):
        return np.asarray(native)[self._q_indices].copy()

    @staticmethod
    def compose(position, rotation, q_int):
        """Build a public configuration from world position, rotation matrix and q_int."""
        quat = pin.Quaternion(np.asarray(rotation, dtype=float)).coeffs()  # x y z w
        return np.r_[position, quat, q_int]

    # ----- dynamics -----------------------------------------------------------
    def mass(self, q):
        """Symmetric rigid-tree mass matrix in public order (no armature)."""
        matrix = np.asarray(pin.crba(self.model, self.data, self.native_q(q))).copy()
        matrix = np.triu(matrix) + np.triu(matrix, 1).T
        return matrix[np.ix_(self._v_indices, self._v_indices)].copy()

    def bias(self, q, v):
        """Coriolis, centrifugal and gravity terms h(q, v) in public order."""
        effort = pin.nonLinearEffects(self.model, self.data, self.native_q(q), self.native_v(v))
        return self.from_native_v(effort)

    def gravity(self, q):
        effort = pin.computeGeneralizedGravity(self.model, self.data, self.native_q(q))
        return self.from_native_v(effort)

    def inverse(self, q, v, a, external=None):
        """RNEA: M a + h - sum J^T f_ext.  ``external`` is a native pin.StdVec_Force or None."""
        if external is None:
            effort = pin.rnea(self.model, self.data, self.native_q(q), self.native_v(v),
                              self.native_v(a, 'a'))
        else:
            effort = pin.rnea(self.model, self.data, self.native_q(q), self.native_v(v),
                              self.native_v(a, 'a'), external)
        return self.from_native_v(effort)

    def forward(self, q, v, tau):
        """Unconstrained tree forward dynamics (ABA) in public order."""
        acc = pin.aba(self.model, self.data, self.native_q(q), self.native_v(v),
                      self.native_v(tau, 'tau'))
        return self.from_native_v(acc)

    def poses(self, q):
        """Body-to-world homogeneous matrices for every CMG body."""
        pin.forwardKinematics(self.model, self.data, self.native_q(q))
        pin.updateFramePlacements(self.model, self.data)
        return {body: np.asarray(self.data.oMf[fid].homogeneous).copy()
                for body, fid in self.body_frame_ids.items()}

    def frame_placement(self, q, name):
        pin.forwardKinematics(self.model, self.data, self.native_q(q))
        fid = self.extra_frame_ids.get(name, self.body_frame_ids.get(name))
        return np.asarray(pin.updateFramePlacement(self.model, self.data, fid).homogeneous).copy()

    def frame_jacobians(self, q, names, reference=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED):
        """Placements and 6 x nv Jacobians (public column order) for named frames."""
        pin.computeJointJacobians(self.model, self.data, self.native_q(q))
        pin.updateFramePlacements(self.model, self.data)
        out = {}
        for name in names:
            fid = self.extra_frame_ids.get(name, self.body_frame_ids.get(name))
            J = np.asarray(pin.getFrameJacobian(self.model, self.data, fid, reference))
            out[name] = (np.asarray(self.data.oMf[fid].homogeneous).copy(),
                         J[:, self._v_indices].copy())
        return out

    def energy(self, q, v):
        """Kinetic, potential and total rigid-body energy (no armature), J."""
        nq, nv = self.native_q(q), self.native_v(v)
        kinetic = float(pin.computeKineticEnergy(self.model, self.data, nq, nv))
        potential = float(pin.computePotentialEnergy(self.model, self.data, nq))
        return {'kinetic_J': kinetic, 'potential_J': potential, 'total_J': kinetic + potential}

    def center_of_mass(self, q, v=None):
        nq = self.native_q(q)
        if v is None:
            return np.asarray(pin.centerOfMass(self.model, self.data, nq)).copy()
        pin.centerOfMass(self.model, self.data, nq, self.native_v(v))
        return np.asarray(self.data.com[0]).copy(), np.asarray(self.data.vcom[0]).copy()

    def centroidal_momentum(self, q, v):
        """Linear and angular momentum about the COM, world axes."""
        h = pin.computeCentroidalMomentum(self.model, self.data, self.native_q(q), self.native_v(v))
        return np.asarray(h.linear).copy(), np.asarray(h.angular).copy()

    def integrate(self, q, v, dt):
        """Configuration update on the free-flyer manifold (base) and R^n (joints)."""
        out = pin.integrate(self.model, self.native_q(q), self.native_v(v * dt, 'v dt'))
        return self.from_native_q(out)
