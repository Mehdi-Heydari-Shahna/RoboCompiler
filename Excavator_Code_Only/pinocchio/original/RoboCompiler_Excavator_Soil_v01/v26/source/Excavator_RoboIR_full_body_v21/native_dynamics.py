"""Native rigid-body components in the accepted source tree coordinates.

This adapter evaluates mass, energy and gravity terms only.  It does not solve
forward dynamics, step a simulator, assign an actuator effort, or infer the
unconnected p0 force.  The 23-coordinate tree is a computational representation;
the batch must project it through the separately checked seven-coordinate
kinematic map before interpreting closed-mechanism terms.

The potential_energy output is each engine's *native* value.  Pinocchio 3.8.0
omits inertias attached to its universe from potential energy; MuJoCo 3.3.7
includes the corresponding fixed body.  potential_energy_ground_offset exposes
the analytically known term to ADD to the native value for all physical bodies.
No fitted energy alignment or measured hardware gravity is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from pinocchio_backend import PinocchioBackend


class NativeDynamics:
    """Evaluate native components without changing the accepted source asset.

    qtree and velocity are finite one-dimensional arrays in ``tree_ids`` order.
    gravity3 is an explicit world-frame acceleration in m/s^2.  Generalized
    gravity is the positive compensation term g(q) in M qdd + ... + g = tau;
    it is not a cylinder-force solution or a constraint reaction.

    Instances own mutable backend data and are not intended for concurrent use.
    """

    def __init__(self, cmg, mjcf_path, mj_mapping):
        self.pin_backend = PinocchioBackend(cmg)
        self.tree_ids = list(self.pin_backend.tree_ids)
        self._make_compact_pinocchio_model()
        self.mj_model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path)))
        self.mj_data = mujoco.MjData(self.mj_model)
        if not isinstance(mj_mapping, Mapping):
            raise ValueError("Expected a MuJoCo source-coordinate mapping")
        self.mj_mapping = deepcopy(mj_mapping)
        self._configure_coordinate_map()
        self._audit_options()

    def _make_compact_pinocchio_model(self):
        """Copy to contiguous depth-first subtrees required by native CRBA.

        The accepted kinematic compiler uses breadth-first joint numbering.
        Its poses and Jacobians are valid, but native Pinocchio CRBA assumes
        compact subtrees.  This computational permutation changes no joint
        transform, inertia, coordinate value, or source record.
        """
        source = self.pin_backend.model
        self.pin_model = pin.Model()
        self.pin_model.name = 'excavator_cmg_pinocchio_compact_dynamics_v06'
        self.pin_model.gravity = pin.Motion.Zero()
        self.pin_model.inertias[0] = source.inertias[0]
        children = {index: [] for index in range(source.njoints)}
        for index in range(1, source.njoints):
            children[int(source.parents[index])].append(index)
        old_to_new = {0: 0}
        source_indices = []
        self.pin_compact_joint_ids = []

        def add_subtree(old_parent):
            for old_id in children[old_parent]:
                new_id = self.pin_model.addJoint(
                    old_to_new[old_parent], source.joints[old_id],
                    source.jointPlacements[old_id], source.names[old_id])
                old_to_new[old_id] = new_id
                self.pin_model.appendBodyToJoint(new_id, source.inertias[old_id], pin.SE3.Identity())
                source_indices.append(int(source.joints[old_id].idx_q))
                self.pin_compact_joint_ids.append(source.names[old_id])
                add_subtree(old_id)

        add_subtree(0)
        self._pin_old_to_new = old_to_new
        self._compact_source_indices = np.asarray(source_indices, dtype=int)
        self._source_compact_indices = np.argsort(self._compact_source_indices)
        if (self.pin_model.nq != source.nq or self.pin_model.nv != source.nv
                or sorted(source_indices) != list(range(source.nq))):
            raise ValueError("Pinocchio dynamics coordinate permutation is invalid")
        compact = self._has_compact_subtrees(self.pin_model)
        if not compact:
            raise ValueError("Pinocchio dynamics tree does not have contiguous subtrees")
        self.pin_dynamics_mapping = {
            'source_tree_ids': self.tree_ids.copy(),
            'compact_tree_ids': self.pin_compact_joint_ids.copy(),
            'compact_to_source_indices': self._compact_source_indices.tolist(),
            'source_to_compact_indices': self._source_compact_indices.tolist(),
            'configuration_map': 'q_compact = q_source[compact_to_source_indices]',
            'velocity_map': 'v_compact = v_source[compact_to_source_indices]',
            'covector_map': 'tau_source = tau_compact[source_to_compact_indices]',
            'subtree_compactness_verified': compact,
            'compactness_verified': compact,
            'reason': 'Pinocchio 3.8.0 CRBA requires contiguous depth-first subtrees',
            'source_geometry_or_inertia_changed': False}
        # Native dynamics must not overwrite the kinematic adapter's Jacobians.
        self.pin_data = self.pin_model.createData()

    def compact_body_poses(self, qtree):
        """Physical body poses from actual FK on the compact dynamics model."""
        q = self._vector(qtree, len(self.tree_ids), 'Tree configuration')
        data = self.pin_model.createData()
        pin.forwardKinematics(self.pin_model, data, q[self._compact_source_indices])
        poses = {}
        for body_id, frame_id in self.pin_backend.body_frame_ids.items():
            frame = self.pin_backend.model.frames[frame_id]
            transform = data.oMi[self._pin_old_to_new[frame.parentJoint]] * frame.placement
            poses[body_id] = np.asarray(transform.homogeneous).copy()
        return poses

    @staticmethod
    def _has_compact_subtrees(model):
        for root in range(1, model.njoints):
            descendants = []
            for candidate in range(root + 1, model.njoints):
                parent = int(model.parents[candidate])
                while parent and parent != root:
                    parent = int(model.parents[parent])
                if parent == root:
                    descendants.append(candidate)
            if descendants != list(range(root + 1, root + 1 + len(descendants))):
                return False
        return True

    def verify_compact_copy(self, qtree):
        """Check copied native joint/body placements and inertias independently.

        This deliberately runs FK on both Pinocchio models, using separate data
        objects.  It does not merely compare stored coordinate permutations.
        """
        q = self._vector(qtree, len(self.tree_ids), 'Tree configuration')
        original = self.pin_backend.model
        original_data, compact_data = original.createData(), self.pin_model.createData()
        pin.forwardKinematics(original, original_data, q)
        pin.updateFramePlacements(original, original_data)
        pin.forwardKinematics(self.pin_model, compact_data, q[self._compact_source_indices])
        max_position = max_rotation = max_inertia = 0.

        def compare_pose(first, second):
            nonlocal max_position, max_rotation
            max_position = max(max_position, float(np.linalg.norm(first.translation-second.translation)))
            rotation = first.rotation.T @ second.rotation
            axial = np.array([rotation[2, 1]-rotation[1, 2],
                              rotation[0, 2]-rotation[2, 0],
                              rotation[1, 0]-rotation[0, 1]]) / 2.
            angle = np.arctan2(np.linalg.norm(axial), (np.trace(rotation)-1.)/2.)
            max_rotation = max(max_rotation, float(abs(angle)))

        for old_index, new_index in self._pin_old_to_new.items():
            compare_pose(original_data.oMi[old_index], compact_data.oMi[new_index])
            source_inertia = np.asarray(original.inertias[old_index].matrix())
            copied_inertia = np.asarray(self.pin_model.inertias[new_index].matrix())
            max_inertia = max(max_inertia, float(np.linalg.norm(copied_inertia-source_inertia)
                                                / max(1., np.linalg.norm(source_inertia))))
        for frame_id in self.pin_backend.body_frame_ids.values():
            frame = original.frames[frame_id]
            copied_pose = compact_data.oMi[self._pin_old_to_new[frame.parentJoint]] * frame.placement
            compare_pose(original_data.oMf[frame_id], copied_pose)
        return {'max_joint_and_body_position_error_m': max_position,
                'max_joint_and_body_rotation_error_rad': max_rotation,
                'max_aggregated_inertia_error_scaled': max_inertia,
                'subtree_compactness_verified': self._has_compact_subtrees(self.pin_model)}

    @staticmethod
    def _vector(value, length, label):
        try:
            if np.iscomplexobj(value):
                raise ValueError(f"{label} must be real-valued")
            result = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} must contain finite scalar numbers") from error
        if result.shape != (length,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{label} must have shape ({length},) and be finite")
        return result.copy()

    def _configure_coordinate_map(self):
        model = self.mj_model
        n = len(self.tree_ids)
        if model.nq != n or model.nv != n or model.njnt != n:
            raise ValueError("Expected one scalar MuJoCo joint per source tree coordinate")
        if self.mj_mapping.get('schema') != 'excavator.mujoco.lowering/0.4':
            raise ValueError("Unsupported MuJoCo source mapping schema")
        ids = self.mj_mapping.get('tree_joint_ids', [])
        if len(ids) != n or len(set(ids)) != n or set(ids) != set(self.tree_ids):
            raise ValueError("MuJoCo and Pinocchio tree coordinates differ")
        records = self.mj_mapping.get('tree', [])
        moving = [record for record in records if record.get('type') != 'fixed']
        if len(moving) != n or {r.get('id') for r in moving} != set(self.tree_ids):
            raise ValueError("Invalid MuJoCo tree-coordinate records")
        by_id = {record['id']: record for record in moving}
        self._mj_q_indices = np.empty(n, dtype=int)
        self._mj_v_indices = np.empty(n, dtype=int)
        self._mj_sign = np.empty(n)
        self._mj_offset = np.empty(n)
        # v_mj = V @ v_source; generalized covectors transform by V.T.
        self._velocity_map = np.zeros((n, n))
        joint_types = {record['id']: record['type']
                       for record in self.pin_backend.cmg['joints']}
        for source_index, jid in enumerate(self.tree_ids):
            record = by_id[jid]
            joint_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if joint_index < 0:
                raise ValueError(f"Missing MuJoCo joint {jid}")
            expected_type = (mujoco.mjtJoint.mjJNT_HINGE if joint_types[jid] == 'revolute'
                             else mujoco.mjtJoint.mjJNT_SLIDE)
            if model.jnt_type[joint_index] != expected_type:
                raise ValueError(f"MuJoCo joint type differs at {jid}")
            try:
                sign = float(record['qpos_sign'])
                offset = float(record['qpos_offset'])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Invalid source-coordinate mapping at {jid}") from error
            if sign not in (-1., 1.) or not np.isfinite(offset):
                raise ValueError(f"Invalid source-coordinate sign or offset at {jid}")
            qi, vi = int(model.jnt_qposadr[joint_index]), int(model.jnt_dofadr[joint_index])
            self._mj_q_indices[source_index] = qi
            self._mj_v_indices[source_index] = vi
            self._mj_sign[source_index], self._mj_offset[source_index] = sign, offset
            self._velocity_map[vi, source_index] = sign
        if (len(set(self._mj_q_indices)) != n or len(set(self._mj_v_indices)) != n):
            raise ValueError("MuJoCo coordinate addresses are not one-to-one")

    def _audit_options(self):
        """Reject extra inertial/energy effects not present in this experiment."""
        model = self.mj_model
        if model.nu or model.ntendon or model.nflex or model.nplugin or model.nmocap:
            raise ValueError("Expected the accepted rigid-body asset without additional mechanisms")
        for name in ('dof_armature', 'dof_damping', 'dof_frictionloss',
                     'jnt_stiffness', 'body_gravcomp'):
            if np.any(np.asarray(getattr(model, name)) != 0.):
                raise ValueError(f"Unexpected MuJoCo {name} in the rigid-body experiment")
        if model.opt.density != 0. or model.opt.viscosity != 0.:
            raise ValueError("Unexpected MuJoCo fluid terms")
        if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY):
            raise ValueError("MuJoCo gravity is disabled by a flag")
        for name in ('armature', 'damping', 'friction'):
            if np.any(np.asarray(getattr(self.pin_model, name)) != 0.):
                raise ValueError(f"Unexpected Pinocchio {name} in the rigid-body experiment")

    def evaluate(self, qtree, velocity, gravity3):
        """Return native mass, gravity-compensation and energy in source order."""
        n = len(self.tree_ids)
        q = self._vector(qtree, n, 'Tree configuration')
        v = self._vector(velocity, n, 'Tree velocity')
        gravity = self._vector(gravity3, 3, 'World gravity')
        # Public models may be inspected by the batch.  Recheck that no extra
        # dynamics terms were introduced before each scientific comparison.
        self._audit_options()

        pmodel, pdata = self.pin_model, self.pin_data
        pmodel.gravity = pin.Motion.Zero()
        pmodel.gravity.linear = gravity
        pq, pv = q[self._compact_source_indices], v[self._compact_source_indices]
        # CRBA guarantees the upper triangle, not an initialized lower triangle.
        upper = np.triu(np.asarray(pin.crba(pmodel, pdata, pq)))
        compact_mass = upper + np.triu(upper, 1).T
        pin_mass = compact_mass[np.ix_(self._source_compact_indices, self._source_compact_indices)]
        pin_gravity = np.asarray(pin.computeGeneralizedGravity(pmodel, pdata, pq))[
            self._source_compact_indices].copy()
        pin_kinetic = float(pin.computeKineticEnergy(pmodel, pdata, pq, pv))
        pin_potential = float(pin.computePotentialEnergy(pmodel, pdata, pq))
        ground = pmodel.inertias[0]
        pin_ground_offset = -float(ground.mass) * float(np.dot(ground.lever, gravity))

        model, data = self.mj_model, self.mj_data
        model.opt.gravity[:] = gravity
        data.qpos[self._mj_q_indices] = self._mj_sign*q + self._mj_offset
        data.qvel[:] = 0.
        # The pipeline stops before force-to-acceleration and constraint solving.
        # In 3.3.7 CRB computes sparse M; makeM also populates legacy qM needed by fullM.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_crb(model, data)
        mujoco.mj_makeM(model, data)
        mj_mass_native = np.empty((n, n))
        mujoco.mj_fullM(model, mj_mass_native, data.qM)
        mujoco.mj_comVel(model, data)
        mujoco.mj_rne(model, data, 0, data.qfrc_bias)
        mj_gravity = self._velocity_map.T @ data.qfrc_bias.copy()
        mujoco.mj_energyPos(model, data)
        mj_potential = float(data.energy[0])
        data.qvel[:] = self._velocity_map @ v
        mujoco.mj_comVel(model, data)
        mujoco.mj_energyVel(model, data)
        mj_kinetic = float(data.energy[1])
        mj_mass = self._velocity_map.T @ mj_mass_native @ self._velocity_map

        result = {
            'pin': {'mass_matrix': pin_mass,
                    'gravity_compensation': pin_gravity,
                    'kinetic_energy': pin_kinetic,
                    'potential_energy': pin_potential,
                    'potential_energy_ground_offset': pin_ground_offset},
            'mujoco': {'mass_matrix': mj_mass,
                       'gravity_compensation': mj_gravity,
                       'kinetic_energy': mj_kinetic,
                       'potential_energy': mj_potential,
                       'potential_energy_ground_offset': 0.}}
        if any(not np.all(np.isfinite(value)) for values in result.values()
               for value in values.values()):
            raise ValueError("Native rigid-body evaluation produced a nonfinite result")
        return result
