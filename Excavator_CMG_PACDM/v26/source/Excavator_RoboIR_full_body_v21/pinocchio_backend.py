"""Compile the accepted excavator CMG directly to a Pinocchio tree and closures.

The tree is a computational representation, not a replacement of the source
physical graph.  Nine omitted revolute edges are represented by two coincident
axis points each.  All point closures and their analytic Jacobians are evaluated
using Pinocchio physical-body frames.  No MuJoCo model, metadata, or exporter is
used in this module.

Native configuration and velocity coordinates equal source tree-joint values in
SI units.  Reversed tree edges use the negative source joint axis.  No nominal
configuration offset is baked into a joint placement.  Spatial Jacobians use
linear-first, world-aligned axes at each physical body's origin.

Only local kinematics is certified by this milestone.  The source inertias are
loaded, including aggregation at fixed connections, but dynamics and force
semantics are not validated.  In particular, p0 effort remains unspecified.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pinocchio as pin

from cmg_io import validate_cmg


def _se3(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return pin.SE3(matrix[:3, :3].copy(), matrix[:3, 3].copy())


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _inverse(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _rotation_error(rotation):
    axial = np.array([rotation[2, 1] - rotation[1, 2],
                      rotation[0, 2] - rotation[2, 0],
                      rotation[1, 0] - rotation[0, 1]]) / 2.
    return float(np.arctan2(np.linalg.norm(axial), (np.trace(rotation)-1.)/2.))


class PinocchioBackend:
    """Source-specific CMG backend with explicit physical-frame mapping.

    ``tree_ids`` is the order of native q and v (one scalar per joint).
    ``independent_ids`` retains the CMG's seven-coordinate ordering;
    ``passive_ids`` lists the remaining sixteen tree coordinates.

    Constructor validation checks the *physical* reference configuration as
    well as the schema.  Coordinate updates deliberately permit open-loop
    states so that closure solving and independent derivative checks can run.
    """

    def __init__(self, cmg, axis_lever_m=0.1):
        validate_cmg(cmg)
        if (not np.isscalar(axis_lever_m) or not np.isfinite(axis_lever_m)
                or axis_lever_m <= 0):
            raise ValueError("Closure axis-point lever must be finite and positive")
        self.cmg = deepcopy(cmg)
        self.axis_lever_m = float(axis_lever_m)
        self._bodies = {record['id']: record for record in self.cmg['bodies']}
        self._joints = {record['id']: record for record in self.cmg['joints']}
        self.root_body = self.cmg['root_body']
        self.independent_ids = list(self.cmg['interpretation']['full_partition']['independent_ids'])
        reference = self.cmg['interpretation']['reference']
        self.reference = dict(zip(reference['coordinate_ids'], reference['q_SI']))
        self._tree_edges, self.cut_ids = self._select_tree()
        self.model = pin.Model()
        self.model.name = 'excavator_cmg_pinocchio_v05'
        # No time integration is performed; zero gravity prevents this kinematic
        # asset from silently implying a validated gravitational experiment.
        self.model.gravity = pin.Motion.Zero()
        self.body_frame_ids = {}
        self._supports = {}
        self._placements = {}
        self._tree_records = []
        self._inertial_records = []
        self.tree_ids = []
        self.joint_ids = {}
        self._build_tree()
        self.passive_ids = [jid for jid in self.tree_ids if jid not in self.independent_ids]
        self.q_indices = {jid: int(self.model.joints[self.joint_ids[jid]].idx_q)
                          for jid in self.tree_ids}
        self.v_indices = {jid: int(self.model.joints[self.joint_ids[jid]].idx_v)
                          for jid in self.tree_ids}
        if (self.model.nq != len(self.tree_ids) or self.model.nv != len(self.tree_ids)
                or [self.q_indices[jid] for jid in self.tree_ids] != list(range(self.model.nq))):
            raise ValueError("Expected scalar source-valued Pinocchio tree coordinates")
        self._cylinders = self._cylinder_records()
        self.data = self.model.createData()
        self.set_configuration(self.reference)
        self._validate_reference()
        self.metadata = self._metadata()

    def _select_tree(self):
        """Kruskal-style tree selection using source records, then deterministic BFS."""
        parents = {body: body for body in self._bodies}

        def representative(body):
            while parents[body] != body:
                parents[body] = parents[parents[body]]
                body = parents[body]
            return body

        required = {joint['id'] for joint in self._joints.values()
                    if joint['type'] in ('fixed', 'prismatic')}
        required.update(self.independent_ids)
        ordered = sorted(required) + sorted(set(self._joints)-required)
        selected = []
        for jid in ordered:
            joint = self._joints[jid]
            a, b = representative(joint['base_body']), representative(joint['follower_body'])
            if a == b:
                if jid in required:
                    raise ValueError(f"Required tree coordinates form a cycle at {jid}")
                continue
            parents[b] = a
            selected.append(jid)
        if len(selected) != len(self._bodies)-1:
            raise ValueError("Physical graph is disconnected")
        omitted = sorted(set(self._joints)-set(selected))
        if any(self._joints[jid]['type'] != 'revolute' for jid in omitted):
            raise ValueError("This backend closure representation supports cut revolute joints only")
        return selected, omitted

    def _add_physical_frame(self, body, support, placement, parent_frame):
        frame = pin.Frame(body, support, parent_frame, _se3(placement), pin.FrameType.BODY)
        frame_id = int(self.model.addFrame(frame, False))
        self.body_frame_ids[body] = frame_id
        self._supports[body] = support
        self._placements[body] = placement.copy()
        record = self._bodies[body]
        if record['kind'] == 'rigid_body':
            inertia = pin.Inertia(float(record['mass_kg']),
                                  np.asarray(record['com_m'], dtype=float),
                                  np.asarray(record['inertia_com_kg_m2'], dtype=float))
            self.model.appendBodyToJoint(support, inertia, _se3(placement))
            self._inertial_records.append({
                'body_id': body, 'support_joint_index': support,
                'body_to_support_placement': placement.tolist(),
                'T_joint_body': placement.tolist(),
                'source_mass_kg': record['mass_kg'],
                'source_com_m': deepcopy(record['com_m']),
                'source_inertia_com_kg_m2': deepcopy(record['inertia_com_kg_m2'])})

    def _build_tree(self):
        adjacency = {body: [] for body in self._bodies}
        for jid in self._tree_edges:
            joint = self._joints[jid]
            adjacency[joint['base_body']].append((jid, joint['follower_body'], 1))
            adjacency[joint['follower_body']].append((jid, joint['base_body'], -1))
        self._add_physical_frame(self.root_body, 0, np.eye(4), 0)
        queue = deque([self.root_body])
        seen = {self.root_body}
        while queue:
            parent = queue.popleft()
            for jid, child, direction in sorted(adjacency[parent]):
                if child in seen:
                    continue
                seen.add(child)
                queue.append(child)
                joint = self._joints[jid]
                base = np.asarray(joint['T_BJ'], dtype=float)
                follower = np.asarray(joint['T_FJ'], dtype=float)
                parent_attach, child_attach = (base, follower) if direction == 1 else (follower, base)
                parent_joint = self._supports[parent]
                fixed_to_parent_joint = self._placements[parent] @ parent_attach
                child_frame_placement = _inverse(child_attach)
                if joint['type'] == 'fixed':
                    support = parent_joint
                    placement = fixed_to_parent_joint @ child_frame_placement
                    joint_index = None
                else:
                    axis = direction * np.asarray(joint['axis'], dtype=float)
                    if joint['type'] == 'revolute':
                        joint_model = pin.JointModelRevoluteUnaligned(axis)
                    elif joint['type'] == 'prismatic':
                        joint_model = pin.JointModelPrismaticUnaligned(axis)
                    else:
                        raise ValueError(f"Unsupported joint type at {jid}")
                    support = int(self.model.addJoint(parent_joint, joint_model,
                                                      _se3(fixed_to_parent_joint), jid))
                    joint_index = support
                    self.joint_ids[jid] = support
                    self.tree_ids.append(jid)
                    placement = child_frame_placement
                self._add_physical_frame(child, support, placement, self.body_frame_ids[parent])
                self._tree_records.append({
                    'source_joint_id': jid, 'parent_body': parent, 'child_body': child,
                    'source_direction': direction, 'pinocchio_joint_index': joint_index,
                    'source_coordinate_scale': 1.0, 'source_coordinate_offset': 0.0})

    def _cylinder_records(self):
        records = []
        for joint in sorted(self._joints.values(), key=lambda r: r['id']):
            if joint['type'] != 'prismatic':
                continue
            endpoints = []
            for body in (joint['base_body'], joint['follower_body']):
                pins = [j for j in self._joints.values() if j['type'] == 'revolute'
                        and body in (j['base_body'], j['follower_body'])]
                if len(pins) != 1:
                    raise ValueError(f"Ambiguous cylinder attachment for {joint['id']} / {body}")
                attachment = pins[0]
                key = 'T_BJ' if attachment['base_body'] == body else 'T_FJ'
                endpoints.append({'body_id': body, 'source_joint_id': attachment['id'],
                                  'point_in_body_m': np.asarray(attachment[key])[:3, 3].tolist()})
            records.append({'source_joint_id': joint['id'], 'endpoints': endpoints})
        return records

    def set_configuration(self, coordinates):
        """Update kinematics from a source-coordinate dictionary or native q vector."""
        if isinstance(coordinates, Mapping):
            missing = set(self.tree_ids)-set(coordinates)
            if missing:
                raise ValueError(f"Missing tree coordinates: {sorted(missing)}")
            try:
                supplied = np.asarray(list(coordinates.values()), dtype=float)
                q = np.asarray([coordinates[jid] for jid in self.tree_ids], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError("Coordinates must be finite scalar numbers") from error
            if supplied.ndim != 1 or not np.all(np.isfinite(supplied)):
                raise ValueError("Coordinates must be finite scalar numbers")
        else:
            try:
                q = np.asarray(coordinates, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError("Coordinates must be finite scalar numbers") from error
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Expected {self.model.nq} finite scalar tree coordinates")
        self.q = q.copy()
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        return self

    def body_poses(self):
        """World poses for every source physical body and the root world frame."""
        return {body: np.asarray(self.data.oMf[fid].homogeneous).copy()
                for body, fid in self.body_frame_ids.items()}

    def body_jacobians(self):
        """Return linear-first world-aligned Jacobians at physical-body origins."""
        return {body: np.asarray(pin.getFrameJacobian(
                    self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).copy()
                for body, fid in self.body_frame_ids.items()}

    @staticmethod
    def _point(poses, jacobians, body, local):
        transform = poses[body]
        displacement = transform[:3, :3] @ local
        position = transform[:3, 3] + displacement
        body_jacobian = jacobians[body]
        jacobian = body_jacobian[:3] - _skew(displacement) @ body_jacobian[3:]
        return position, jacobian

    def closure(self):
        """Return 54 point residuals in metres and their 54 x 23 analytic Jacobian.

        Rows are ordered by sorted cut-joint id, first axis origin then the
        second axis point.  Each residual is base-point minus follower-point in
        world axes.  Redundant rows are retained; rank is determined separately.
        """
        poses, jacobians = self.body_poses(), self.body_jacobians()
        residuals, derivatives = [], []
        for jid in self.cut_ids:
            joint = self._joints[jid]
            axis = np.asarray(joint['axis'], dtype=float)
            for lever in (0., self.axis_lever_m):
                base_transform = np.asarray(joint['T_BJ'], dtype=float)
                follower_transform = np.asarray(joint['T_FJ'], dtype=float)
                base_point = base_transform[:3, 3] + base_transform[:3, :3] @ (lever*axis)
                follower_point = follower_transform[:3, 3] + follower_transform[:3, :3] @ (lever*axis)
                base, base_jacobian = self._point(poses, jacobians, joint['base_body'], base_point)
                follower, follower_jacobian = self._point(poses, jacobians, joint['follower_body'], follower_point)
                residuals.append(base-follower)
                derivatives.append(base_jacobian-follower_jacobian)
        return np.concatenate(residuals), np.vstack(derivatives)

    def source_coordinates(self, near=None):
        """Recover all 32 source joint coordinates; unwrap cut angles near a reference.

        Full physical closure must be checked separately.  Cut-coordinate
        recovery from an open tree state alone does not certify closure.
        """
        reference = self.reference if near is None else near
        if not isinstance(reference, Mapping):
            raise ValueError("Angle reference must be a source-coordinate dictionary")
        result = {jid: float(self.q[index]) for jid, index in self.q_indices.items()}
        poses = self.body_poses()
        for jid in self.cut_ids:
            joint = self._joints[jid]
            base = poses[joint['base_body']] @ np.asarray(joint['T_BJ'])
            follower = poses[joint['follower_body']] @ np.asarray(joint['T_FJ'])
            rotation = base[:3, :3].T @ follower[:3, :3]
            axial = np.array([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0],
                              rotation[1, 0]-rotation[0, 1]])/2.
            sine = float(np.dot(np.asarray(joint['axis']), axial))
            cosine = float((np.trace(rotation)-1.)/2.)
            value = float(np.arctan2(sine, cosine))
            target = float(reference.get(jid, self.reference[jid]))
            if not np.isfinite(target):
                raise ValueError("Angle reference must be finite")
            result[jid] = value + 2.*np.pi*round((target-value)/(2.*np.pi))
        return result

    def cylinder_measurements(self):
        """Distances between source attachment-frame origins (not hydraulic stroke)."""
        poses = self.body_poses()
        distances = {}
        for record in self._cylinders:
            points = []
            for endpoint in record['endpoints']:
                pose = poses[endpoint['body_id']]
                points.append(pose[:3, 3] + pose[:3, :3] @ np.asarray(endpoint['point_in_body_m']))
            distances[record['source_joint_id']] = float(np.linalg.norm(points[1]-points[0]))
        return distances

    def cylinder_jacobians(self):
        """Analytic attachment-distance derivatives, ordered by native tree q."""
        poses, jacobians = self.body_poses(), self.body_jacobians()
        result = {}
        for record in self._cylinders:
            values = [self._point(poses, jacobians, endpoint['body_id'],
                                  np.asarray(endpoint['point_in_body_m']))
                      for endpoint in record['endpoints']]
            delta = values[1][0]-values[0][0]
            distance = np.linalg.norm(delta)
            if distance <= np.finfo(float).eps:
                raise ValueError("Undefined derivative for coincident attachment origins")
            result[record['source_joint_id']] = delta @ (values[1][1]-values[0][1])/distance
        return result

    def _validate_reference(self):
        residual, _ = self.closure()
        if np.max(np.abs(residual), initial=0.) > 1e-8:
            raise ValueError("CMG reference does not satisfy physical cut-joint closure")
        recovered = self.source_coordinates()
        poses = self.body_poses()
        for jid, joint in self._joints.items():
            base = poses[joint['base_body']] @ np.asarray(joint['T_BJ'])
            follower = poses[joint['follower_body']] @ np.asarray(joint['T_FJ'])
            relative = _inverse(base) @ follower
            expected = np.eye(4)
            axis = np.asarray(joint['axis'], dtype=float)
            if joint['type'] == 'revolute':
                skew = _skew(axis)
                q = self.reference[jid]
                expected[:3, :3] = np.eye(3)+np.sin(q)*skew+(1.-np.cos(q))*(skew@skew)
            elif joint['type'] == 'prismatic':
                expected[:3, 3] = axis*self.reference[jid]
            if (np.linalg.norm(relative[:3, 3]-expected[:3, 3]) > 1e-8
                    or _rotation_error(expected[:3, :3].T @ relative[:3, :3]) > 1e-8):
                raise ValueError(f"CMG reference does not close source joint {jid}")
            if joint['type'] != 'fixed' and abs(recovered[jid]-self.reference[jid]) > 1e-8:
                raise ValueError(f"Reference source-coordinate disagreement at {jid}")

    def _metadata(self):
        return {
            'schema': 'excavator.pinocchio.mapping/0.5',
            'cmg_id': self.cmg['id'], 'pinocchio_version': pin.__version__,
            'native_configuration': 'Source tree-joint coordinates in SI; no nominal offsets',
            'native_velocity': 'Time derivatives of source tree-joint coordinates',
            'tree_coordinate_ids': list(self.tree_ids),
            'q_indices': dict(self.q_indices), 'v_indices': dict(self.v_indices),
            'independent_coordinate_ids': list(self.independent_ids),
            'passive_tree_coordinate_ids': list(self.passive_ids),
            'cut_joint_ids': list(self.cut_ids),
            'tree_edges': deepcopy(self._tree_records),
            'body_frame_ids': dict(self.body_frame_ids),
            'body_support_joint_indices': dict(self._supports),
            'T_joint_body': {body: placement.tolist() for body, placement in self._placements.items()},
            'inertial_records': deepcopy(self._inertial_records),
            'inertia_convention': 'Source body COM inertia; appendBodyToJoint transforms and sums at the nearest supporting joint, including universe for grounded bodies',
            'closure_convention': 'Two world-space axis-point differences per cut revolute; sorted cut id, origin then offset point, xyz rows; redundant rows retained',
            'axis_lever_m': self.axis_lever_m,
            'axis_lever_status': 'Numerical closure lever, not measured part geometry',
            'body_jacobian_convention': 'Linear then angular velocity; world-aligned axes at body origin',
            'cylinder_measurements': deepcopy(self._cylinders),
            'cylinder_measurement_semantics': 'Source revolute attachment-frame origin distance; not certified pin-centre distance or hydraulic extension',
            'source_counts': deepcopy(self.cmg['source_counts']),
            'physical_body_records': [deepcopy(self._bodies[bid]) for bid in sorted(self._bodies)],
            'physical_joint_records': [deepcopy(self._joints[jid]) for jid in sorted(self._joints)],
            'source_actuators': deepcopy(self.cmg['actuators']),
            'source_command_map': deepcopy(self.cmg['command_map']),
            'source_internal_mode': deepcopy(self.cmg['internal_mode']),
            'source_findings': deepcopy(self.cmg['source_findings']),
            'unconnected_effort_joint_ids': deepcopy(self.cmg['command_map']['unconnected_effort_joint_ids']),
            'unknown_p0_force_preserved': 'p0' in self.cmg['command_map']['unconnected_effort_joint_ids'],
            'scope': 'Local kinematics only; no force value, constrained dynamics, hydraulic calibration, or hardware limits inferred'
        }
