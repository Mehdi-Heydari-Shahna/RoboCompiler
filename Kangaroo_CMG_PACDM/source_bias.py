"""Assemble analytic velocity-dependent terms from physical CMG bodies.

NumPy calculations follow the tree traversal in source_dynamics. Jacobian
time derivatives are analytic along the tree velocity; finite differences
are used for validation.

Arrays follow the scalar tree-coordinate order. Body Jacobians and their
derivatives use world-aligned, linear-first components at the body origin.
The generalized bias is the left-hand-side term h in M qdd + h = tau + J.T lambda."""
from __future__ import annotations

import numpy as np

from source_dynamics import SourceDynamics, _array, _skew, _transform


class SourceBiasDynamics(SourceDynamics):
    """Independent rigid-body M, energy, gravity and analytic velocity bias.

    ``evaluate(q, velocity, gravity)`` extends ``SourceDynamics.evaluate`` with
    ``body_jacobian_dots``, ``body_com_jacobian_dots``, ``coriolis_forces`` and
    ``bias_forces``.  The vector called coriolis_forces includes centrifugal
    terms; no particular Coriolis matrix factorization is implied.

    ``closure(evaluation, cut_ids, axis_lever_m=0.1)`` returns residual, J and
    Jdot for two-point revolute closures in the CMG cut-ID order.
    """

    def evaluate(self, q, velocity, gravity):
        # The inherited implementation independently validates the source
        # records and input shapes and computes all first-order kinematics.
        q = _array(q, (self.nv,), 'q')
        velocity = _array(velocity, (self.nv,), 'velocity')
        gravity = _array(gravity, (3,), 'gravity')
        result = super().evaluate(q, velocity, gravity)
        poses, jacobians = result['poses'], result['jacobians']
        body_velocities = {body: jacobian @ velocity
                           for body, jacobian in jacobians.items()}
        jacobian_dots = {self.root_body: np.zeros((6, self.nv))}

        for parent, child, kind, attach, _, axis, column in self._traversal:
            parent_pose, child_pose = poses[parent], poses[child]
            parent_jacobian = jacobians[parent]
            parent_dot = jacobian_dots[parent]
            parent_velocity, child_velocity = body_velocities[parent], body_velocities[child]
            shift = child_pose[:3, 3] - parent_pose[:3, 3]
            shift_dot = child_velocity[:3] - parent_velocity[:3]

            child_dot = parent_dot.copy()
            child_dot[:3] -= (_skew(shift_dot) @ parent_jacobian[3:]
                             + _skew(shift) @ parent_dot[3:])
            if column is not None:
                joint_pose = parent_pose @ attach
                world_axis = joint_pose[:3, :3] @ axis
                axis_dot = np.cross(parent_velocity[3:], world_axis)
                if kind == 'revolute':
                    joint_offset = joint_pose[:3, 3] - parent_pose[:3, 3]
                    joint_velocity = (parent_velocity[:3]
                                      + np.cross(parent_velocity[3:], joint_offset))
                    lever = child_pose[:3, 3] - joint_pose[:3, 3]
                    lever_dot = child_velocity[:3] - joint_velocity
                    child_dot[:3, column] += (np.cross(axis_dot, lever)
                                             + np.cross(world_axis, lever_dot))
                    child_dot[3:, column] += axis_dot
                elif kind == 'prismatic':
                    child_dot[:3, column] += axis_dot
            jacobian_dots[child] = child_dot

        com_jacobian_dots = {}
        body_bias_accelerations = {}
        body_com_bias_accelerations = {}
        body_coriolis_contributions = {}
        coriolis = np.zeros(self.nv)
        for body_id, (mass, local_com, local_inertia) in self._inertias.items():
            rotation = poses[body_id][:3, :3]
            offset = rotation @ local_com
            omega = body_velocities[body_id][3:]
            offset_dot = np.cross(omega, offset)
            body_jacobian, body_dot = jacobians[body_id], jacobian_dots[body_id]
            com_dot = (body_dot[:3] - _skew(offset_dot) @ body_jacobian[3:]
                       - _skew(offset) @ body_dot[3:])
            com_bias = com_dot @ velocity
            angular_bias = body_dot[3:] @ velocity
            inertia_world = rotation @ local_inertia @ rotation.T
            force = mass * com_bias
            torque = inertia_world @ angular_bias + np.cross(omega, inertia_world @ omega)
            body_term = (result['body_com_jacobians'][body_id].T @ force
                         + body_jacobian[3:].T @ torque)
            coriolis += body_term
            com_jacobian_dots[body_id] = com_dot
            body_bias_accelerations[body_id] = body_dot @ velocity
            body_com_bias_accelerations[body_id] = com_bias
            body_coriolis_contributions[body_id] = body_term

        bias = coriolis + result['gravity_compensation']
        arrays = [coriolis, bias, *jacobian_dots.values(), *com_jacobian_dots.values(),
                  *body_velocities.values(), *body_bias_accelerations.values(),
                  *body_com_bias_accelerations.values(), *body_coriolis_contributions.values()]
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError('Source bias evaluation produced nonfinite values')
        result.update({
            'configuration': q,
            'velocity': velocity,
            'gravity': gravity,
            'body_velocities': body_velocities,
            'body_jacobian_dots': jacobian_dots,
            'body_com_jacobian_dots': com_jacobian_dots,
            'body_bias_accelerations': body_bias_accelerations,
            'body_com_bias_accelerations': body_com_bias_accelerations,
            'body_coriolis_contributions': body_coriolis_contributions,
            'coriolis_forces': coriolis,
            'bias_forces': bias,
        })
        return result

    @staticmethod
    def _point(evaluation, body_id, local_point):
        pose = evaluation['poses'][body_id]
        body_jacobian = evaluation['jacobians'][body_id]
        body_dot = evaluation['body_jacobian_dots'][body_id]
        omega = evaluation['body_velocities'][body_id][3:]
        offset = pose[:3, :3] @ local_point
        offset_dot = np.cross(omega, offset)
        point = pose[:3, 3] + offset
        jacobian = body_jacobian[:3] - _skew(offset) @ body_jacobian[3:]
        derivative = (body_dot[:3] - _skew(offset_dot) @ body_jacobian[3:]
                      - _skew(offset) @ body_dot[3:])
        return point, jacobian, derivative

    def closure(self, evaluation, cut_ids, axis_lever_m=0.1):
        """Evaluate analytic two-point closure kinematics from raw attachments.

        For each revolute joint: three base-minus-follower origin rows followed
        by three base-minus-follower axis-point rows.  The lever is a numerical
        length in metres, not measured component geometry.  Redundant rows are
        retained.  Cut joints must be omitted from this instance's source tree.
        """
        lever = _array(axis_lever_m, (), 'axis_lever_m')
        if float(lever) <= 0.:
            raise ValueError('axis_lever_m must be positive')
        cut_ids = list(cut_ids)
        if any(not isinstance(jid, str) for jid in cut_ids) or len(set(cut_ids)) != len(cut_ids):
            raise ValueError('Cut IDs must be unique strings')
        joints = {joint['id']: joint for joint in self.cmg['joints']}
        residuals, jacobians, derivatives = [], [], []
        for jid in cut_ids:
            if jid not in joints or joints[jid]['type'] != 'revolute' or jid in self.tree_ids:
                raise ValueError('Each cut ID must identify a source revolute joint outside the tree')
            joint = joints[jid]
            axis = _array(joint['axis'], (3,), f'{jid} axis')
            if abs(np.linalg.norm(axis) - 1.) > 1e-10:
                raise ValueError(f'{jid} moving axis must be a unit vector')
            base_transform = _transform(joint['T_BJ'], f'{jid} T_BJ')
            follower_transform = _transform(joint['T_FJ'], f'{jid} T_FJ')
            for distance in (0., float(lever)):
                base_local = base_transform[:3, 3] + base_transform[:3, :3] @ (distance * axis)
                follower_local = (follower_transform[:3, 3]
                                  + follower_transform[:3, :3] @ (distance * axis))
                base, base_jacobian, base_dot = self._point(evaluation, joint['base_body'], base_local)
                follower, follower_jacobian, follower_dot = self._point(
                    evaluation, joint['follower_body'], follower_local)
                residuals.append(base - follower)
                jacobians.append(base_jacobian - follower_jacobian)
                derivatives.append(base_dot - follower_dot)
        if not cut_ids:
            return np.zeros(0), np.zeros((0, self.nv)), np.zeros((0, self.nv))
        result = np.concatenate(residuals), np.vstack(jacobians), np.vstack(derivatives)
        if any(not np.all(np.isfinite(value)) for value in result):
            raise ValueError('Source closure evaluation produced nonfinite values')
        return result
