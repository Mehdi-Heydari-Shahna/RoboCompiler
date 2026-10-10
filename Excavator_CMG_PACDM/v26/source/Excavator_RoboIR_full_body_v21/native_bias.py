"""Native rigid-body bias and inverse dynamics for the source-coordinate tree.

Pinocchio 3.8.0 and MuJoCo 3.3.7 operate on separate generated models.  The
accepted v0.6 mass/energy/gravity adapter is retained without modification.
These are instantaneous rigid-body evaluations, with no constraint-force solve,
time integration, actuator allocation, friction, or inference of the p0 effort.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pinocchio as pin

from native_dynamics import NativeDynamics


def _skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class NativeBiasDynamics(NativeDynamics):
    """Return forces and analytic closure derivatives in source tree order.

    Force convention: ``M(q) a + coriolis_forces + gravity_compensation``.
    ``coriolis_forces`` includes centrifugal effects.  It is evaluated at zero
    gravity by the native recursion, rather than subtracting two large loads.
    Every input is finite, real, and in SI units. Instances own mutable data.
    """

    def _prepare_mujoco(self, q, v, gravity):
        model, data = self.mj_model, self.mj_data
        model.opt.gravity[:] = gravity
        data.qpos[self._mj_q_indices] = self._mj_sign*q + self._mj_offset
        data.qvel[:] = self._velocity_map @ v
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)

    def evaluate(self, qtree, velocity, gravity3):
        """Extend v0.6 components with actual native velocity bias and full bias."""
        n = len(self.tree_ids)
        q = self._vector(qtree, n, 'Tree configuration')
        v = self._vector(velocity, n, 'Tree velocity')
        gravity = self._vector(gravity3, 3, 'World gravity')
        result = super().evaluate(q, v, gravity)
        pq, pv = q[self._compact_source_indices], v[self._compact_source_indices]
        pmodel, pdata = self.pin_model, self.pin_data
        # Copy each returned view before any following native recursion.
        pmodel.gravity = pin.Motion.Zero()
        pin_c = np.asarray(pin.nonLinearEffects(pmodel, pdata, pq, pv))[
            self._source_compact_indices].copy()
        pmodel.gravity.linear = gravity
        pin_h = np.asarray(pin.nonLinearEffects(pmodel, pdata, pq, pv))[
            self._source_compact_indices].copy()

        self._prepare_mujoco(q, v, np.zeros(3))
        native_force = np.empty(n)
        mujoco.mj_rne(self.mj_model, self.mj_data, 0, native_force)
        mj_c = self._velocity_map.T @ native_force
        self.mj_model.opt.gravity[:] = gravity
        mujoco.mj_rne(self.mj_model, self.mj_data, 0, native_force)
        mj_h = self._velocity_map.T @ native_force
        result['pin'].update(coriolis_forces=pin_c, bias_forces=pin_h)
        result['mujoco'].update(coriolis_forces=mj_c, bias_forces=mj_h)
        if any(not np.all(np.isfinite(value)) for values in result.values()
               for value in values.values()):
            raise ValueError('Native bias evaluation produced a nonfinite result')
        return result

    def inverse_dynamics(self, qtree, velocity, acceleration, gravity3):
        """Native RNEA forces for prescribed tree acceleration, no loop reactions.

        This deliberately calls native inverse dynamics instead of constructing
        ``M @ acceleration + bias``.  The prescribed acceleration may be an open
        tree test state; the caller must separately enforce physical loop closure
        before interpreting it as a closed-mechanism state.
        """
        n = len(self.tree_ids)
        q = self._vector(qtree, n, 'Tree configuration')
        v = self._vector(velocity, n, 'Tree velocity')
        a = self._vector(acceleration, n, 'Tree acceleration')
        gravity = self._vector(gravity3, 3, 'World gravity')
        self._audit_options()
        self.pin_model.gravity = pin.Motion.Zero()
        self.pin_model.gravity.linear = gravity
        compact_force = pin.rnea(self.pin_model, self.pin_data,
                                q[self._compact_source_indices],
                                v[self._compact_source_indices],
                                a[self._compact_source_indices])
        pin_force = np.asarray(compact_force)[self._source_compact_indices].copy()
        self._prepare_mujoco(q, v, gravity)
        self.mj_data.qacc[:] = self._velocity_map @ a
        native_force = np.empty(n)
        mujoco.mj_rne(self.mj_model, self.mj_data, 1, native_force)
        mj_force = self._velocity_map.T @ native_force
        if not (np.all(np.isfinite(pin_force)) and np.all(np.isfinite(mj_force))):
            raise ValueError('Native inverse dynamics produced a nonfinite result')
        return {'pin': pin_force, 'mujoco': mj_force}

    def constraint_kinematics(self, qtree, velocity):
        """Analytic closure residual, J, dJ/dt, and (dJ/dt)v in each backend.

        Rows follow the accepted adapter: sorted cut joint, origin/axis point,
        then world xyz; columns use ``tree_ids``.  The 54 rows are intentionally
        redundant.  MuJoCo point derivatives come from ``mj_jacDot`` after its
        documented kinematics/comPos/comVel pipeline. Pinocchio frame derivatives
        include the derivative of the rotated point offset. No finite difference
        or dynamics solve is used in either native derivative implementation.
        """
        n = len(self.tree_ids)
        q = self._vector(qtree, n, 'Tree configuration')
        v = self._vector(velocity, n, 'Tree velocity')
        backend = self.pin_backend
        backend.set_configuration(q)
        pin.computeJointJacobiansTimeVariation(backend.model, backend.data, q, v)
        pin.updateFramePlacements(backend.model, backend.data)
        poses = backend.body_poses()
        body_jac = backend.body_jacobians()
        body_jacdot = {
            body: np.asarray(pin.getFrameJacobianTimeVariation(
                backend.model, backend.data, fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).copy()
            for body, fid in backend.body_frame_ids.items()}
        self._prepare_mujoco(q, v, self.mj_model.opt.gravity.copy())
        results = {'pin': [[], [], []], 'mujoco': [[], [], []]}

        def point_values(body, local):
            transform = poses[body]
            displacement = transform[:3, :3] @ local
            p = transform[:3, 3] + displacement
            j, dj = body_jac[body], body_jacdot[body]
            omega = j[3:] @ v
            displacement_dot = np.cross(omega, displacement)
            jp = j[:3] - _skew(displacement) @ j[3:]
            djp = (dj[:3] - _skew(displacement_dot) @ j[3:]
                   - _skew(displacement) @ dj[3:])
            body_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_index < 0:
                raise ValueError(f'Missing MuJoCo physical body {body}')
            mp = (self.mj_data.xpos[body_index]
                  + self.mj_data.xmat[body_index].reshape(3, 3) @ local)
            mj_j = np.zeros((3, n))
            mj_dj = np.zeros((3, n))
            mujoco.mj_jac(self.mj_model, self.mj_data, mj_j, None, mp, body_index)
            mujoco.mj_jacDot(self.mj_model, self.mj_data, mj_dj, None, mp, body_index)
            return {'pin': (p, jp, djp),
                    'mujoco': (mp.copy(), mj_j @ self._velocity_map,
                               mj_dj @ self._velocity_map)}

        for jid in backend.cut_ids:
            joint = backend._joints[jid]
            axis = np.asarray(joint['axis'], dtype=float)
            for lever in (0., backend.axis_lever_m):
                endpoints = []
                for body_key, transform_key in (('base_body', 'T_BJ'),
                                                 ('follower_body', 'T_FJ')):
                    transform = np.asarray(joint[transform_key], dtype=float)
                    local = transform[:3, 3] + transform[:3, :3] @ (lever*axis)
                    endpoints.append(point_values(joint[body_key], local))
                for engine in results:
                    for component in range(3):
                        results[engine][component].append(
                            endpoints[0][engine][component]-endpoints[1][engine][component])
        output = {}
        for engine, (residuals, jacobians, derivatives) in results.items():
            jacobian_dot = np.vstack(derivatives)
            output[engine] = {'residual': np.concatenate(residuals),
                              'jacobian': np.vstack(jacobians),
                              'jacobian_dot': jacobian_dot,
                              'jdot_velocity': jacobian_dot @ v}
        if any(not np.all(np.isfinite(value)) for values in output.values()
               for value in values.values()):
            raise ValueError('Native closure derivatives produced a nonfinite result')
        return output
