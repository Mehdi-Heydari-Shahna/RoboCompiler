"""Online PACDM arm control with a measured floating undercarriage.

The accepted PACDM files are imported unchanged.  Assembly happens in a shadow
configuration only: this module never writes MuJoCo qpos, qvel, qacc or ctrl.
The native plant remains responsible for physical loop/contact enforcement.

FloatingSource uses a virtual free body at the original source-world origin.
MuJoCo's actual free joint is at body_53.  The offset, angular-velocity and
centripetal-acceleration transformations below are therefore necessary; copying
the native six base numbers into the source oracle would be incorrect.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import mujoco
import numpy as np

_SOURCE = Path(__file__).resolve().parent / "source" / "Excavator_RoboIR_full_body_v21"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from excavator_pacdm import compile_pacdm
from floating_dynamics import FloatingSource


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result.copy()


class TrackedArmController:
    """Six commanded independent accelerations plus the dynamic passive pin.

    ``evaluate(desired7, velocity7, acceleration7, d)`` returns the eight
    physical arm-port efforts.  The seventh reference component is recorded
    but never servoed: q22 remains passive.  p0's requested effort is explicitly
    zero.  Hydraulic limits/dynamics belong to the caller's HydraulicBank.

    The source-body dynamics condition arm effort on the latest measured base
    acceleration, rather than pretending the supporting base is fixed.  The
    optional acceleration filter is off by default and, if enabled, is reported
    explicitly; it affects the controller, never the plant or force audit.
    """

    def __init__(self, cmg, mapping, e, a, r, inverse, initial_closed_tree, m, d,
                 *, kp=144.0, kd=24.0, base_acceleration_filter_s=0.0):
        self.e, self.a, self.r, self.inverse, self.m = e, a, r, inverse, m
        self.graph, self.pacdm = compile_pacdm(cmg, mapping, e.tree_ids)
        self.floating = FloatingSource(cmg, e.source)
        self.active = np.asarray(e.active, dtype=int)
        jids = np.array([m.joint(name).id for name in e.tree_ids], dtype=int)
        self.qi = m.jnt_qposadr[jids].copy()
        self.vi = m.jnt_dofadr[jids].copy()
        self.base_id = m.body("body_53").id
        self.base_joint = int(m.body_jntadr[self.base_id])
        if (m.body_jntnum[self.base_id] != 1 or
                m.jnt_type[self.base_joint] != int(mujoco.mjtJoint.mjJNT_FREE)):
            raise ValueError("body_53 must have one unconstrained floating free joint")
        self.base_qi = int(m.jnt_qposadr[self.base_joint])
        self.base_vi = int(m.jnt_dofadr[self.base_joint])
        record = next(j for j in cmg["joints"] if j["id"] == "fixed_world_base")
        self.T0 = np.asarray(record["T_BJ"]) @ np.linalg.inv(np.asarray(record["T_FJ"]))
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (6,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (6,)).copy()
        self.acceleration_filter_s = float(base_acceleration_filter_s)
        if (not np.isfinite(self.kp).all() or not np.isfinite(self.kd).all() or
                np.any(self.kp < 0) or np.any(self.kd < 0) or
                not np.isfinite(self.acceleration_filter_s) or self.acceleration_filter_s < 0):
            raise ValueError("Controller gains and filter time constant must be finite and nonnegative")
        self.seed = self.graph.lift(_vector(initial_closed_tree, 23, "Initial closed tree"))
        self.N, self.mapping = self.pacdm.mapping(self.seed)
        self.initial_acquisition = None
        if self.N is None:
            self.seed, self.initial_acquisition = self.pacdm.acquire(
                self.seed[self.graph.active].copy(), self.seed)
            if not self.initial_acquisition["success"]:
                raise ValueError(f"Initial PACDM assembly failed: {self.initial_acquisition}")
            self.N, self.mapping = self.pacdm.mapping(self.seed)
        if self.N is None:
            raise ValueError(f"Initial PACDM mapping failed: {self.mapping}")
        self.seed = self._polish(self.seed, np.asarray(self.mapping["rows"], dtype=int))
        self.N, self.mapping = self.pacdm.mapping(self.seed)
        self.command_peak = np.zeros(8)
        self.shadow_max = 0.0
        self.shadow_velocity_max = 0.0
        self.closure_max = 0.0
        self.tangent_max = 0.0
        self.mapping_difference_max = 0.0
        self.inverse_equilibrium_max = 0.0
        self.acceleration_closure_max = 0.0
        self.base_coupling_force_max = 0.0
        self.base_acceleration_peak = np.zeros(6)
        self.evaluations = self.continuation_steps = self.fallback_count = 0
        self.control_elapsed_s = self.control_max_elapsed_s = 0.0
        self.last_solution = self.last_component = self.last_floating = None
        self.last_base_state = None
        self._filtered_acceleration = None
        self._last_sample_time = None

    def _polish(self, q, rows):
        """Original accepted 5e-13 precision after PACDM physical correction."""
        q = q.copy()
        for _ in range(6):
            residual, jacobian, _ = self.graph.residual(q)
            if np.max(np.abs(residual)) < 5e-13:
                break
            q[self.graph.passive] -= np.linalg.solve(
                jacobian[np.ix_(rows, self.graph.passive)], residual[rows])
        if np.any(q < self.graph.lower) or np.any(q > self.graph.upper):
            raise ValueError("PACDM polish left the original numerical branch guards")
        return q

    def _assemble(self, measured_independent):
        # Bounded continuation also covers deliberate large reference/state
        # snapshots without substituting a different closure solver.
        start = self.seed[self.graph.active].copy()
        count = max(1, int(np.ceil(np.max(np.abs(measured_independent - start)) / 0.02)))
        for index in range(1, count + 1):
            target = measured_independent if index == count else start + (
                measured_independent - start) * (index / count)
            rows = np.asarray(self.mapping["rows"], dtype=int)
            predicted = self.seed[self.graph.passive] + self.N[self.graph.passive] @ (
                target - self.seed[self.graph.active])
            candidate, info = self.pacdm.correct(target, predicted, None, rows, maxiter=12)
            if not info["success"]:
                self.fallback_count += 1
                candidate, info = self.pacdm.acquire(target, self.seed)
            if not info["success"]:
                raise ValueError(f"Online PACDM closure failed: {info}")
            candidate = self._polish(candidate, rows)
            tangent, mapping = self.pacdm.mapping(candidate)
            if tangent is None:
                raise ValueError(f"Online PACDM rank/full-closure gate failed: {mapping}")
            self.seed, self.N, self.mapping = candidate, tangent, mapping
            self.closure_max = max(self.closure_max, float(mapping["residual_inf"]))
            self.tangent_max = max(self.tangent_max, float(mapping["tangent_residual"]))
            self.continuation_steps += 1
        closed = self.seed[:23].copy()
        self.r._check_guards(closed)
        return closed

    def base_state(self, d):
        """Read native root state and transform it to virtual-world coordinates.

        Native freejoint linear velocity is world-aligned; angular velocity and
        angular acceleration are body-aligned.  Source angular quantities are
        world-aligned.  qpos is used directly so this method does not depend on
        stale body-placement caches following an mj_step call.
        """
        qi, vi = self.base_qi, self.base_vi
        position = _vector(d.qpos[qi:qi + 3], 3, "Native base position")
        quaternion = _vector(d.qpos[qi + 3:qi + 7], 4, "Native base quaternion")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-12:
            raise ValueError("Invalid native freejoint quaternion")
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, quaternion / norm)
        native_rotation = matrix.reshape(3, 3)
        rotation = native_rotation @ self.T0[:3, :3].T
        offset = rotation @ self.T0[:3, 3]
        native_velocity = _vector(d.qvel[vi:vi + 6], 6, "Native base velocity")
        native_acceleration = _vector(d.qacc[vi:vi + 6], 6, "Native base acceleration")
        omega = native_rotation @ native_velocity[3:]
        alpha = native_rotation @ native_acceleration[3:]
        virtual_velocity = np.r_[native_velocity[:3] - np.cross(omega, offset), omega]
        virtual_acceleration = np.r_[native_acceleration[:3] - np.cross(alpha, offset)
                                     - np.cross(omega, np.cross(omega, offset)), alpha]
        return dict(position=position - offset, rotation=rotation,
                    velocity=virtual_velocity, acceleration=virtual_acceleration,
                    native_position=position, native_rotation=native_rotation,
                    native_velocity=native_velocity, native_acceleration=native_acceleration)

    def evaluate(self, desired7, velocity7, acceleration7, d):
        started = time.perf_counter()
        desired = _vector(desired7, 7, "Desired independent position")
        desired_velocity = _vector(velocity7, 7, "Desired independent velocity")
        desired_acceleration = _vector(acceleration7, 7, "Desired independent acceleration")
        native_q = d.qpos[self.qi].copy()
        native_v = d.qvel[self.vi].copy()
        independent = native_q[self.active]
        independent_velocity = native_v[self.active]
        closed = self._assemble(independent)
        gravity = np.asarray(self.m.opt.gravity).copy()
        component = self.r.component(closed, independent_velocity, gravity)
        difference = float(np.max(np.abs(self.N[:23] - component["tangent_map"])))
        if difference > 1e-8:
            raise ValueError("PACDM tangent and all-row differential map disagree")
        self.mapping_difference_max = max(self.mapping_difference_max, difference)
        base = self.base_state(d)
        acceleration = base["acceleration"].copy()
        sample_time = float(d.time)
        if self.acceleration_filter_s > 0:
            if self._filtered_acceleration is None:
                self._filtered_acceleration = acceleration.copy()
            elif self._last_sample_time is not None and sample_time > self._last_sample_time:
                weight = -np.expm1(-(sample_time - self._last_sample_time) / self.acceleration_filter_s)
                self._filtered_acceleration += weight * (acceleration - self._filtered_acceleration)
            acceleration = self._filtered_acceleration.copy()
        self._last_sample_time = sample_time
        base["used_acceleration"] = acceleration.copy()
        floating = self.floating.evaluate(closed, base["position"], base["rotation"],
                                         np.r_[base["velocity"], component["velocity"]], gravity)
        # Full dynamic arm row: Mqq*qdd + hq + Mqb*base_acceleration = arm_effort.
        # Added chassis/track bodies are attached below body_53 and have zero
        # Jacobian columns for all arm joints, so they add no arm-row terms.
        coupling = floating["mass_matrix"][6:, :6] @ acceleration
        component["mass_matrix"] = floating["mass_matrix"][6:, 6:].copy()
        component["bias_forces"] = floating["bias_forces"][6:].copy() + coupling
        requested = (desired_acceleration[:6] + self.kp * (desired[:6] - independent[:6])
                     + self.kd * (desired_velocity[:6] - independent_velocity[:6]))
        solution = self.inverse.solve(component, requested, p0_force=0.0)
        force = solution["physical_efforts"].copy()
        self.command_peak = np.maximum(self.command_peak, np.abs(force))
        self.shadow_max = max(self.shadow_max, float(np.max(np.abs(native_q - closed))))
        self.shadow_velocity_max = max(self.shadow_velocity_max, float(np.max(
            np.abs(native_v - component["velocity"]))))
        self.base_coupling_force_max = max(self.base_coupling_force_max, float(np.max(np.abs(coupling))))
        self.base_acceleration_peak = np.maximum(self.base_acceleration_peak, np.abs(base["acceleration"]))
        self.inverse_equilibrium_max = max(self.inverse_equilibrium_max, float(
            solution["diagnostics"]["reduced_equilibrium_normalized_residual"]))
        self.acceleration_closure_max = max(self.acceleration_closure_max, float(
            solution["diagnostics"]["closure_acceleration_max_abs"]))
        self.last_solution, self.last_component = solution, component
        self.last_floating, self.last_base_state = floating, base
        self.evaluations += 1
        elapsed = time.perf_counter() - started
        self.control_elapsed_s += elapsed
        self.control_max_elapsed_s = max(self.control_max_elapsed_s, elapsed)
        return force

    def diagnostics(self):
        """Small JSON-ready controller evidence; timing is observed wall time."""
        return dict(method="online PACDM prediction/correction with accepted 5e-13 polish",
                    evaluations=self.evaluations, continuation_steps=self.continuation_steps,
                    fallback_count=self.fallback_count, pacdm_closure_max=self.closure_max,
                    pacdm_tangent_max=self.tangent_max,
                    pacdm_point_tangent_difference_max=self.mapping_difference_max,
                    shadow_position_max_SI=self.shadow_max,
                    shadow_velocity_max_SI=self.shadow_velocity_max,
                    inverse_equilibrium_relative_max=self.inverse_equilibrium_max,
                    inverse_acceleration_closure_max=self.acceleration_closure_max,
                    base_acceleration_peak_SI=self.base_acceleration_peak.tolist(),
                    base_acceleration_coupling_peak_SI=self.base_coupling_force_max,
                    command_peak_SI=self.command_peak.tolist(),
                    measured_control_mean_wall_s=self.control_elapsed_s / max(1, self.evaluations),
                    measured_control_max_wall_s=self.control_max_elapsed_s,
                    base_acceleration_filter_s=self.acceleration_filter_s,
                    native_state_projection=False, q22_actuated=False, p0_requested_force_N=0.0,
                    floating_base_feedforward=True,
                    floating_oracle_scope="Original 25 source bodies; arm-row dynamics conditioned on measured base motion",
                    real_time_deadline_certified=False)
