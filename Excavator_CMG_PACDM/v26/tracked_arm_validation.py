"""Reproduce controller snapshot parity and a one-second native hydraulic hold.

Run ``python tracked_arm_validation.py`` from any working directory.  The
snapshot test deliberately sets prescribed states on a separate robot-only
oracle, then verifies that controller evaluation does not change those states.
The hold test performs native integration with no state overwrite after its
initialization.  These component checks do not certify the mobile mission.
"""
from __future__ import annotations

from project import ROOT, SOURCE, RESULTS, save_json
import argparse
import hashlib
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from benchmark_model import context, PORTS
from digging_export import export_scene
from digging_path import make_path, reference_at
from hydraulic_actuators import HydraulicBank
from tracked_arm import TrackedArmController


def validate(output=None):
    start = time.perf_counter()
    checks = []

    def gate(name, value, maximum):
        passed = bool(np.isfinite(value) and value <= maximum)
        checks.append(dict(name=name, value=float(value), limit=float(maximum), passed=passed))

    cmg, mapping, e, a, r, inverse = context()
    path = make_path(e, r)
    xml = RESULTS / "assets" / "tracked_arm_component_oracle.xml"
    # This exporter changes no baseline file.  Recreate the oracle explicitly
    # rather than relying on an already-generated result XML in the source ZIP.
    export_scene(xml, soil=False, dt=0.0005)
    m = mujoco.MjModel.from_xml_path(str(xml))
    joints = np.array([m.joint(name).id for name in e.tree_ids])
    qi = m.jnt_qposadr[joints]
    vi = m.jnt_dofadr[joints]
    d = mujoco.MjData(m)
    d.qpos[qi] = path["closed_tree"][0]
    mujoco.mj_forward(m, d)
    ctrl = TrackedArmController(cmg, mapping, e, a, r, inverse,
                                path["closed_tree"][0], m, d)
    snapshots = []
    mutation_count = 0
    for index, trajectory_time in enumerate(np.linspace(0.0, 14.0, 12)):
        independent = reference_at(trajectory_time, path)[0]
        q = r.reconstruct(independent)
        independent_velocity = 0.08 * np.sin(np.arange(7) + index)
        component = r.component(q, independent_velocity, m.opt.gravity)
        rotation = Rotation.from_rotvec([
            0.1 * np.sin(index), -0.15 * np.cos(index), 0.2 * index]).as_matrix()
        native_rotation = rotation @ ctrl.T0[:3, :3]
        # Includes the mobile mission's -0.5 m source-world approach offset.
        virtual_position = np.array([-0.5 + 0.03 * index, -0.3, 1.0])
        d.qpos[:3] = virtual_position + rotation @ ctrl.T0[:3, 3]
        d.qpos[3:7] = Rotation.from_matrix(native_rotation).as_quat()[[3, 0, 1, 2]]
        d.qpos[qi] = q
        d.qvel[:6] = 0.3 * np.sin(np.arange(6) + index)
        d.qvel[vi] = component["velocity"]
        mujoco.mj_forward(m, d)
        base_acceleration = 0.5 * np.cos(np.arange(6) + index)
        # Prescribed snapshot input only.  No such assignment occurs in hold
        # integration or TrackedArmController.evaluate.
        d.qacc[:6] = base_acceleration
        before = [arr.copy() for arr in (d.qpos, d.qvel, d.qacc, d.ctrl)]
        command = ctrl.evaluate(independent, np.zeros(7), np.zeros(7), d)
        mutation_count += sum(not np.array_equal(x, y) for x, y in zip(
            before, (d.qpos, d.qvel, d.qacc, d.ctrl)))
        full_mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, full_mass, d.qM)
        native_mass = full_mass[np.ix_(vi, vi)]
        native_bias = d.qfrc_bias[vi] + full_mass[np.ix_(vi, np.arange(6))] @ base_acceleration
        mass_error = float(np.max(np.abs(ctrl.last_component["mass_matrix"] - native_mass))
                           / max(1.0, np.max(np.abs(native_mass))))
        bias_error = float(np.max(np.abs(ctrl.last_component["bias_forces"] - native_bias))
                           / max(1.0, np.max(np.abs(native_bias))))
        pose_error = 0.0
        for body in cmg["bodies"]:
            if body["kind"] != "rigid_body":
                continue
            source_pose = ctrl.last_floating["poses"][body["id"]]
            body_id = m.body(body["id"]).id
            pose_error = max(pose_error,
                             float(np.max(np.abs(source_pose[:3, 3] - d.xpos[body_id]))),
                             float(np.max(np.abs(source_pose[:3, :3] - d.xmat[body_id].reshape(3, 3)))))
        position_error = float(np.max(np.abs(ctrl.last_base_state["position"] - virtual_position)))
        snapshots.append(dict(index=index, trajectory_time_s=float(trajectory_time),
                              virtual_position_m=virtual_position.tolist(),
                              mass_relative_error=mass_error,
                              effective_bias_relative_error=bias_error,
                              source_body_pose_absolute_error=pose_error,
                              virtual_position_error_m=position_error,
                              requested_p0_force_N=float(command[0])))
    for metric, limit in [("mass_relative_error", 1e-8),
                          ("effective_bias_relative_error", 1e-8),
                          ("source_body_pose_absolute_error", 1e-8),
                          ("virtual_position_error_m", 1e-10),
                          ("requested_p0_force_N", 0.0)]:
        gate("twelve snapshots: " + metric, max(abs(s[metric]) for s in snapshots), limit)
    gate("controller leaves native snapshot qpos/qvel/qacc/ctrl unchanged", mutation_count, 0)
    snapshot_diagnostics = ctrl.diagnostics()
    gate("snapshot PACDM full closure", snapshot_diagnostics["pacdm_closure_max"], 1e-8)
    gate("snapshot PACDM tangent", snapshot_diagnostics["pacdm_tangent_max"], 1e-8)
    gate("snapshot PACDM and all-row maps agree", snapshot_diagnostics["pacdm_point_tangent_difference_max"], 1e-8)
    gate("snapshot inverse projected equilibrium", snapshot_diagnostics["inverse_equilibrium_relative_max"], 1e-9)

    # A separate native data object prevents prescribed snapshot states from
    # leaking into the dynamic experiment.
    d = mujoco.MjData(m)
    d.qpos[qi] = path["closed_tree"][0]
    mujoco.mj_forward(m, d)
    hold = TrackedArmController(cmg, mapping, e, a, r, inverse,
                                path["closed_tree"][0], m, d)
    desired = path["independent"][0]
    command = hold.evaluate(desired, np.zeros(7), np.zeros(7), d)
    ports = np.array([e.tree_ids.index(name) for name in PORTS])
    bank = HydraulicBank(r.nominal[ports], command)
    dt = float(m.opt.timestep)
    step_count = round(1.0 / dt)
    control_every = round(0.01 / dt)
    max_error = max_loop = max_power_identity = 0.0
    mutation_count = 0
    for step in range(step_count):
        if step % control_every == 0:
            before = [arr.copy() for arr in (d.qpos, d.qvel, d.qacc, d.ctrl)]
            command = hold.evaluate(desired, np.zeros(7), np.zeros(7), d)
            mutation_count += sum(not np.array_equal(x, y) for x, y in zip(
                before, (d.qpos, d.qvel, d.qacc, d.ctrl)))
        item = bank.evaluate(d.qpos[qi][ports], d.qvel[vi][ports], command, dt)
        d.ctrl[:] = item["effort"]
        max_power_identity = max(max_power_identity, abs(item["fluid_identity_W"]))
        mujoco.mj_step(m, d)
        bank.advance(item, dt)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise ValueError("Nonfinite state during native hydraulic hold")
        max_error = max(max_error, float(np.max(np.abs(d.qpos[qi][e.active[:6]] - desired[:6]))))
        equality = d.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        max_loop = max(max_loop, float(np.max(np.abs(d.efc_pos[equality]), initial=0.0)))
    warnings = sum(int(entry.number) for entry in d.warning)
    hold_diagnostics = hold.diagnostics()
    gate("one-second hold completes", abs(float(d.time) - 1.0), 1e-10)
    gate("hold native warning count", warnings, 0)
    gate("hold peak controlled-angle error rad", max_error, 0.03)
    gate("hold peak native loop gap m", max_loop, 1e-4)
    gate("hold controller never modifies native state", mutation_count, 0)
    gate("hold hydraulic instantaneous fluid identity W", max_power_identity, 1e-6)
    gate("hold PACDM full closure", hold_diagnostics["pacdm_closure_max"], 1e-8)
    gate("hold PACDM tangent", hold_diagnostics["pacdm_tangent_max"], 1e-8)
    gate("hold PACDM and all-row maps agree", hold_diagnostics["pacdm_point_tangent_difference_max"], 1e-8)
    gate("hold inverse projected equilibrium", hold_diagnostics["inverse_equilibrium_relative_max"], 1e-9)
    gate("hold requested p0 force zero", hold.command_peak[0], 0)
    gate("q22 has no actuator port", int("q22" in PORTS), 0)
    report = dict(
        passed=all(check["passed"] for check in checks), checks=checks,
        scope="Controller component validation; robot-only floating oracle and native hydraulic hold, not mobile mission acceptance",
        snapshot_count=len(snapshots), snapshots=snapshots,
        snapshot_controller=snapshot_diagnostics,
        hold=dict(duration_s=float(d.time), timestep_s=dt, integration_steps=step_count,
                  controller_period_s=control_every * dt,
                  peak_controlled_angle_error_rad=max_error, peak_native_loop_gap_m=max_loop,
                  warning_count=warnings, native_state_projection=False,
                  controller=hold_diagnostics),
        software=dict(mujoco=mujoco.__version__, numpy=np.__version__),
        source_sha256={name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                       for name in ["data/accepted_cmg_v04.json", "excavator_pacdm.py", "pacdm.py"]},
        controller_sha256=hashlib.sha256((ROOT / "tracked_arm.py").read_bytes()).hexdigest(),
        elapsed_s=time.perf_counter() - start)
    output = RESULTS / "tracked_arm_component_validation.json" if output is None else output
    save_json(output, report)
    print(f"Tracked arm components: {sum(c['passed'] for c in checks)}/{len(checks)} gates; report={output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()
    result = validate(arguments.output)
    raise SystemExit(0 if result["passed"] else 1)
