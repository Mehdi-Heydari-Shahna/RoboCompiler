"""Independent Pinocchio witnesses for the unchanged PACDM Panda model.

No MuJoCo module is imported. Native Pinocchio frame Jacobians and classical
accelerations check PACDM's route, including its acceleration curvature. The
permanent finger equality is checked by a full multiplier system and an
independently written constant physical reduction. These are finite numerical
model tests, not measurements or contact validation.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import platform

import numpy as np
import pinocchio as pin
import scipy
from scipy.spatial.transform import Rotation

from panda.model import PandaGraph
from panda.pin_backend import PinBackend
from panda.task import TaskGraph
from vendor.pacdm_original import PACDM


def _skew(x):
    return np.array([[0., -x[2], x[1]], [x[2], 0., -x[0]],
                     [-x[1], x[0], 0.]])


def _normalized_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual)-expected)) /
                 max(1., float(np.max(np.abs(expected)))))


def _target_kinematics(active, velocity, acceleration):
    """Explicit world geometric derivatives of R0 Rx(rx) Ry(ry) Rz(rz).

    This is independent of TaskGraph's SE(3) residual, adjoints and Jdot.
    Return pose, linear-first twist/acceleration and active-coordinate map.
    """
    rx, ry, rz = np.asarray(active)[3:6]
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                             np.sin(ry), np.cos(rz), np.sin(rz))
    rxm = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    rym = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rzm = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    nominal = np.diag([1., -1., -1.])
    angular_map = np.column_stack((nominal[:, 0], nominal @ rxm[:, 1],
                                   nominal @ rxm @ rym[:, 2]))
    rates = np.asarray(velocity)[3:6]
    rates_dot = np.asarray(acceleration)[3:6]
    omega1 = angular_map[:, 0]*rates[0]
    omega2 = angular_map[:, 1]*rates[1]
    angular_map_dot = np.column_stack((np.zeros(3),
                                      np.cross(omega1, angular_map[:, 1]),
                                      np.cross(omega1+omega2, angular_map[:, 2])))
    pose = np.eye(4)
    pose[:3, :3] = nominal @ rxm @ rym @ rzm
    pose[:3, 3] = np.asarray(active)[:3]
    twist = np.r_[np.asarray(velocity)[:3], angular_map @ rates]
    acc = np.r_[np.asarray(acceleration)[:3],
                angular_map @ rates_dot + angular_map_dot @ rates]
    task_map = np.zeros((6, 8))
    task_map[:3, :3] = np.eye(3)
    task_map[3:, 3:6] = angular_map
    return pose, twist, acc, task_map


def _add_tool_frame(backend, body, transform):
    frame = backend.model.frames[backend.body_frame_ids[body]]
    transform = np.asarray(transform)
    placement = frame.placement * pin.SE3(transform[:3, :3], transform[:3, 3])
    frame_id = backend.model.addFrame(pin.Frame(
        "independent_validation_tool", frame.parentJoint, backend.body_frame_ids[body],
        placement, pin.FrameType.OP_FRAME), False)
    backend.data = backend.model.createData()
    return int(frame_id)


def _tool_jacobian(backend, frame_id, q):
    matrix = pin.computeFrameJacobian(backend.model, backend.data,
                                     backend._native_q(q), frame_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return np.asarray(matrix)[:, backend._v_indices].copy()


def _tool_motion(backend, frame_id, q, v, a):
    pin.forwardKinematics(backend.model, backend.data, backend._native_q(q),
                          backend._native_v(v, "v"), backend._native_v(a, "a"))
    pin.updateFramePlacements(backend.model, backend.data)
    pose = np.asarray(backend.data.oMf[frame_id].homogeneous).copy()
    velocity = pin.getFrameVelocity(backend.model, backend.data, frame_id,
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    acceleration = pin.getFrameClassicalAcceleration(
        backend.model, backend.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return pose, np.asarray(velocity.vector).copy(), np.asarray(acceleration.vector).copy()


def run(cmg, reference, output=None):
    """Return a JSON-compatible report; all thresholds are declared below.

    The reference must contain the arrays returned by panda.task.build_reference.
    The tool is specified by CMG, so a mismatch with the PACDM adapter fails.
    """
    report = {"passed": False, "checks": {}, "details": {
        "scope": "Finite independent Pinocchio model/route checks; no hardware, contact, friction, grasp, global workspace or structural-arm-loop validation.",
        "independence": [
            "PinBackend compiles CMG directly and evaluates native Pinocchio FK/Jacobians/CRBA/RNEA/ABA; no MuJoCo import.",
            "Native tool velocity and classical acceleration are compared with explicit world XYZ-target derivatives, independent of PACDM residual/Jdot code.",
            "The physical finger equality is solved both with a 10-by-10 multiplier system and a separately written 9-by-8 reduction.",
            "PACDM core and original Panda adapter are imported unchanged. The six-dimensional tool closure is an imposed task, not a permanent arm loop.",
        ],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pinocchio": pin.__version__},
        "random_seed": 20260924,
        "failures": [],
    }}
    checks, details = report["checks"], report["details"]
    maxima, minima = {}, {}

    def check(name, value, limit, unit="dimensionless", relation="<="):
        value, limit = float(value), float(limit)
        passed = np.isfinite(value) and ({"<=": value <= limit, ">=": value >= limit,
                                         "==": value == limit}[relation])
        checks[name] = {"passed": bool(passed),
                        "value": value if np.isfinite(value) else None,
                        "limit": limit, "unit": unit, "relation": relation}

    def maximum(name, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite witness: {name}")
        maxima[name] = max(maxima.get(name, 0.), value)

    def minimum(name, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite witness: {name}")
        minima[name] = min(minima.get(name, value), value)

    try:
        ids = list(cmg["coordinate_ids"])
        expected_ids = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
        if ids != expected_ids:
            raise ValueError("This independent physical reduction requires the documented Panda CMG order")
        ref = {key: np.asarray(reference[key], dtype=float) for key in (
            "time", "q", "v", "a", "augmented_q", "augmented_v", "augmented_a",
            "mapping", "active", "active_v", "active_a")}
        count = len(ref["time"])
        shapes = {"q": (count, 9), "v": (count, 9), "a": (count, 9),
                  "augmented_q": (count, 15), "augmented_v": (count, 15),
                  "augmented_a": (count, 15), "mapping": (count, 15, 8),
                  "active": (count, 8), "active_v": (count, 8), "active_a": (count, 8)}
        if count < 2 or any(ref[k].shape != v for k, v in shapes.items()):
            raise ValueError("Reference array dimensions are invalid")
        if any(not np.all(np.isfinite(a)) for a in ref.values()):
            raise ValueError("Reference contains nonfinite values")
        if not np.all(np.diff(ref["time"]) > 0):
            raise ValueError("Reference time is not strictly increasing")
        graph = TaskGraph(cmg)
        backend = PinBackend(cmg)
        physical_graph = PandaGraph(cmg)
        physical_solver = PACDM(physical_graph)
        fid = _add_tool_frame(backend, cmg["tool"]["body"], cmg["tool"]["T_body_tool"])
        # This reduction is constructed directly from the physical equality,
        # independently of PACDM's differential mapping.
        reduction = np.zeros((9, 8))
        reduction[:7, :7] = np.eye(7)
        reduction[7:, 7] = 1.
        constraint = np.zeros((1, 9))
        constraint[0, 7:] = [1., -1.]
        moment = np.asarray(cmg["actuation"]["moment_matrix"], dtype=float)
        armature = np.asarray(cmg["armature"], dtype=float)
        if np.any(armature < 0):
            raise ValueError("Armature must be nonnegative")
        damping = np.asarray(cmg["damping"], dtype=float)
        stiffness = np.asarray(cmg["stiffness"], dtype=float)
        spring_ref = np.asarray(cmg["spring_reference"], dtype=float)
        aba_backend = PinBackend(cmg)
        aba_backend.model.armature[aba_backend._v_indices] = armature
        rng = np.random.default_rng(details["random_seed"])

        check("tree_coordinate_count", backend.model.nv, 9, "count", "==")
        check("physical_constraint_rank", np.linalg.matrix_rank(constraint), 1, "count", "==")
        check("physical_reduction_tangent", np.max(abs(constraint @ reduction)), 1e-14)
        check("reduced_actuator_map_identity", np.max(abs(reduction.T @ moment-np.eye(8))), 1e-14)
        check("physical_reference_equals_augmented", max(np.max(abs(ref[key]-ref["augmented_"+key][:, :9]))
                                                         for key in ("q", "v", "a")), 1e-13, "mixed SI")
        check("full_route_physical_finger_position_equality", np.max(abs(ref["q"] @ constraint.T)), 1e-9, "m")
        check("full_route_physical_finger_velocity_equality", np.max(abs(ref["v"] @ constraint.T)), 1e-10, "m/s")
        check("full_route_physical_finger_acceleration_equality", np.max(abs(ref["a"] @ constraint.T)), 1e-9, "m/s2")

        # Uniform witnesses plus every component's position/speed/acceleration
        # extremes prevent stationary endpoints from concealing curvature errors.
        indices = list(np.linspace(0, count-1, min(97, count)).astype(int))
        for key in ("q", "v", "a", "active", "active_v", "active_a"):
            indices.extend(np.argmax(abs(ref[key]), axis=0).tolist())
            indices.extend(np.argmin(ref[key], axis=0).tolist())
        indices = np.unique(indices)
        details["reference_samples"] = count
        details["route_witness_indices"] = indices.tolist()
        details["route_witness_times_s"] = ref["time"][indices].tolist()
        details["route_witness_count"] = len(indices)
        details["reference_duration_s"] = float(ref["time"][-1]-ref["time"][0])
        details["frame_convention"] = "World axes at tool origin; six-vectors are linear then angular; acceleration uses native getFrameClassicalAcceleration."
        details["target_orientation_convention"] = "R_world_tool = diag(1,-1,-1) Rx(rx) Ry(ry) Rz(rz)"
        details["physical_reduction"] = reduction.tolist()
        details["armature"] = armature.tolist()
        details["finite_difference_step"] = 2e-7
        route_completed = 0
        for index in indices:
            q, v, a = (ref[key][index] for key in ("q", "v", "a"))
            nmap = ref["mapping"][index]
            target, target_v, target_a, expected_map = _target_kinematics(
                ref["active"][index], ref["active_v"][index], ref["active_a"][index])
            jacobian = _tool_jacobian(backend, fid, q)
            pose, native_v, native_a = _tool_motion(backend, fid, q, v, a)
            maximum("native_tool_target_position", np.linalg.norm(pose[:3, 3]-target[:3, 3]))
            maximum("native_tool_target_rotation", Rotation.from_matrix(pose[:3, :3] @ target[:3, :3].T).magnitude())
            maximum("native_tool_velocity", np.max(abs(native_v-target_v)))
            maximum("native_tool_classical_acceleration", np.max(abs(native_a-target_a)))
            maximum("native_frame_velocity_Jv", np.max(abs(native_v-jacobian @ v)))
            maximum("native_tool_PACDM_tangent_map", np.max(abs(jacobian @ nmap[:9]-expected_map)))
            maximum("native_redundancy_tool_velocity", np.max(abs(jacobian @ nmap[:9, 6])))
            maximum("native_finger_tool_velocity", np.max(abs(jacobian @ nmap[:9, 7])))
            tree_poses, tree_twists = graph.tree.poses(q)
            pin_poses = backend.poses(q)
            for body, p in pin_poses.items():
                maximum("native_vs_graph_body_FK_position", np.linalg.norm(p[:3, 3]-tree_poses[body][:3, 3]))
                maximum("native_vs_graph_body_FK_rotation", np.max(abs(p[:3, :3]-tree_poses[body][:3, :3])))
            spatial = tree_twists[graph.tool_body]
            geometric = np.vstack((spatial[3:]-_skew(pose[:3, 3]) @ spatial[:3], spatial[:3]))
            maximum("native_vs_graph_tool_Jacobian", np.max(abs(jacobian-geometric)))
            task_wrench = rng.uniform(-10., 10., 6)
            generalized = jacobian.T @ task_wrench
            maximum("task_wrench_reduction", np.max(abs(nmap[:9].T @ generalized-expected_map.T @ task_wrench)))
            maximum("task_wrench_power", abs(v @ generalized-target_v @ task_wrench))
            # Drop only PACDM's acceleration-curvature term. Detect this through
            # native classical acceleration, not the same residual used to form it.
            wrong_a = (nmap @ ref["active_a"][index])[:9]
            wrong_motion = _tool_motion(backend, fid, q, v, wrong_a)[2]
            maximum("negative_omitted_acceleration_curvature", np.max(abs(wrong_motion-target_a)))
            route_completed += 1
        check("route_witnesses_completed", route_completed, len(indices), "count", "==")

        # Independent central FK differences provide a third Jacobian check at
        # twelve route witnesses. The same small probe is stated in the report.
        fd_indices = indices[np.unique(np.linspace(0, len(indices)-1, min(12, len(indices))).astype(int))]
        for index in fd_indices:
            q = ref["q"][index]
            analytic = _tool_jacobian(backend, fid, q)
            finite = np.empty((6, 9))
            step = 2e-7
            for col in range(9):
                delta = np.eye(9)[col]*step
                plus = _tool_motion(backend, fid, q+delta, np.zeros(9), np.zeros(9))[0]
                minus = _tool_motion(backend, fid, q-delta, np.zeros(9), np.zeros(9))[0]
                finite[:3, col] = (plus[:3, 3]-minus[:3, 3])/(2*step)
                finite[3:, col] = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec()/(2*step)
            maximum("native_tool_Jacobian_FK_finite_difference", np.max(abs(analytic-finite)))
        details["Jacobian_FD_witness_count"] = len(fd_indices)

        records = {joint["id"]: joint for joint in cmg["joints"]}
        lower = np.array([records[name]["limits"]["lower"] for name in ids])
        upper = np.array([records[name]["limits"]["upper"] for name in ids])
        random_q = lower+(upper-lower)*rng.uniform(.1, .9, (32, 9))
        random_q[:, 8] = random_q[:, 7]
        dynamics_q = np.vstack((ref["q"][indices], random_q))
        details["random_interior_samples"] = len(random_q)
        details["random_coordinate_interval_fraction"] = [.1, .9]
        details["dynamics_sample_count"] = len(dynamics_q)
        dynamics_completed = 0
        for q in dynamics_q:
            reduced_v = rng.uniform(-.5, .5, 8)
            reduced_v[-1] *= .04
            v = reduction @ reduced_v
            a = rng.uniform(-2., 2., 9)
            rigid_mass = backend.mass(q)
            mass = rigid_mass + np.diag(armature)
            bias = backend.bias(q, v)
            inverse = backend.inverse(q, v, a)
            maximum("RNEA_vs_CRBA_and_bias", _normalized_error(inverse, rigid_mass @ a+bias))
            minimum("rigid_mass_min_eigenvalue", np.linalg.eigvalsh(rigid_mass).min())
            minimum("mass_with_armature_min_eigenvalue", np.linalg.eigvalsh(mass).min())
            maximum("mass_symmetry", np.max(abs(mass-mass.T)))
            reduced_mass = reduction.T @ mass @ reduction
            minimum("reduced_mass_min_eigenvalue", np.linalg.eigvalsh(reduced_mass).min())
            kinetic = backend.energy(q, v)["kinetic_J"]
            maximum("native_kinetic_energy_mass_identity", abs(kinetic-.5*v @ rigid_mass @ v))
            actuation = rng.uniform(-10., 10., 8)
            effort = moment @ actuation-damping*v-stiffness*(q-spring_ref)
            reduced_a = np.linalg.solve(reduced_mass, reduction.T @ (effort-bias))
            lifted_a = reduction @ reduced_a
            multiplier_system = np.block([[mass, -constraint.T],
                                          [constraint, np.zeros((1, 1))]])
            rhs = np.r_[effort-bias, 0.]
            full = np.linalg.solve(multiplier_system, rhs)
            maximum("full_KKT_vs_physical_reduction_acceleration", _normalized_error(full[:9], lifted_a))
            maximum("full_KKT_equation_residual", np.max(abs(multiplier_system @ full-rhs)))
            maximum("reduced_acceleration_finger_equality", np.max(abs(constraint @ lifted_a)))
            maximum("physical_virtual_work", abs(v @ effort-reduced_v @ (reduction.T @ effort)))
            maximum("constraint_reaction_power", abs(v @ (constraint.T @ full[9:])))
            physical_map, info = physical_solver.mapping(q)
            if not info["success"]:
                raise RuntimeError(f"Physical PACDM mapping failed: {info}")
            maximum("physical_PACDM_vs_independent_reduction", np.max(abs(physical_map-reduction)))
            maximum("physical_PACDM_rank_error", abs(info["rank_full"]-1))
            expected_effort = inverse+armature*a
            native_inverse = aba_backend.inverse(q, v, a)
            maximum("native_RNEA_armature_identity", _normalized_error(native_inverse, expected_effort))
            native_a = pin.aba(aba_backend.model, aba_backend.data,
                               aba_backend._native_q(q), aba_backend._native_v(v, "v"),
                               aba_backend._native_v(expected_effort, "effort"))
            recovered_a = np.asarray(native_a)[aba_backend._v_indices]
            maximum("native_ABA_RNEA_roundtrip", _normalized_error(recovered_a, a))
            maximum("negative_armature_omission", np.max(abs(mass-rigid_mass)))
            dynamics_completed += 1
        check("dynamics_samples_completed", dynamics_completed, len(dynamics_q), "count", "==")
        details["native_ABA_scope"] = "Unconstrained 9-coordinate tree round-trip with source armature; physical equality is tested separately by KKT and 8-coordinate reduction."

        wrong_offset = np.asarray(cmg["tool"]["T_body_tool"]).copy()
        wrong_offset[2, 3] += .01
        altered = deepcopy(cmg)
        for joint in altered["joints"]:
            if joint["id"] == "joint2":
                joint["axis"] = [1., 0., 0.]
        wrong_backend = PinBackend(altered)
        wrong_axis_errors = []
        for index in indices:
            q = ref["q"][index]
            target, _, _, _ = _target_kinematics(ref["active"][index], np.zeros(8), np.zeros(8))
            body = backend.poses(q)[graph.tool_body]
            wrong_tool = body @ wrong_offset
            maximum("negative_wrong_tool_offset", np.linalg.norm(wrong_tool[:3, 3]-target[:3, 3]))
            wrong_axis_tool = wrong_backend.poses(q)[graph.tool_body] @ np.asarray(cmg["tool"]["T_body_tool"])
            wrong_axis_errors.append(float(np.linalg.norm(wrong_axis_tool[:3, 3]-target[:3, 3])))
        maximum("negative_wrong_joint_axis", max(wrong_axis_errors))
        details["negative_controls"] = {
            "tool_offset": "Add 0.010 m to the local tool z offset; original route target is unchanged.",
            "joint_axis": "Replace joint2 source local z rotation axis with local x; original route target is unchanged.",
            "curvature": "Use a=N*a_active and omit the Jdot correction; compare native classical tool acceleration with explicit target acceleration.",
            "armature": "Omit source joint armature from the mass matrix.",
        }
        for name, limit, unit in [
            ("native_tool_target_position", 1e-8, "m"),
            ("native_tool_target_rotation", 1e-8, "rad"),
            ("native_tool_velocity", 2e-8, "mixed m/s, rad/s"),
            ("native_tool_classical_acceleration", 2e-7, "mixed m/s2, rad/s2"),
            ("native_frame_velocity_Jv", 1e-10, "mixed m/s, rad/s"),
            ("native_tool_PACDM_tangent_map", 2e-8, "mixed SI"),
            ("native_redundancy_tool_velocity", 2e-9, "mixed SI"),
            ("native_finger_tool_velocity", 2e-9, "mixed SI"),
            ("native_vs_graph_body_FK_position", 2e-9, "m"),
            ("native_vs_graph_body_FK_rotation", 2e-9, "matrix entries"),
            ("native_vs_graph_tool_Jacobian", 2e-9, "mixed SI"),
            ("native_tool_Jacobian_FK_finite_difference", 2e-8, "mixed SI"),
            ("task_wrench_reduction", 2e-7, "mixed N, N m"),
            ("task_wrench_power", 2e-7, "W"),
            ("RNEA_vs_CRBA_and_bias", 1e-10, "normalized infinity error"),
            ("mass_symmetry", 1e-12, "mixed generalized SI"),
            ("native_kinetic_energy_mass_identity", 1e-10, "J"),
            ("full_KKT_vs_physical_reduction_acceleration", 1e-9, "normalized infinity error"),
            ("full_KKT_equation_residual", 1e-9, "mixed SI"),
            ("reduced_acceleration_finger_equality", 1e-10, "m/s2"),
            ("physical_virtual_work", 1e-10, "W"),
            ("constraint_reaction_power", 1e-10, "W"),
            ("physical_PACDM_vs_independent_reduction", 1e-12, "dimensionless"),
            ("physical_PACDM_rank_error", 0, "count"),
            ("native_RNEA_armature_identity", 1e-10, "normalized infinity error"),
            ("native_ABA_RNEA_roundtrip", 1e-9, "normalized infinity error"),
        ]:
            check(name, maxima.get(name, np.inf), limit, unit)
        for name in ("rigid_mass_min_eigenvalue", "mass_with_armature_min_eigenvalue", "reduced_mass_min_eigenvalue"):
            check(name, minima.get(name, -np.inf), 1e-6, "mixed generalized SI", ">=")
        for name, limit, unit in [
            ("negative_omitted_acceleration_curvature", 1e-4, "mixed m/s2, rad/s2"),
            ("negative_wrong_tool_offset", .0099, "m"),
            ("negative_wrong_joint_axis", .01, "m"),
            ("negative_armature_omission", .09, "mixed generalized SI"),
        ]:
            check(name+"_detected", maxima.get(name, 0.), limit, unit, ">=")
    except Exception as error:
        details["failures"].append(f"{type(error).__name__}: {error}")
        check("mechanics_validation_completed", 0, 1, "boolean", "==")

    details["maxima"] = maxima
    details["minima"] = minima
    report["passed_checks"] = sum(item["passed"] for item in checks.values())
    report["total_checks"] = len(checks)
    report["passed"] = bool(checks and all(item["passed"] for item in checks.values()))
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    from panda.model import load_model
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "data" / "reference.npz", allow_pickle=False) as reference:
        result = run(load_model(), reference, root / "results" / "pin_mechanics.json")
    print(json.dumps({"passed": result["passed"], "passed_checks": result["passed_checks"],
                      "total_checks": result["total_checks"],
                      "failures": {name: item for name, item in result["checks"].items() if not item["passed"]},
                      "errors": result["details"]["failures"]}, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
