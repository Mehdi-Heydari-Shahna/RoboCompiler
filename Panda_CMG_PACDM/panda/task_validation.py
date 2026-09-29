"""Route-wide checks of the imposed tool task and its PACDM reference.

The arm is a serial robot. Virtual target coordinates encode an imposed task
closure; only the source finger equality is a permanent physical constraint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation

from vendor.pacdm_original import PACDM
from .pin_backend import PinBackend
from .task import TaskGraph


def _nan_max(*values):
    """max() that propagates NaN; the built-in max() silently skips a NaN argument."""
    values = [float(value) for value in values]
    return float('nan') if any(np.isnan(values)) else max(values)


def _nan_min(*values):
    """min() that propagates NaN; the built-in min() silently skips a NaN argument."""
    values = [float(value) for value in values]
    return float('nan') if any(np.isnan(values)) else min(values)


def _json_safe(value):
    """Replace NaN/inf by None so that a failing record can still be written as strict JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_task(cmg, reference, output=None):
    """Validate reference arrays and the same BPoly interpolation as simulation."""
    report = {"passed": False, "checks": {}, "details": {
        "scope": "Finite numerical checks of the prescribed task route and the virtual tool constraint on a serial arm.",
        "coordinate_interpretation": "9 physical coordinates + 6 massless target coordinates; 6 imposed pose equations + 1 physical finger equation; 8 independent coordinates.",
        "redundancy_policy": "Joint3 is prescribed independently of the tool pose. The default mission sweeps +0.25/-0.20 rad during inspection and returns to zero.",
        "interpolation": "BPoly.from_derivatives(time, stack([q,v,a], axis=1)), exactly as used by the contact controller.",
    }}
    checks, details = report["checks"], report["details"]

    def check(name, value, limit, unit="dimensionless", relation="<="):
        value = float(value)
        passed = bool(np.isfinite(value) and (value <= limit if relation == "<=" else
                                             value >= limit if relation == ">=" else value == limit))
        checks[name] = {"passed": passed, "value": value if np.isfinite(value) else None,
                        "limit": limit, "unit": unit, "relation": relation}

    try:
        ref = reference
        t, q = np.asarray(ref["time"]), np.asarray(ref["q"])
        state = np.asarray(ref["augmented_q"])
        velocity, acceleration = np.asarray(ref["augmented_v"]), np.asarray(ref["augmented_a"])
        maps = np.asarray(ref["mapping"])
        active, av, aa = (np.asarray(ref[key]) for key in ("active", "active_v", "active_a"))
        n = len(t)
        graph, pin = TaskGraph(cmg), PinBackend(cmg)
        solver = PACDM(graph)
        expected = {"q": (n, 9), "v": (n, 9), "a": (n, 9),
                    "augmented_q": (n, 15), "augmented_v": (n, 15), "augmented_a": (n, 15),
                    "mapping": (n, 15, 8), "active": (n, 8), "active_v": (n, 8), "active_a": (n, 8)}
        shape_errors = sum(np.shape(ref[key]) != shape for key, shape in expected.items())
        check("reference_array_shape_errors", shape_errors, 0, "count", "==")
        if shape_errors:
            raise ValueError("Reference shapes are inconsistent")
        check("reference_nonfinite_values", sum(np.size(ref[key]) - np.count_nonzero(np.isfinite(ref[key]))
                                                for key in expected), 0, "count", "==")
        check("reference_time_monotonic", int(np.all(np.diff(t) > 0)), 1, "boolean", "==")
        check("physical_reference_matches_augmented", np.max(abs(q-state[:, :9])), 1e-13, "mixed rad/m")
        check("independent_reference_matches_augmented", np.max(abs(active-state[:, graph.active])), 1e-13, "mixed rad/m")
        check("velocity_active_coordinates", np.max(abs(velocity[:, graph.active]-av)), 1e-12, "mixed SI/s")
        check("acceleration_active_coordinates", np.max(abs(acceleration[:, graph.active]-aa)), 1e-12, "mixed SI/s2")
        check("physical_joint_limit_violation", _nan_max(0., float(np.max(graph.lower[:9]-q)),
                                                  float(np.max(q-graph.upper[:9]))), 1e-10, "mixed rad/m")
        check("source_finger_coupling_over_route", np.max(abs(q[:, 7]-q[:, 8])), 1e-9, "m")

        maxima = {key: 0. for key in ("closure", "tangent", "velocity", "acceleration", "active_identity",
                                         "Pinocchio_target_position", "Pinocchio_target_rotation")}
        min_rcond, bad_rank, failures = np.inf, 0, []
        # Recompute every sample rather than trusting maxima saved at generation.
        for i, (x, v, a, mapping) in enumerate(zip(state, velocity, acceleration, maps)):
            r, jacobian, delta = graph.residual(x)
            maxima["closure"] = _nan_max(maxima["closure"], float(np.max(abs(r))))
            maxima["tangent"] = _nan_max(maxima["tangent"], float(np.max(abs(jacobian @ mapping))))
            maxima["velocity"] = _nan_max(maxima["velocity"], float(np.max(abs(jacobian @ v))))
            maxima["active_identity"] = _nan_max(maxima["active_identity"], float(np.max(abs(mapping[graph.active]-np.eye(8)))))
            _, info = solver.mapping(x)
            min_rcond = _nan_min(min_rcond, info["rcond"])
            bad_rank += int(info["rank_full"] != 7 or info["rank_passive"] != 7)
            if not info["success"]:
                failures.append({"index": i, "time_s": float(t[i]), "info": info})
            endpoint = pin.poses(x[:9])[graph.tool_body] @ graph.tool_transform
            target = graph.target_pose(x)
            maxima["Pinocchio_target_position"] = _nan_max(maxima["Pinocchio_target_position"], float(np.linalg.norm(endpoint[:3, 3]-target[:3, 3])))
            maxima["Pinocchio_target_rotation"] = _nan_max(maxima["Pinocchio_target_rotation"], float(Rotation.from_matrix(endpoint[:3, :3] @ target[:3, :3].T).magnitude()))
            # A different probe size from generation tests sensitivity of Jdot.
            epsilon = 2e-5 / max(1., np.linalg.norm(v))
            jdot = (graph.residual(x+epsilon*v)[1]-graph.residual(x-epsilon*v)[1])/(2*epsilon)
            maxima["acceleration"] = _nan_max(maxima["acceleration"], float(np.max(abs(jacobian @ a + jdot @ v))))
        for name, limit, unit in [
                ("closure", 1e-8, "mixed rad/m"), ("tangent", 1e-9, "mixed SI"),
                ("velocity", 1e-9, "mixed SI/s"), ("acceleration", 2e-7, "mixed SI/s2"),
                ("active_identity", 1e-12, "dimensionless"),
                ("Pinocchio_target_position", 1e-8, "m"), ("Pinocchio_target_rotation", 1e-8, "rad")]:
            check("route_"+name, maxima[name], limit, unit)
        check("route_incorrect_constraint_rank_samples", bad_rank, 0, "count", "==")
        check("route_PACDM_mapping_failures", len(failures), 0, "count", "==")
        check("route_minimum_reciprocal_condition", min_rcond, 1e-4, "dimensionless", ">=")
        details["sample_count"] = n
        details["duration_s"] = float(t[-1]-t[0])
        details["sample_spacing_s"] = [float(np.min(np.diff(t))), float(np.max(np.diff(t)))]
        details["route_maxima"] = maxima
        details["mapping_failures"] = failures
        details["arm_position_min_rad"] = q[:, :7].min(axis=0).tolist()
        details["arm_position_max_rad"] = q[:, :7].max(axis=0).tolist()
        details["arm_peak_speed_rad_s"] = np.max(abs(ref["v"][:, :7]), axis=0).tolist()
        details["arm_peak_acceleration_rad_s2"] = np.max(abs(ref["a"][:, :7]), axis=0).tolist()
        details["speed_acceleration_interpretation"] = "Observed reference values; the MJCF source does not specify hardware speed or acceleration limits."

        interpolation = BPoly.from_derivatives(t, np.stack([ref["q"], ref["v"], ref["a"]], axis=1))
        target_interpolation = BPoly.from_derivatives(t, np.stack([active, av, aa], axis=1))
        for order, key, unit in [(0, "q", "mixed rad/m"), (1, "v", "mixed SI/s"), (2, "a", "mixed SI/s2")]:
            check("BPoly_knot_"+key+"_reproduction", np.max(abs(interpolation(t, nu=order)-ref[key])), 1e-6, unit)
        indices = np.unique(np.linspace(0, n-2, 100).astype(int))
        midpoint = (t[indices]+t[indices+1])/2
        errors = {"position": 0., "rotation": 0., "constraint": 0., "velocity_constraint": 0.}
        for time in midpoint:
            physical = interpolation(time)
            prescribed = target_interpolation(time)
            augmented = np.r_[physical, prescribed[:6]]
            augv = np.r_[interpolation(time, nu=1), target_interpolation(time, nu=1)[:6]]
            endpoint = pin.poses(physical)[graph.tool_body] @ graph.tool_transform
            target = graph.target_pose(augmented)
            errors["position"] = _nan_max(errors["position"], float(np.linalg.norm(endpoint[:3, 3]-target[:3, 3])))
            errors["rotation"] = _nan_max(errors["rotation"], float(Rotation.from_matrix(endpoint[:3, :3] @ target[:3, :3].T).magnitude()))
            residual, jacobian, _ = graph.residual(augmented)
            errors["constraint"] = _nan_max(errors["constraint"], float(np.max(abs(residual))))
            errors["velocity_constraint"] = _nan_max(errors["velocity_constraint"], float(np.max(abs(jacobian @ augv))))
        for name, limit, unit in [("position", 1e-6, "m"), ("rotation", 1e-6, "rad"),
                                  ("constraint", 1e-6, "mixed rad/m"), ("velocity_constraint", 2e-6, "mixed SI/s")]:
            check("BPoly_midpoint_"+name, errors[name], limit, unit)
        details["midpoint_sample_count"] = len(midpoint)
        details["midpoint_maxima"] = errors

        # A tilted operating pose supplements near-home derivatives in mechanics
        # validation. Perturb target position/orientation to leave the manifold.
        index = int(np.argmax(abs(active[:, 3])))
        off = state[index].copy()
        off[graph.nt:graph.nt+6] += [.003, -.002, .001, .015, -.01, .012]
        analytic = graph.residual(off)[1]
        h = 2e-7
        fd = np.column_stack([(graph.residual(off+h*np.eye(graph.n)[j])[0]-
                               graph.residual(off-h*np.eye(graph.n)[j])[0])/(2*h)
                              for j in range(graph.n)])
        check("tilted_off_manifold_Jacobian_FD", np.max(abs(analytic-fd)), 2e-7, "mixed SI")
        details["tilted_derivative_sample_s"] = float(t[index])

        # A genuine arm redundancy change: target fixed, joint3 varied.
        initial = state[index].copy()
        redundant_errors, redundancy_success = [], 0
        for offset in [-.06, .06]:
            independent = initial[graph.active].copy()
            independent[6] += offset
            solved, info = solver.acquire(independent, initial)
            if info["success"]:
                redundancy_success += 1
                target = graph.target_pose(solved)
                endpoint = pin.poses(solved[:9])[graph.tool_body] @ graph.tool_transform
                redundant_errors.append(_nan_max(float(np.linalg.norm(endpoint[:3, 3]-target[:3, 3])),
                                            float(Rotation.from_matrix(endpoint[:3, :3] @ target[:3, :3].T).magnitude())))
        check("local_redundancy_alternatives_solved", redundancy_success, 2, "count", "==")
        check("local_redundancy_target_pose_error", (_nan_max(*redundant_errors) if redundant_errors else np.inf), 1e-8, "mixed rad/m")
        details["redundancy_test_offsets_rad"] = [-.06, .06]
    except Exception as error:
        details["error"] = f"{type(error).__name__}: {error}"
        check("task_reference_validation_completed", 0, 1, "boolean", "==")

    report["passed"] = bool(checks and all(value["passed"] for value in checks.values()))
    report["passed_checks"] = sum(value["passed"] for value in checks.values())
    report["total_checks"] = len(checks)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    from .model import load_model
    root = Path(__file__).resolve().parents[1]
    source = root/"data"/"reference.npz"
    if not source.exists():
        source = root/"data"/"task_reference.npz"
    with np.load(source, allow_pickle=False) as data:
        result = validate_task(load_model(), data, root/"results"/"task_validation.json")
    print(json.dumps({"passed": result["passed"], "passed_checks": result["passed_checks"],
                      "total_checks": result["total_checks"],
                      "failures": {key: value for key, value in result["checks"].items() if not value["passed"]},
                      "details": result["details"]}, indent=2))
