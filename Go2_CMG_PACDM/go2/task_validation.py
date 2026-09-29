"""Independent checks of saved PACDM gait-reference geometry and derivatives.

This checks the prescribed reference, not executed contact stability. Actual
slip, falls, contact forces, and tracking belong to simulation acceptance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from go2.contact import ContactGraph, FootKinematics, support_audit
from vendor.pacdm_original import rank


def _json_safe(value):
    """Replace NaN/inf by None: a non-finite audit value is a failed check, and the
    evidence must still be writable as strict JSON (allow_nan=False)."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_reference(root):
    """Validate data/reference.npz and return JSON-safe acceptance evidence.

    q,v,a use the source x/y/z/yaw/pitch/roll plus twelve joint chart.
    N maps base6 + toe-target12 rates into physical chart rates. Toe
    accelerations are verified with an independently recomputed directional
    Jacobian derivative; they are not compared only to stored derivatives.
    """
    root = Path(root)
    cmg = json.loads((root / "data" / "go2_cmg.json").read_text(encoding="utf-8"))
    with np.load(root / "data" / "reference.npz", allow_pickle=False) as source:
        data = {name: source[name] for name in
                ("time", "q", "v", "a", "feet", "stance", "N", "active", "active_v", "active_a")}
    count = len(data["time"])
    shapes = {"time": (count,), "q": (count, 18), "v": (count, 18), "a": (count, 18),
              "feet": (count, 4, 3), "stance": (count, 4), "N": (count, 18, 18),
              "active": (count, 18), "active_v": (count, 18), "active_a": (count, 18)}
    checks = []

    def check(name, value, limit, passed=None):
        if isinstance(value, np.generic):
            value = value.item()
        checks.append(dict(name=name, passed=bool(value <= limit if passed is None else passed),
                           value=value, limit=limit))

    good_shape = count >= 3 and all(data[name].shape == shape for name, shape in shapes.items())
    check("reference schema", good_shape, True, good_shape)
    finite = all(np.all(np.isfinite(array)) for array in data.values())
    check("finite reference values", finite, True, finite)
    if not good_shape or not finite:
        return _json_safe(dict(passed=False, checks=checks, samples=count))

    q, v, a, mappings = (data[name] for name in ("q", "v", "a", "N"))
    qa, va, aa = (data[name] for name in ("active", "active_v", "active_a"))
    stance = data["stance"].astype(bool)
    time = data["time"]
    check("strictly increasing reference time", float(np.min(np.diff(time))), 0.,
          np.all(np.diff(time) > 0))
    check("base target assembly", float(np.max(abs(q[:, :6] - qa[:, :6]))), 1e-12)
    check("saved toe targets equal independent coordinates",
          float(np.max(abs(data["feet"] - qa[:, 6:].reshape(count, 4, 3)))), 1e-12)
    check("velocity equals PACDM differential map",
          float(np.max(abs(v - np.einsum("nij,nj->ni", mappings, va)))), 1e-10)
    check("base acceleration equals prescribed acceleration", float(np.max(abs(a[:, :6] - aa[:, :6]))), 1e-10)

    records = {joint["id"]: joint for joint in cmg["joints"]}
    lower = np.asarray([records[name]["limits"]["lower"] for name in cmg["coordinate_ids"]])
    upper = np.asarray([records[name]["limits"]["upper"] for name in cmg["coordinate_ids"]])
    bound_violation = max(0., float(np.max(lower - q)), float(np.max(q - upper)))
    check("reference respects authored joint and chart limits", bound_violation, 1e-10)
    check("base Euler chart remains regular", float(np.max(abs(q[:, 4]))), 1.45)

    kinematics = FootKinematics(cmg, q[0])
    graph = ContactGraph(cmg, q[0])
    target_derivative = np.c_[np.zeros((12, 6)), np.eye(12)]
    residual_max = velocity_max = tangent_max = curvature_max = curvature_norm_max = 0.
    rcond_min = np.inf
    ranks = set()
    # Every sample is checked geometrically and at velocity level. The
    # acceleration calculation uses a different FD step from task creation.
    curvature_samples = np.unique(np.linspace(0, count - 1, min(count, 151), dtype=int))
    curvature_set = set(curvature_samples.tolist())
    reconstruction_samples = set(np.unique(np.linspace(0, count - 1, min(count, 25), dtype=int)).tolist())
    mapping_reconstruction_max = 0.
    foot_velocities = []
    for index in range(count):
        points, jacobian = kinematics.points_and_jacobians(q[index])
        j = jacobian.reshape(12, 18)
        residual_max = max(residual_max, float(np.max(abs(points - data["feet"][index]))))
        foot_velocity = j @ v[index]
        foot_velocities.append(foot_velocity.reshape(4, 3))
        velocity_max = max(velocity_max, float(np.max(abs(foot_velocity - va[index, 6:]))))
        tangent_max = max(tangent_max, float(np.max(abs(j @ mappings[index] - target_derivative))))
        passive = j[:, 6:]
        ranks.add(rank(passive))
        rcond_min = min(rcond_min, float(1 / np.linalg.cond(passive, p=1)))
        if index in curvature_set:
            step = 5e-6
            j_plus = kinematics.points_and_jacobians(q[index] + step * v[index])[1].reshape(12, 18)
            j_minus = kinematics.points_and_jacobians(q[index] - step * v[index])[1].reshape(12, 18)
            jdot = (j_plus - j_minus) / (2 * step)
            curvature = jdot @ v[index]
            acceleration_error = j @ a[index] + curvature - aa[index, 6:]
            curvature_max = max(curvature_max, float(np.max(abs(acceleration_error))))
            curvature_norm_max = max(curvature_norm_max, float(np.max(abs(curvature))))
        if index in reconstruction_samples:
            augmented = np.r_[q[index], qa[index, 6:]]
            rebuilt, info = graph.solver.mapping(augmented)
            if not info["success"]:
                mapping_reconstruction_max = np.inf
            else:
                mapping_reconstruction_max = max(mapping_reconstruction_max,
                                                 float(np.max(abs(rebuilt[:18] - mappings[index]))))
    check("all sampled foot task closures", residual_max, 1e-8)
    check("all sampled foot velocity constraints", velocity_max, 1e-7)
    check("all sampled full tangent-map equations", tangent_max, 1e-8)
    check("independent PACDM map reconstruction", mapping_reconstruction_max, 1e-7)
    check("directional-Jdot acceleration constraints", curvature_max, 2e-5)
    check("leg-dependent constraint rank", sorted(ranks), [12], ranks == {12})
    check("leg-dependent Jacobian conditioning", rcond_min, 1e-10, rcond_min >= 1e-10)

    count_stance = np.sum(stance, axis=1)
    modes, first_indices = np.unique(stance, axis=0, return_index=True)
    switches = int(np.sum(np.any(stance[1:] != stance[:-1], axis=1)))
    check("scheduled support count", int(np.min(count_stance)), 2, np.all(count_stance >= 2))
    check("scheduled support changes", switches, 1, switches >= 1)
    stable_stance = stance.copy()
    stable_stance[1:] &= stance[:-1]
    stable_stance[:-1] &= stance[1:]
    velocities = np.asarray(foot_velocities)
    stationary_error = float(np.max(abs(velocities[stable_stance]))) if np.any(stable_stance) else np.inf
    check("interior stance targets remain stationary", stationary_error, 1e-7)
    mode_audits = []
    for mode, index in zip(modes, first_indices):
        if not np.any(mode):
            continue
        audit = support_audit(cmg, q[index], mode)
        mode_audits.append(audit)
        expected_rank = 3 * int(np.sum(mode))
        label = ",".join(audit["stance_feet"])
        check("support rank and tangent: " + label, audit["rank"], expected_rank,
              audit["success"] and audit["rank"] == expected_rank and
              audit["mobility"] == 18 - expected_rank and audit["tangent_residual"] <= 1e-8)

    actuation = np.asarray(cmg["actuation"]["moment_matrix"])
    check("eighteen physical coordinates and twelve source actuators", list(actuation.shape), [18, 12],
          actuation.shape == (18, 12))
    check("six floating-base coordinates unactuated", float(np.max(abs(actuation[:6]))), 0.)
    check("no permanent structural loops", len(cmg["closures"]), 0)
    return _json_safe(dict(passed=all(item["passed"] for item in checks), checks=checks, samples=count,
                acceleration_samples=len(curvature_samples), duration_s=float(time[-1] - time[0]),
                max_acceleration_curvature_m_s2=curvature_norm_max,
                scheduled_support_counts=sorted(set(count_stance.astype(int).tolist())),
                scheduled_support_switches=switches, mode_audits=mode_audits,
                scope="Prescribed gait geometry and derivatives; simulation contact execution is validated separately."))
