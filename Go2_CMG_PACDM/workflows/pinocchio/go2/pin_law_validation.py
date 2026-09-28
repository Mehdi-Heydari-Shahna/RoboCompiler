"""Deterministic, controller-free audits of the Pinocchio contact plant.

Checks analytic rigid-point Jacobians against FK finite differences and
executes free-flight, compression, slip, frictionless and release cases.
These are local numerical checks, not hardware/contact identification.
"""
from pathlib import Path
import json
import numpy as np

from .model import load_model
from .pin_simulation import Plant, KN, DN, RADIUS, HURDLES


def validate_contact_law(root=None, output=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    cmg = load_model(root / "data/go2_cmg.json")
    with np.load(root / "data/reference.npz", allow_pickle=False) as saved:
        home = saved["q"][0].copy()
    checks = {}
    examples = {}

    def gate(name, value, limit, relation="<=", unit="dimensionless"):
        value = float(value)
        passed = np.isfinite(value) and (value <= limit if relation == "<=" else
                                        value >= limit if relation == ">=" else value == limit)
        checks[name] = dict(value=value, limit=float(limit), relation=relation,
                            unit=unit, passed=bool(passed))

    # Each physical material point is fixed in its body for this derivative;
    # the contact-point Jacobian must include the sphere-radius lever arm.
    plant = Plant(cmg)
    q = home.copy()
    q[:6] = [.12, -.07, .43, .23, -.12, .09]
    direction = np.linspace(-.31, .29, 18)
    points, linear, angular, push_jac = plant.kinematics(q)
    poses = plant.b.poses(q)
    epsilon = 2e-7
    plus = plant.b.poses(q + epsilon * direction)
    minus = plant.b.poses(q - epsilon * direction)
    n = np.array([.2, -.3, 1.]); n /= np.linalg.norm(n)
    errors = []
    center_errors = []
    for index, foot in enumerate(cmg["feet"]):
        body = foot["body"]
        local_center = np.asarray(foot["point_m"])
        material_point = local_center + poses[body][:3, :3].T @ (-RADIUS * n)

        def fd_point(point):
            pplus = plus[body][:3, :3] @ point + plus[body][:3, 3]
            pminus = minus[body][:3, :3] @ point + minus[body][:3, 3]
            return (pplus - pminus) / (2 * epsilon)

        world_velocity = linear[index] @ direction + np.cross(angular[index] @ direction, -RADIUS * n)
        errors.append(np.max(abs(world_velocity - fd_point(material_point))))
        center_errors.append(np.max(abs(linear[index] @ direction - fd_point(local_center))))
    center = plant.base_com
    fd_com = ((plus["base"][:3, :3] @ center + plus["base"][:3, 3]) -
              (minus["base"][:3, :3] @ center + minus["base"][:3, 3])) / (2 * epsilon)
    gate("sphere_material_point_Jacobian_FD", max(errors), 2e-8, unit="m/s")
    gate("sphere_center_Jacobian_FD", max(center_errors), 2e-8, unit="m/s")
    gate("base_COM_push_Jacobian_FD", np.max(abs(push_jac @ direction - fd_com)), 2e-8, unit="m/s")
    examples["jacobian"] = dict(q=q.tolist(), direction=direction.tolist(), normal=n.tolist(), epsilon=epsilon)

    dt = .001
    for name, dz, velocity, mu in [
        ("flight", .5, [0., 0., 0.], .8),
        ("compression", -.003, [0., 0., 0.], .8),
        ("sliding", -.001, [1., .3, 0.], .2),
        ("frictionless", -.001, [1., .3, 0.], 0.),
        ("release", -.001, [0., 0., .8], .8),
    ]:
        plant = Plant(cmg)
        q = home.copy(); q[2] += dz
        v = np.zeros(18); v[:3] = velocity
        points, linear, angular, _ = plant.kinematics(q)
        contacts = plant.contacts(points, linear, angular)
        qn, vn, diagnostics = plant.step(q, v, np.zeros(12), np.zeros(3), dt, mu)
        examples[name] = dict(q=q.tolist(), v=v.tolist(), dt_s=dt, mu=mu,
                              tau_Nm=[0.] * 12, push_N=[0.] * 3,
                              contact_count=len(contacts), forces_N=diagnostics["forces"].tolist(),
                              q_next=qn.tolist(), v_next=vn.tolist())
        gate(name + ".momentum_balance", diagnostics["dynamics_residual"], 1e-8, unit="N or Nm")
        gate(name + ".fixed_point_convergence", diagnostics["contact_residual"], 1e-9, unit="Ns")
        gate(name + ".constitutive_residual", diagnostics["law_residual"], 1e-6, unit="N")
        gate(name + ".nonnegative_normal", diagnostics["minimum_normal"], -1e-10, ">=", "N")
        gate(name + ".friction_cone_excess", diagnostics["cone_excess"], 1e-10, unit="N")

        if name == "flight":
            expected_v = np.zeros(18); expected_v[2] = -9.81 * dt
            gate("flight.no_contacts", len(contacts), 0, "==", "count")
            gate("flight.no_contact_forces", np.max(abs(diagnostics["forces"])), 0., "==", "N")
            gate("flight.gravity_velocity", np.max(abs(vn - expected_v)), 1e-11, unit="mixed SI")
            gate("flight.semiimplicit_position", np.max(abs(qn - q - dt * expected_v)), 1e-11, unit="mixed SI")
        else:
            gate(name + ".four_contact_candidates", len(contacts), 4, "==", "count")
            normal_error = 0.
            tangential_power = 0.
            tangential_norm = 0.
            saturation_error = 0.
            direction_error = 0.
            for leg, key, gap, basis, jac, _ in contacts:
                force = basis @ diagnostics["forces"][leg]
                speed = jac @ vn
                # Independently reconstruct the implicit normal spring force
                # from the end-of-step linearized compression and velocity.
                expected_normal = max(0., -KN * (gap + dt * speed[2]) - DN * speed[2])
                normal_error = max(normal_error, abs(force[2] - expected_normal))
                tangential_norm = max(tangential_norm, np.linalg.norm(force[:2]))
                tangential_power += float(force[:2] @ speed[:2])
                if name == "sliding":
                    saturation_error = max(saturation_error, abs(np.linalg.norm(force[:2]) - mu * force[2]))
                    force_direction = force[:2] / np.linalg.norm(force[:2])
                    slip_direction = speed[:2] / np.linalg.norm(speed[:2])
                    direction_error = max(direction_error, np.linalg.norm(force_direction + slip_direction))
            gate(name + ".independent_normal_law", normal_error, 1e-6, unit="N")
            if name == "compression":
                gate("compression.positive_support", np.min(diagnostics["normal"]), 1., ">=", "N")
            if name == "sliding":
                gate("sliding.saturated_friction", saturation_error, 1e-8, unit="N")
                gate("sliding.force_opposes_slip", direction_error, 1e-7)
                gate("sliding.dissipative_work_rate", tangential_power, 0., unit="W")
            if name == "frictionless":
                gate("frictionless.zero_tangential_force", tangential_norm, 1e-12, unit="N")
            if name == "release":
                gate("release.zero_tensile_or_sticking_force", np.max(abs(diagnostics["forces"])), 1e-10, unit="N")

    # A sphere overlapping a rail side must see its outward side normal,
    # rather than receive an invented vertical support force.
    plant = Plant(cmg)
    x, height = HURDLES[-1]
    positions = np.tile([0., 0., 1.], (4, 1)).astype(float)
    positions[0] = [x - .025 - RADIUS / 2, 0., height / 2]
    contacts = plant.contacts(positions, np.zeros((4, 3, 18)), np.zeros((4, 3, 18)))
    rail = contacts[0]
    gate("rail_side.contact_count", len(contacts), 1, "==", "count")
    gate("rail_side.surface_selected", rail[1][1] == f"rail{len(HURDLES)-1}", 1, "==", "boolean")
    gate("rail_side.signed_gap", abs(rail[2] + RADIUS / 2), 1e-12, unit="m")
    gate("rail_side.outward_normal", np.max(abs(rail[3][2] - [-1., 0., 0.])), 1e-12)
    examples["rail_side"] = dict(points_m=positions.tolist(), expected_gap_m=-RADIUS/2,
                                  expected_normal=[-1., 0., 0.])
    result = dict(passed=all(c["passed"] for c in checks.values()),
                  passed_checks=sum(c["passed"] for c in checks.values()), total_checks=len(checks),
                  checks=checks, examples=examples,
                  scope="Local deterministic Pinocchio plant checks with zero motor commands and no controller. "
                        "Not a hardware contact calibration, exhaustive stability proof or independent contact engine comparison.")
    destination = Path(output) if output is not None else root / "results_pinocchio/contact_law_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    result = validate_contact_law()
    print(json.dumps({k: result[k] for k in ("passed", "passed_checks", "total_checks")}, indent=2))
    for name, check in result["checks"].items():
        if not check["passed"]:
            print(name, check)
    raise SystemExit(0 if result["passed"] else 1)
