"""Two pressure/flow-limited hydraulic sprocket drives.

These are declared, synthetic engineering parameters, NOT identified excavator
hardware. The model contains two independently metered, four-quadrant hydraulic
motors, ideal reduction gears, compressible lines, cross-port leakage, output
friction, supply/return valves, relief and tank make-up. Braking energy is
dissipated; no accumulator, regeneration, engine, pump efficiency or thermal
prediction is claimed. The supply is an ideal 25 MPa reservoir with bounded flow.

For each drive, C dp/dt = q - [D*w, -D*w] - leak - relief + make-up,
and output torque is D*(p_A-p_B) minus dissipative output friction. D includes
gear ratio. Both chamber volumes are constant effective fluid volumes.

The bounded backward-Euler pressure solve is stable at the advertised time
steps. Returned torque uses END pressure and is held for the subsequent physics
step. Its exactly reconciled discrete energy ledger exposes numerical storage
integration dissipation separately from physical losses. The ledger uses the
supplied, held shaft speed; a coupled simulator must separately audit actual
shaft work over its step and report the partitioned coupling error.

Motor inertia belongs in the mechanical simulator, not this pressure bank. The
integrated tracked model uses explicit physical rotors with 45:1 gear constraints
and zero joint armature. reflected_rotor_inertia_kg_m2 is a reference equivalent
at the sprocket, not an additional inertia: adding it would double-count the
explicit rotor. Include rotor bodies in the complete mechanical energy audit.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


PARAMETERS = {
    "provenance": "Declared synthetic engineering assumptions; not factory ratings or calibrated hardware",
    "supply_pressure_Pa": 25e6,
    "initial_charge_pressure_Pa": 0.3e6,
    "bulk_modulus_Pa": 8e8,
    "effective_chamber_volume_m3": 0.004,
    "motor_displacement_m3_rev": 125e-6,
    "gear_ratio": 45.0,
    "motor_rotor_inertia_kg_m2": 0.008,
    "reflected_rotor_inertia_kg_m2": 16.2,
    "valve_max_flow_m3_s": 0.0015,
    "pump_max_flow_per_drive_m3_s": 0.0015,
    "pump_max_flow_bank_m3_s": 0.003,
    "pressure_response_s": 0.025,
    "cross_port_leakage_m3_s_Pa": 3e-13,
    "output_coulomb_friction_Nm": 180.0,
    "output_viscous_friction_Nm_s_rad": 80.0,
    "friction_smoothing_speed_rad_s": 0.01,
    "regeneration": False,
    "pressure_integrator": "Bounded backward Euler; numerical storage loss separately reported",
}


def _bounded_pressure(old, flow, speed, dt, compliance, displacement, leakage, ps):
    """Solve the two-chamber monotone linear complementarity problem exactly."""
    diagonal = compliance / dt + leakage
    matrix = np.array([[diagonal, -leakage], [-leakage, diagonal]])
    rhs = compliance / dt * old + flow + np.array([-displacement * speed, displacement * speed])
    unconstrained = np.linalg.solve(matrix, rhs)
    if np.all(unconstrained >= 0.0) and np.all(unconstrained <= ps):
        return unconstrained, np.zeros(2), np.zeros(2)
    # Status -1: tank lower rail, 0: free, +1: pressure relief upper rail.
    # There are only nine sets, so enumerate rather than hide clipping work.
    for status in itertools.product((-1, 0, 1), repeat=2):
        free = np.array([i for i, value in enumerate(status) if value == 0], dtype=int)
        fixed = np.array([i for i, value in enumerate(status) if value != 0], dtype=int)
        pressure = np.array([ps if value == 1 else 0.0 for value in status])
        if free.size:
            pressure[free] = np.linalg.solve(matrix[np.ix_(free, free)], rhs[free] - matrix[np.ix_(free, fixed)] @ pressure[fixed])
        if np.any(pressure < -1e-6) or np.any(pressure > ps + 1e-6):
            continue
        rail_flow = rhs - matrix @ pressure
        feasible = all((value == 0 and abs(rail_flow[i]) < 1e-10)
                       or (value == 1 and rail_flow[i] >= -1e-12)
                       or (value == -1 and rail_flow[i] <= 1e-12)
                       for i, value in enumerate(status))
        if feasible:
            relief = np.array([max(rail_flow[i], 0.0) if value == 1 else 0.0 for i, value in enumerate(status)])
            makeup = np.array([max(-rail_flow[i], 0.0) if value == -1 else 0.0 for i, value in enumerate(status)])
            return np.clip(pressure, 0.0, ps), relief, makeup
    raise ArithmeticError("No feasible hydraulic pressure active set")


class TrackDriveBank:
    """Left/right hydraulic motors; SI units and positive shaft conventions.

    evaluate() does not mutate state. Apply item['effort'] to the respective
    sprocket joints for one dt, then advance(item, dt) exactly once.
    """

    def __init__(self, capacity_scale=1.0, initial_pressure_Pa=None):
        if not np.isfinite(capacity_scale) or capacity_scale <= 0:
            raise ValueError("capacity_scale must be finite and positive")
        self.p = dict(PARAMETERS)
        self.ps = self.p["supply_pressure_Pa"] * float(capacity_scale)
        self.qmax = self.p["valve_max_flow_m3_s"] * float(capacity_scale)
        self.pump_per_drive = self.p["pump_max_flow_per_drive_m3_s"] * float(capacity_scale)
        self.pump_bank = self.p["pump_max_flow_bank_m3_s"] * float(capacity_scale)
        self.C = self.p["effective_chamber_volume_m3"] / self.p["bulk_modulus_Pa"]
        self.D = self.p["motor_displacement_m3_rev"] * self.p["gear_ratio"] / (2 * np.pi)
        self.ideal_torque_limit_Nm = self.D * self.ps
        self.pressure = np.full((2, 2), min(self.p["initial_charge_pressure_Pa"], self.ps)) if initial_pressure_Pa is None else np.array(initial_pressure_Pa, dtype=float)
        if self.pressure.shape != (2, 2) or not np.all(np.isfinite(self.pressure)) or np.any(self.pressure < 0) or np.any(self.pressure > self.ps):
            raise ValueError("Initial pressure must be a 2x2 finite array within supply rails")
        self.elapsed_s = 0.0

    def stored_energy(self):
        return float(0.5 * self.C * np.sum(self.pressure**2))

    def evaluate(self, speed_rad_s, command_torque_Nm, dt):
        speed = np.asarray(speed_rad_s, dtype=float)
        command = np.asarray(command_torque_Nm, dtype=float)
        if speed.shape != (2,) or command.shape != (2,) or not np.all(np.isfinite(speed)) or not np.all(np.isfinite(command)):
            raise ValueError("Speed and torque command must be finite two-element arrays")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        old = self.pressure.copy()
        k = self.p["cross_port_leakage_m3_s_Pa"]
        friction = self.p["output_coulomb_friction_Nm"] * np.tanh(speed / self.p["friction_smoothing_speed_rad_s"]) + self.p["output_viscous_friction_Nm_s_rad"] * speed
        desired_hydraulic = command + friction
        desired_difference = np.clip(desired_hydraulic / self.D, -self.ps, self.ps)
        # Backpressure required by a finite return orifice, plus charge margin.
        low = np.minimum(self.ps, self.p["initial_charge_pressure_Pa"] + self.ps * (self.D * np.abs(speed) / self.qmax)**2)
        target = np.column_stack((low + np.maximum(desired_difference, 0), low + np.maximum(-desired_difference, 0)))
        target = np.clip(target, 0, self.ps)
        old_leak = k * (old[:, 0] - old[:, 1])
        displacement_flow = np.column_stack((self.D * speed, -self.D * speed))
        request = displacement_flow + np.column_stack((old_leak, -old_leak)) + self.C * (target - old) / self.p["pressure_response_s"]
        capacity = self.qmax * np.sqrt(np.maximum(np.where(request >= 0, self.ps - old, old), 0) / self.ps)
        flow = np.clip(request, -capacity, capacity)
        pump_saturated = False
        for drive in range(2):
            positive = flow[drive] > 0
            total = float(flow[drive, positive].sum())
            if total > self.pump_per_drive:
                flow[drive, positive] *= self.pump_per_drive / total
                pump_saturated = True
        total = float(np.maximum(flow, 0).sum())
        if total > self.pump_bank:
            flow[flow > 0] *= self.pump_bank / total
            pump_saturated = True
        next_pressure = np.empty((2, 2))
        relief_flow = np.empty((2, 2))
        makeup_flow = np.empty((2, 2))
        for drive in range(2):
            next_pressure[drive], relief_flow[drive], makeup_flow[drive] = _bounded_pressure(old[drive], flow[drive], speed[drive], dt, self.C, self.D, k, self.ps)
        difference = next_pressure[:, 0] - next_pressure[:, 1]
        hydraulic_torque = self.D * difference
        effort = hydraulic_torque - friction
        supply = self.ps * np.maximum(flow, 0).sum(axis=1)
        throttle = supply - np.sum(next_pressure * flow, axis=1)
        leakage = k * difference**2
        relief = self.ps * relief_flow.sum(axis=1)
        friction_power = friction * speed
        mechanical = effort * speed
        integration_loss = self.C * np.sum((next_pressure - old)**2, axis=1) / (2 * dt)
        stored_change = self.C * np.sum(next_pressure**2 - old**2, axis=1) / (2 * dt)
        power_arrays = dict(supply=supply, throttle=throttle, leakage=leakage, relief=relief,
                            friction=friction_power, mechanical=mechanical,
                            hydraulic_mechanical=hydraulic_torque * speed,
                            numerical_storage_loss=integration_loss, fluid_storage_rate=stored_change)
        power = {name: float(value.sum()) for name, value in power_arrays.items()}
        identity = stored_change - (supply - throttle - leakage - relief - friction_power - mechanical - integration_loss)
        # Differential BE identity uses the end-state derivative and has no
        # discrete integration-loss term; provided independently for auditing.
        differential_rate = self.C * np.sum(next_pressure * (next_pressure - old), axis=1) / dt
        differential_identity = differential_rate - (supply - throttle - leakage - relief - hydraulic_torque * speed)
        return dict(effort=effort, hydraulic_torque=hydraulic_torque, friction_torque=friction,
                    pressure_old=old, pressure_next=next_pressure,
                    pressure_derivative=(next_pressure - old) / dt, flow=flow,
                    relief_flow=relief_flow, makeup_flow=makeup_flow,
                    target_pressure=target, power=power, power_per_drive=power_arrays,
                    fluid_identity_W=float(np.max(np.abs(identity))),
                    differential_identity_W=float(np.max(np.abs(differential_identity))),
                    pressure_limit_active=bool(np.any(np.abs(desired_hydraulic) >= self.ideal_torque_limit_Nm) or np.any(relief_flow > 0) or np.any(makeup_flow > 0)),
                    valve_saturated=bool(np.any(np.abs(request) > capacity + 1e-15)),
                    pump_saturated=pump_saturated, dt=float(dt), elapsed_start_s=self.elapsed_s)

    def advance(self, item, dt):
        if not np.isclose(float(dt), item["dt"], rtol=0, atol=1e-15):
            raise ValueError("advance dt differs from evaluated dt")
        if item["elapsed_start_s"] != self.elapsed_s or not np.array_equal(item["pressure_old"], self.pressure):
            raise ValueError("Stale drive evaluation or repeated advance")
        self.pressure[:] = item["pressure_next"]
        self.elapsed_s += float(dt)


def run_validation():
    """Deterministic component evidence; does not validate tracked locomotion."""
    checks = []
    scenarios = []
    def check(name, passed, value, limit):
        checks.append(dict(name=name, passed=bool(passed), value=float(value), limit=float(limit)))
    for dt in (0.001, 0.0005):
        for scenario in ("blocked_shaft", "free_spin", "reverse_and_brake", "externally_oversped"):
            bank = TrackDriveBank()
            initial_energy = bank.stored_energy()
            speed = np.zeros(2)
            inertia = 1200.0  # Declared test flywheel, not excavator equivalent inertia.
            totals = {name: 0.0 for name in ("supply", "throttle", "leakage", "relief", "friction", "mechanical", "numerical_storage_loss")}
            max_residual = 0.0
            minimum_loss = 0.0
            peak_supply = 0.0
            peak_pressure = 0.0
            max_flow = 0.0
            saturated_steps = 0
            negative_work = 0.0
            mechanical_midpoint_error = 0.0
            peak_speed = 0.0
            peak_torque = 0.0
            for index in range(round(3.0 / dt)):
                t = index * dt
                if scenario == "blocked_shaft":
                    speed[:] = 0
                    command = np.array([1e9, -1e9])
                elif scenario == "externally_oversped":
                    speed[:] = [8.0, -8.0]
                    command = np.zeros(2)
                elif scenario == "reverse_and_brake":
                    command = np.full(2, 18000.0 if t < 1.0 else (-18000.0 if t < 2.0 else 0.0))
                else:
                    command = np.full(2, 18000.0)
                item = bank.evaluate(speed, command, dt)
                for name in totals:
                    totals[name] += item["power"][name] * dt
                negative_work += np.maximum(-item["effort"] * speed, 0).sum() * dt
                losses = [item["power"][name] for name in ("throttle", "leakage", "relief", "friction", "numerical_storage_loss")]
                minimum_loss = min(minimum_loss, *losses)
                max_residual = max(max_residual, item["fluid_identity_W"], item["differential_identity_W"])
                peak_supply = max(peak_supply, item["power"]["supply"])
                peak_pressure = max(peak_pressure, float(item["pressure_next"].max()))
                max_flow = max(max_flow, float(np.maximum(item["flow"], 0).sum()))
                peak_torque = max(peak_torque, float(np.max(np.abs(item["hydraulic_torque"]))))
                saturated_steps += int(item["valve_saturated"] or item["pressure_limit_active"] or item["pump_saturated"])
                if scenario in ("free_spin", "reverse_and_brake"):
                    next_speed = speed + dt * item["effort"] / inertia
                    midpoint_work = float(item["effort"] @ (0.5 * (speed + next_speed))) * dt
                    mechanical_midpoint_error += midpoint_work - item["power"]["mechanical"] * dt
                    speed = next_speed
                peak_speed = max(peak_speed, float(np.max(np.abs(speed))))
                bank.advance(item, dt)
            net = totals["supply"] - sum(totals[key] for key in ("throttle", "leakage", "relief", "friction", "mechanical", "numerical_storage_loss"))
            residual_J = bank.stored_energy() - initial_energy - net
            label = f"{scenario}_dt_{dt:g}"
            check(label + "_pressure_bound", peak_pressure <= bank.ps + 1e-6, peak_pressure, bank.ps)
            check(label + "_supply_power_bound", peak_supply <= bank.ps * bank.pump_bank + 1e-6, peak_supply, bank.ps * bank.pump_bank)
            check(label + "_flow_bound", max_flow <= bank.pump_bank + 1e-12, max_flow, bank.pump_bank)
            check(label + "_nonnegative_losses", minimum_loss >= -1e-8, minimum_loss, -1e-8)
            check(label + "_energy_identity_W", max_residual <= 1e-5, max_residual, 1e-5)
            check(label + "_integrated_energy_J", abs(residual_J) <= 1e-5, abs(residual_J), 1e-5)
            check(label + "_hydraulic_torque_bound", peak_torque <= bank.ideal_torque_limit_Nm + 1e-6, peak_torque, bank.ideal_torque_limit_Nm)
            if scenario == "blocked_shaft":
                check(label + "_zero_shaft_work", abs(totals["mechanical"]) < 1e-12, abs(totals["mechanical"]), 1e-12)
                check(label + "_command_saturation_exercised", saturated_steps > 0, saturated_steps, 0)
            if scenario == "reverse_and_brake":
                check(label + "_negative_shaft_work_exercised", negative_work > 100.0, negative_work, 100.0)
            if scenario == "externally_oversped":
                check(label + "_overspeed_braking", totals["mechanical"] < -100.0, totals["mechanical"], -100.0)
                check(label + "_relief_exercised", totals["relief"] > 100.0, totals["relief"], 100.0)
            scenarios.append(dict(name=label, energies_J=totals, fluid_energy_change_J=bank.stored_energy()-initial_energy,
                                  energy_residual_J=residual_J, peak_supply_W=peak_supply, peak_speed_rad_s=peak_speed,
                                  final_speed_rad_s=speed.tolist(), negative_shaft_work_J=float(negative_work),
                                  flywheel_partitioned_coupling_error_J=mechanical_midpoint_error))
    # Deterministic simultaneous arbitrary state/command stress, including rails.
    rng = np.random.default_rng(26026)
    max_stress_residual = 0.0
    for _ in range(500):
        pressure = rng.uniform(0, PARAMETERS["supply_pressure_Pa"], (2, 2))
        bank = TrackDriveBank(initial_pressure_Pa=pressure)
        item = bank.evaluate(rng.uniform(-12, 12, 2), rng.uniform(-1e6, 1e6, 2), rng.choice([.00025, .0005, .001, .01]))
        max_stress_residual = max(max_stress_residual, item["fluid_identity_W"], item["differential_identity_W"])
    check("randomized_500_state_energy_identity_W", max_stress_residual <= 1e-5, max_stress_residual, 1e-5)
    return dict(parameters=PARAMETERS, scope="Hydraulic component validation only; no machine calibration claim",
                passed=all(row["passed"] for row in checks), passed_count=sum(row["passed"] for row in checks),
                check_count=len(checks), checks=checks, scenarios=scenarios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="JSON", help="Write deterministic component validation report")
    args = parser.parse_args()
    if args.validate:
        report = run_validation()
        path = Path(args.validate)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Hydraulic track drive validation: {report['passed_count']}/{report['check_count']} checks passed")
        raise SystemExit(0 if report["passed"] else 1)
    parser.print_help()
