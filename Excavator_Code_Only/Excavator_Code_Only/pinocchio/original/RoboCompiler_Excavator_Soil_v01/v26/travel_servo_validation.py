"""Independent flywheel stability checks for the production hydraulic speed servo.

This checks the actual TrackDriveBank and ShaftSpeedPI on a known mechanical
plant: two independent I*dw/dt=tau flywheels. It is not a tracked-locomotion
surrogate. In particular, 20 kg m2 represents a low-inertia stress condition
close to one free sprocket plus its reflected explicit rotor; 1200 kg m2 is a
separate heavy load case, not an identified excavator equivalent inertia.

The fixed 1 kHz speed loop is held across physics steps. The 100 Hz diagnostic
preserves the previously observed instability. Acceptance thresholds and the
8-second reference are declared here before running the checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from track_drive import TrackDriveBank
from track_servo import ShaftSpeedPI


PROTOCOL = {
    "physics_steps_s": [0.001, 0.0005],
    "production_servo_period_s": 0.001,
    "diagnostic_servo_period_s": 0.01,
    "inertias_kg_m2": [20., 1200.],
    "duration_s": 8.,
    "reference_rad_s": [[0., 0.], [.2, .65], [3., -.35]],
    "final_rms_window_s": [7., 8.],
    "final_rms_limit_rad_s": .02,
    "power_identity_tolerance_W": 1e-5,
    "integrated_energy_tolerance_J": 1e-5,
    "loss_nonnegativity_tolerance_W": 1e-8,
    "coupling_convergence_max_fine_coarse_ratio": .75,
}
LOSS_KEYS = ("throttle", "leakage", "relief", "friction", "numerical_storage_loss")


def flywheel_case(dt, inertia, servo_period):
    bank = TrackDriveBank()
    servo = ShaftSpeedPI()
    speed = np.zeros(2)
    command = np.zeros(2)
    initial_fluid = bank.stored_energy()
    work = {key: 0. for key in ("supply", "mechanical", *LOSS_KEYS)}
    mid_work = 0.
    rms_error_squared = []
    final_speeds = []
    peak = dict(pressure=0., supply=0., flow=0., hydraulic_torque=0.,
                speed=0., command=0., identity=0.)
    minimum_loss = 0.
    minimum_pressure = float(bank.pressure.min())
    max_mechanical_identity = 0.
    period_steps = round(servo_period / dt)
    if not np.isclose(period_steps * dt, servo_period, rtol=0., atol=1e-14):
        raise ValueError("Servo period must be an integer multiple of physics dt")
    for i in range(round(PROTOCOL['duration_s'] / dt)):
        time = i * dt
        desired = np.full(2, 0. if time < .2 else (.65 if time < 3. else -.35))
        if i % period_steps == 0:
            command = servo.evaluate(desired, speed, servo_period)
        item = bank.evaluate(speed, command, dt)
        next_speed = speed + dt * item['effort'] / inertia
        # Exact work of a constant torque on this known inertia over one step.
        mechanical_work = float(item['effort'] @ ((speed + next_speed) / 2.)) * dt
        kinetic_change = float(.5 * inertia * (next_speed @ next_speed - speed @ speed))
        max_mechanical_identity = max(max_mechanical_identity, abs(kinetic_change - mechanical_work))
        mid_work += mechanical_work
        for key in work:
            work[key] += dt * item['power'][key]
        minimum_loss = min(minimum_loss, *(item['power'][key] for key in LOSS_KEYS))
        minimum_pressure = min(minimum_pressure, float(item['pressure_next'].min()))
        for key, value in dict(pressure=float(item['pressure_next'].max()),
                               supply=item['power']['supply'],
                               flow=float(np.maximum(item['flow'], 0.).sum()),
                               hydraulic_torque=float(np.max(abs(item['hydraulic_torque']))),
                               speed=float(np.max(abs(next_speed))),
                               command=float(np.max(abs(command))),
                               identity=max(item['fluid_identity_W'], item['differential_identity_W'])).items():
            peak[key] = max(peak[key], value)
        speed = next_speed
        bank.advance(item, dt)
        if time + dt >= PROTOCOL['final_rms_window_s'][0]:
            rms_error_squared.append(float(np.mean((speed - desired)**2)))
            final_speeds.append(speed.copy())
    fluid_change = bank.stored_energy() - initial_fluid
    kinetic_final = float(.5 * inertia * (speed @ speed))
    source_minus_losses = work['supply'] - sum(work[key] for key in LOSS_KEYS)
    hydraulic_residual = fluid_change - (source_minus_losses - work['mechanical'])
    coupling = mid_work - work['mechanical']
    combined_before_coupling = kinetic_final + fluid_change - source_minus_losses
    result = dict(dt_s=dt, servo_period_s=servo_period, inertia_kg_m2=inertia,
                  final_rms_error_rad_s=float(np.sqrt(np.mean(rms_error_squared))),
                  final_speed_range_rad_s=[float(np.min(final_speeds)), float(np.max(final_speeds))],
                  final_speed_rad_s=speed.tolist(), peak=peak,
                  minimum_loss_W=minimum_loss, minimum_pressure_Pa=minimum_pressure,
                  work_J=work, exact_flywheel_midpoint_work_J=mid_work,
                  kinetic_energy_change_J=kinetic_final, fluid_energy_change_J=fluid_change,
                  hydraulic_integrated_energy_residual_J=hydraulic_residual,
                  mechanical_integrated_energy_residual_J=kinetic_final - mid_work,
                  mechanical_single_step_residual_max_J=max_mechanical_identity,
                  partition_coupling_J=coupling,
                  combined_balance_before_partition_coupling_J=combined_before_coupling,
                  combined_balance_after_partition_coupling_J=combined_before_coupling - coupling,
                  coupling_explanation="Actual constant-torque flywheel work minus the hydraulic bank's held-start-speed work. This signed partition error is reported, not counted as a physical loss.")
    checks = []

    def check(name, value, limit, passed):
        checks.append(dict(name=name, value=float(value), limit=float(limit), passed=bool(passed)))

    check('final_rms_tracking', result['final_rms_error_rad_s'], .02, result['final_rms_error_rad_s'] <= .02)
    check('pressure_upper', peak['pressure'], bank.ps, peak['pressure'] <= bank.ps + 1e-6)
    check('pressure_lower', minimum_pressure, 0., minimum_pressure >= -1e-6)
    check('supply_power', peak['supply'], bank.ps * bank.pump_bank, peak['supply'] <= bank.ps * bank.pump_bank + 1e-6)
    check('supply_flow', peak['flow'], bank.pump_bank, peak['flow'] <= bank.pump_bank + 1e-12)
    check('hydraulic_torque', peak['hydraulic_torque'], bank.ideal_torque_limit_Nm, peak['hydraulic_torque'] <= bank.ideal_torque_limit_Nm + 1e-6)
    check('command_torque', peak['command'], 20000., peak['command'] <= 20000. + 1e-8)
    check('nonnegative_losses', minimum_loss, -1e-8, minimum_loss >= -1e-8)
    check('hydraulic_power_identity', peak['identity'], 1e-5, peak['identity'] <= 1e-5)
    for key in ('hydraulic_integrated_energy_residual_J', 'mechanical_integrated_energy_residual_J',
                'combined_balance_after_partition_coupling_J'):
        check(key, abs(result[key]), 1e-5, abs(result[key]) <= 1e-5)
    result['checks'] = checks
    result['passed'] = all(c['passed'] for c in checks)
    return result


def run_validation():
    cases = [flywheel_case(dt, inertia, PROTOCOL['production_servo_period_s'])
             for dt in PROTOCOL['physics_steps_s'] for inertia in PROTOCOL['inertias_kg_m2']]
    diagnostics = [flywheel_case(dt, 20., PROTOCOL['diagnostic_servo_period_s'])
                   for dt in PROTOCOL['physics_steps_s']]
    checks = []
    for case in cases:
        prefix = f"I{case['inertia_kg_m2']:g}_dt{case['dt_s']:g}_"
        checks.extend(dict(c, name=prefix + c['name']) for c in case['checks'])
    for inertia in PROTOCOL['inertias_kg_m2']:
        coarse, fine = [c for c in cases if c['inertia_kg_m2'] == inertia]
        ratio = abs(fine['partition_coupling_J']) / max(abs(coarse['partition_coupling_J']), 1e-12)
        checks.append(dict(name=f'I{inertia:g}_partition_coupling_refines', value=ratio,
                           limit=.75, passed=bool(ratio <= .75)))
    for diagnostic in diagnostics:
        rms = diagnostic['final_rms_error_rad_s']
        checks.append(dict(name=f"dt{diagnostic['dt_s']:g}_old_100Hz_failure_reproduced",
                           value=rms, limit=.02, passed=bool(rms > .02)))
    root = Path(__file__).resolve().parent
    return dict(passed=all(c['passed'] for c in checks), passed_count=sum(c['passed'] for c in checks),
                total=len(checks), protocol=PROTOCOL, scope=__doc__,
                source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('track_drive.py', 'track_servo.py', 'travel_servo_validation.py')},
                cases=cases, old_100Hz_failure_diagnostics=diagnostics, checks=checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results' / 'travel_servo_validation.json')
    args = parser.parse_args()
    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(('PASS' if report['passed'] else 'FAIL'), f"{report['passed_count']}/{report['total']}", args.output)
    for case in report['cases']:
        print(f"I={case['inertia_kg_m2']:g} dt={case['dt_s']:g}: RMS={case['final_rms_error_rad_s']:.6g} rad/s, coupling={case['partition_coupling_J']:.6g} J")
    raise SystemExit(0 if report['passed'] else 1)
