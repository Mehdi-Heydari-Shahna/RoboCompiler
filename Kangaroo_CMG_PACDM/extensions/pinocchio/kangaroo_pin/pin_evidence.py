"""Bind saved Kangaroo rollouts to their declared scenario and recompute metrics.

This audit never integrates the equations of motion.  The caller supplies the
expected scenario (case table), rather than trusting labels inside the files.
The entire saved trajectory is replayed through the stated control law, the
exact actuator filter, the disturbance schedule, the motor/pelvis integration
rule and the energy ledger; every summary metric is recomputed from the
arrays.  Stale, truncated or mislabelled files therefore cannot silently
inherit another case's acceptance.  Native dynamics checks are in
``pin_checks.audit_rollout``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from .pin_simulation import (ACT_RANGE_N, PIN_SETTINGS, SAMPLE_PERIOD_S, Controller,
                             case_metrics, code_sha256, load_reference)

ATOL, RTOL = 1e-10, 1e-9


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case(root, plant, case):
    """Return a JSON-safe evidence audit for one expected case record."""
    root = Path(root)
    name = case['name']
    paths = dict(trajectory=root / 'results' / f'{name}.npz', summary=root / 'results' / f'{name}.json',
                 reference=plant.reference_path, model=root / 'original_v22/data/whole_body_cmg.json')
    checks = {}
    details = dict(expected=case, input_sha256={}, absolute_tolerance=ATOL, relative_tolerance=RTOL)

    def exact(key, valid, **extra):
        checks[key] = dict(passed=bool(valid), **extra)

    def close(key, observed, expected, atol=ATOL, rtol=RTOL):
        observed = np.asarray(observed, dtype=float)
        expected = np.asarray(expected, dtype=float)
        ok = observed.shape == expected.shape and np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))
        error = float(np.max(np.abs(observed - expected))) if ok and observed.size else (0. if ok else None)
        checks[key] = dict(passed=bool(ok and np.allclose(observed, expected, atol=atol, rtol=rtol)),
                           max_absolute_error=error, absolute_tolerance=atol, relative_tolerance=rtol)

    def finish():
        return dict(passed=bool(checks) and all(c['passed'] for c in checks.values()),
                    checks=checks, details=details)

    try:
        for key, path in paths.items():
            exact('input.' + key, path.is_file())
            if path.is_file():
                details['input_sha256'][key] = _sha(path)
        if not all(c['passed'] for c in checks.values()):
            return finish()
        summary = json.loads(paths['summary'].read_text())
        with np.load(paths['trajectory'], allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        reference = load_reference(paths['reference'])
        cfg = {**plant.A.DEFAULTS, **case.get('config', {})}
        dt = case['dt']
        duration = case['duration']
        steps = round(duration / dt)
        n = len(a['time'])
        complete = n == steps + 1 and summary.get('completed_steps') == steps and summary.get('stop_reason') is None
        if case.get('may_stop_early'):
            # Negative control only: the run may end when PACDM leaves the admissible branch.
            stopped = (n < steps + 1 and summary.get('completed_steps') == n - 1
                       and str(summary.get('stop_reason', '')).startswith('PACDM assembly stopped'))
            exact('scenario.complete_or_declared_branch_exit', complete or stopped,
                  completed_steps=n - 1, stop_reason=summary.get('stop_reason'))
        else:
            exact('scenario.complete_duration', complete)
        exact('provenance.code_sha256', summary.get('code_sha256') == code_sha256())
        for key, value in dict(name=name, timestep_s=dt, duration_s=duration, configuration=cfg,
                               no_contact=case.get('no_contact', False),
                               passive_control=case.get('passive', False), open_tree=False).items():
            exact('summary_metadata.' + key, summary.get(key) == value)
        exact('summary_metadata.pinocchio_settings', summary.get('pinocchio_settings') == PIN_SETTINGS)
        exact('scenario.backend_label', str(a['dynamics_backend']) ==
              f'Pinocchio {pin.__version__} + unchanged PACDM + native PGS/ADMM contact')
        exact('scenario.timestep', float(a['timestep_s']) == dt)
        exact('scenario.coordinate_ids', list(a['coordinate_ids']) == plant.ids)
        exact('scenario.actuator_ids', list(a['actuator_ids']) == [x['id'] for x in plant.cmg['actuators']])
        exact('scenario.corner_names', list(a['corner_names']) == plant.corner_names)
        shapes = dict(time=(n,), base=(n, 7), motor=(n, plant.na), xi=(n, 6 + plant.na),
                      act=(n, plant.na), command=(n, plant.na), contact_seen=(n,),
                      lam=(n, 3 * len(plant.corner_names)), gap=(n, len(plant.corner_names)),
                      push=(n, 3), kinetic=(n,), potential=(n,), armature_kinetic=(n,),
                      work_step=(n, 6), pacdm_closure=(n,), tangent=(n,), acceleration_closure=(n,),
                      rcond=(n,), rank=(n, 2), reduced_equation=(n,), contact_solver=(n,),
                      contact_iterations=(n,), ncp=(n, 3), tilt=(n,), com=(n, 3),
                      linear_momentum=(n, 3), angular_momentum=(n, 3), foot_position=(n, 2, 3),
                      torso_com=(n, 3), saturated=(n,), contact_moment=(n, 3), energy=(n,),
                      work=(n, 6), ledger=(n,), polish_shift=(n,))
        for key, shape in shapes.items():
            exact('array_shape.' + key, key in a and a[key].shape == shape,
                  observed=list(a[key].shape) if key in a else None)
        if not all(checks[k]['passed'] for k in checks if k.startswith('array_shape.')):
            return finish()
        finite = all(bool(np.all(np.isfinite(a[k]))) for k in shapes)
        exact('trajectory.finite', finite)
        if not finite:
            return finish()
        time_ = a['time']
        exact('scenario.time_grid', np.array_equal(time_, np.arange(n) * dt))
        stored = a['stored_step']
        exact('stored_states.every_5_ms', set(range(0, n, round(SAMPLE_PERIOD_S / dt))) <= set(stored.tolist()))
        exact('stored_states.consistent_motor',
              bool(np.array_equal(a['stored_z'][:, plant.active], a['motor'][stored])))
        # --- control law replay (1 kHz command, clip, slew, feedforward gating)
        control_stride = round(cfg['control_period_s'] / dt)
        controller = Controller(reference, cfg, plant.na, plant.force_bounds)
        command = np.zeros(plant.na)
        contact_seen = False
        seen_log, command_log, saturated_log = [], [], []
        for k in range(n):
            if k > 0:
                contact_seen = bool(np.any(a['lam'][k - 1, 2::3] > 0.)
                                    or np.any(a['gap'][k - 1] <= PIN_SETTINGS['contact_detection_gap_m']))
            saturated = False
            if k % control_stride == 0:
                command, saturated = controller.update(time_[k], a['motor'][k], a['xi'][k, 6:],
                                                       contact_seen, case.get('passive', False))
            seen_log.append(contact_seen)
            command_log.append(command.copy())
            saturated_log.append(saturated)
        exact('control.contact_flag_replay', np.array_equal(np.asarray(seen_log), a['contact_seen']))
        close('control.command_replay', a['command'], np.asarray(command_log), atol=1e-9)
        exact('control.saturation_flags', np.array_equal(np.asarray(saturated_log), a['saturated']))
        # --- exact first-order force response (filterexact) with bounds
        filt = 1. - np.exp(-dt / cfg['actuator_time_constant_s'])
        bounds = plant.force_bounds
        act = np.zeros(plant.na)
        act_log = [act.copy()]
        for k in range(n - 1):
            act = np.clip(act + (np.clip(a['command'][k], bounds[:, 0], bounds[:, 1]) - act) * filt,
                          -ACT_RANGE_N, ACT_RANGE_N)
            act_log.append(act.copy())
        close('actuator.filter_replay', a['act'], np.asarray(act_log), atol=1e-9)
        force = np.clip(a['act'], bounds[:, 0], bounds[:, 1])
        exact('actuator.within_source_bounds', bool(np.all(force >= bounds[:, 0] - 1e-12)
                                                     and np.all(force <= bounds[:, 1] + 1e-12)))
        # --- disturbance schedule
        close('disturbance.schedule', a['push'], np.array([plant.A.external_push(t, cfg) for t in time_]))
        # --- state integration rule
        if n > 1:
            close('integration.motor_update', a['motor'][1:], a['motor'][:-1] + dt * a['xi'][1:, 6:], atol=1e-12)
            predicted = np.array([pin.SE3ToXYZQUAT(pin.XYZQUATToSE3(b) * pin.exp6(pin.Motion(x[:6] * dt)))
                                  for b, x in zip(a['base'][:-1], a['xi'][1:])])
            close('integration.pelvis_update', a['base'][1:], predicted, atol=1e-12)
        # --- energy ledger and motor work channel
        energy = a['kinetic'] + a['potential'] + a['armature_kinetic']
        close('energy.stored_sum', a['energy'], energy)
        work = np.vstack([np.zeros(6), np.cumsum(a['work_step'][:-1], axis=0)])
        close('energy.work_accumulation', a['work'], work)
        close('energy.ledger', a['ledger'], energy - energy[0] - work.sum(axis=1))
        nxt = np.vstack([a['xi'][1:, 6:], a['final_xi_next'][None, 6:]])
        close('energy.motor_channel', a['work_step'][:, 0], dt * np.sum(force * .5 * (a['xi'][:, 6:] + nxt), axis=1))
        exact('energy.ideal_loop_and_limit_channels_zero', bool(np.all(a['work_step'][:, 2] == 0.)
                                                                and np.all(a['work_step'][:, 4] == 0.)))
        # --- contact solver log
        allowed = {-1} if case.get('no_contact') else {0, 1}
        exact('contact.solver_codes', set(np.unique(a['contact_solver']).tolist()) <= allowed)
        exact('contact.ncp_within_acceptance', bool(np.max(a['ncp']) <= PIN_SETTINGS['ncp_acceptance']))
        # --- summary metrics recomputed from arrays
        recomputed = case_metrics(plant, a, reference, cfg, dt, duration)
        details['recomputed_summary'] = recomputed
        for key, value in recomputed.items():
            if key not in summary:
                exact('summary_metric.' + key, False, reason='missing')
            elif value is None or isinstance(value, (bool, int, str)):
                exact('summary_metric.' + key, summary[key] == value)
            elif isinstance(value, dict):
                close('summary_metric.' + key, [summary[key][k] for k in value], list(value.values()))
            else:
                close('summary_metric.' + key, summary[key], value)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        exact('audit.input_interpretation', False, reason=f'{type(exc).__name__}: {exc}')
    return finish()
