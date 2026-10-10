"""Bind saved rollout evidence to its declared scenario and recompute metrics.

This audit never integrates the equations of motion. Independent native
Pinocchio dynamics and PACDM checks are performed by ``pin_checks``. Here the
entire trajectory is checked against the specified model, reference, control
law, disturbance schedule, and summary, so stale or mislabeled files cannot
silently inherit another case's acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def audit_case(root, name, dt, payload, ff):
    """Return a JSON-safe full-mission evidence audit for one expected case.

The caller supplies the expected scenario, rather than trusting labels in
the files. Numeric evidence comparisons use atol=1e-10, rtol=1e-9. The time
grid, coordinate ordering, model structure, and discrete metadata must match
exactly. Failure to read or interpret an input produces a failed audit.
"""
    root = Path(root)
    paths = {
        'model': root / 'data/stewart.cmg.json',
        'case_model': root / f'results/{name}.cmg.json',
        'trajectory': root / f'results/{name}.npz',
        'summary': root / f'results/{name}.json',
        'reference': root / 'results/reference.npz',
    }
    checks = {}
    details = dict(expected=dict(name=name, timestep_s=dt, duration_s=22.,
                                 payload_mass_kg=payload, feedforward=bool(ff),
                                 disturbance_scale=1.),
                   input_sha256={}, absolute_tolerance=1e-10,
                   relative_tolerance=1e-9, recomputed_summary={})

    def exact(key, valid, **extra):
        checks[key] = dict(passed=bool(valid), **extra)

    def close(key, observed, expected):
        observed = np.asarray(observed)
        expected = np.asarray(expected)
        shape_ok = observed.shape == expected.shape
        finite = (shape_ok and np.all(np.isfinite(observed))
                  and np.all(np.isfinite(expected)))
        valid = finite and np.allclose(observed, expected, atol=1e-10, rtol=1e-9)
        error = float(np.max(np.abs(observed - expected))) if finite else None
        checks[key] = dict(passed=bool(valid), max_absolute_error=error,
                           absolute_tolerance=1e-10, relative_tolerance=1e-9)

    def finish():
        return dict(passed=bool(checks) and all(c['passed'] for c in checks.values()),
                    checks=checks, details=details)

    try:
        for key, path in paths.items():
            exact('input.' + key, path.is_file())
            if path.is_file():
                details['input_sha256'][path.relative_to(root).as_posix()] = (
                    hashlib.sha256(path.read_bytes()).hexdigest())
        if not all(c['passed'] for c in checks.values()):
            return finish()
        base = json.loads(paths['model'].read_text(encoding='utf-8'))
        cmg = json.loads(paths['case_model'].read_text(encoding='utf-8'))
        summary = json.loads(paths['summary'].read_text(encoding='utf-8'))
        with np.load(paths['trajectory'], allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        with np.load(paths['reference'], allow_pickle=False) as z:
            ref = {k: z[k] for k in z.files}

        expected_model = deepcopy(base)
        payload_body = next(b for b in expected_model['bodies'] if b['id'] == 'payload')
        factor = payload / payload_body['mass_kg']
        payload_body['mass_kg'] = payload
        payload_body['inertia_kg_m2'] = [
            [v * factor for v in row] for row in payload_body['inertia_kg_m2']]
        exact('scenario.exact_model', cmg == expected_model)
        ids = base['coordinate_ids']
        active = np.asarray([ids.index(j) for j in base['independent_ids']], dtype=int)
        n = round(22. / dt) + 1
        nt = len(ids)
        na = len(active)
        shapes = {
            'time': (n,), 'q': (n, nt),
            'q_augmented': (n, nt + 3 * len(base['closures'])),
            'velocity': (n, nt), 'acceleration': (n, nt),
            'length_acceleration': (n, na), 'target_pose': (n, 6),
            'force': (n, na), 'wrench': (n, 6), 'actuator_error_m': (n, na),
        }
        scalar_arrays = [
            'pose_error_m', 'angle_error_rad', 'closure_error_m',
            'augmented_closure_residual', 'velocity_closure_residual',
            'acceleration_closure_residual', 'tangent_residual',
            'reduced_equation_residual', 'power_W', 'external_power_W',
            'energy_J', 'reduced_mass_min_eigenvalue', 'mapping_rcond',
            'rank_full', 'rank_passive',
        ]
        shapes.update({k: (n,) for k in scalar_arrays})
        for key, shape in shapes.items():
            exact('array_shape.' + key, key in a and a[key].shape == shape,
                  expected_shape=list(shape),
                  observed_shape=list(a[key].shape) if key in a else None)
        for key in ('coordinate_ids', 'dynamics_backend', 'timestep_s'):
            exact('array_present.' + key, key in a)
        if not all(checks[k]['passed'] for k in checks
                   if k.startswith(('array_shape.', 'array_present.'))):
            return finish()
        finite = all(bool(np.all(np.isfinite(a[k]))) for k in shapes)
        exact('trajectory.finite', finite)
        if not finite:
            return finish()
        time = a['time']
        exact('scenario.time_grid', np.array_equal(time, np.arange(n) * dt)
              and time[0] == 0. and time[-1] == 22.)
        exact('scenario.coordinate_ids', np.array_equal(a['coordinate_ids'], ids))
        exact('scenario.reference_coordinate_ids',
              np.array_equal(ref['coordinate_ids'], ids))
        exact('scenario.npz_timestep', a['timestep_s'].shape == ()
              and float(a['timestep_s']) == dt)
        exact('scenario.backend', a['dynamics_backend'].shape == ()
              and str(a['dynamics_backend']) == 'Pinocchio 3.8.0 + PACDM')
        metadata = dict(
            name=name, timestep_s=dt, duration_s=22., samples=n,
            payload_mass_kg=payload, feedforward=bool(ff), disturbance_scale=1.,
            engine='Pinocchio 3.8.0 tree dynamics + unchanged PACDM reduction',
            integrator='Semi-implicit Euler in six independent leg lengths, PACDM passive assembly every step')
        for key, value in metadata.items():
            exact('summary_metadata.' + key, key in summary and summary[key] == value
                  and (key != 'feedforward' or isinstance(summary[key], bool)))
        exact('summary_metadata.elapsed_s', isinstance(summary.get('elapsed_s'), (int, float))
              and np.isfinite(summary['elapsed_s']) and summary['elapsed_s'] > 0.)
        fallback = summary.get('fallback_count')
        exact('summary_metadata.fallback_count', type(fallback) is int and 0 <= fallback < n)
        details['fallback_count_scope'] = ('The summary fallback counter has no per-step log; '
                                            'only its integer range is checked here.')

        # Independently reconstruct the reference spline at every saved time.
        exact('reference.time_domain', ref['time'][0] == 0.
              and ref['time'][-1] == 22. and np.all(np.diff(ref['time']) > 0.))
        target = CubicSpline(ref['time'], ref['q'])(time)
        target_v = CubicSpline(ref['time'], ref['velocity'])(time)
        feedforward = CubicSpline(ref['time'], ref['feedforward_force'])(time)
        close('trajectory.target_pose', a['target_pose'], target[:, :6])
        close('trajectory.physical_augmented_prefix', a['q_augmented'][:, :nt], a['q'])
        position_error = np.linalg.norm(a['q'][:, :3] - target[:, :3], axis=1)
        desired_rotation = Rotation.from_euler('ZYX', target[:, 3:6])
        actual_rotation = Rotation.from_euler('ZYX', a['q'][:, 3:6])
        angle_error = (desired_rotation.inv() * actual_rotation).magnitude()
        actuator_error = a['q'][:, active] - target[:, active]
        close('trajectory.position_error', a['pose_error_m'], position_error)
        close('trajectory.angle_error', a['angle_error_rad'], angle_error)
        close('trajectory.actuator_error', a['actuator_error_m'], actuator_error)

        # Gains and force limit come from the declared base model, not from
        # the possibly mislabeled case model. Payload does not change them.
        actuation = base['actuation']
        requested = ((feedforward if ff else np.zeros_like(feedforward))
                     - actuation['length_kp_N_m'] * actuator_error
                     + actuation['length_kd_N_s_m']
                     * (target_v[:, active] - a['velocity'][:, active]))
        limit = actuation['force_limit_N']
        expected_force = np.clip(requested, -limit, limit)
        close('trajectory.force_control_law', a['force'], expected_force)
        expected_wrench = np.zeros((n, 6))
        for start, duration, vector in [
                (5.5, .35, [80., 0., 0., 0., 0., 0.]),
                (11.2, .35, [0., 120., 0., 0., 0., 0.]),
                (14.2, .4, [0., 0., 0., 12., 0., 0.])]:
            mask = (time >= start) & (time <= start + duration)
            expected_wrench[mask] += (
                np.sin(np.pi * (time[mask] - start) / duration)[:, None]
                * np.asarray(vector))
        close('trajectory.disturbance_schedule', a['wrench'], expected_wrench)
        actuator_power = np.sum(a['force'] * a['velocity'][:, active], axis=1)
        close('trajectory.actuator_power', a['power_W'], actuator_power)
        yaw, pitch = a['q'][:, 3], a['q'][:, 4]
        yaw_rate, pitch_rate, roll_rate = a['velocity'][:, 3:6].T
        omega = np.column_stack((
            -np.sin(yaw) * pitch_rate + np.cos(yaw) * np.cos(pitch) * roll_rate,
            np.cos(yaw) * pitch_rate + np.sin(yaw) * np.cos(pitch) * roll_rate,
            yaw_rate - np.sin(pitch) * roll_rate))
        external_power = (np.sum(a['wrench'][:, :3] * a['velocity'][:, :3], axis=1)
                          + np.sum(a['wrench'][:, 3:] * omega, axis=1))
        close('trajectory.external_power', a['external_power_W'], external_power)
        for key in ('rank_full', 'rank_passive'):
            exact('trajectory.integer_' + key,
                  np.array_equal(a[key], np.rint(a[key])) and np.all(a[key] >= 0))

        joints = {j['id']: j for j in base['joints']}
        lower = np.asarray([joints[j]['limits']['lower'] for j in ids])
        upper = np.asarray([joints[j]['limits']['upper'] for j in ids])
        margins = np.minimum(a['q'] - lower, upper - a['q'])
        prismatic = [i for i, j in enumerate(ids) if joints[j]['type'] == 'prismatic']
        revolute = [i for i, j in enumerate(ids) if joints[j]['type'] == 'revolute']
        dock = time >= 21.
        metrics = dict(
            rms_position_error_m=float(np.sqrt(np.mean(position_error ** 2))),
            max_position_error_m=float(position_error.max()),
            max_orientation_error_deg=float(np.rad2deg(angle_error.max())),
            docking_max_position_error_m=float(position_error[dock].max()),
            docking_max_orientation_error_deg=float(np.rad2deg(angle_error[dock].max())),
            max_closure_error_m=float(a['closure_error_m'].max()),
            max_actuator_force_N=float(np.max(np.abs(a['force']))),
            max_requested_force_N=float(np.max(np.abs(requested))),
            saturation_samples=int(np.count_nonzero(np.any(np.abs(requested) > limit, axis=1))),
            min_prismatic_limit_margin_m=float(margins[:, prismatic].min()),
            min_revolute_limit_margin_rad=float(margins[:, revolute].min()),
            min_reduced_mass_eigenvalue=float(a['reduced_mass_min_eigenvalue'].min()),
            min_mapping_rcond=float(a['mapping_rcond'].min()),
            min_rank_full=int(a['rank_full'].min()),
            min_rank_passive=int(a['rank_passive'].min()),
            all_finite=finite,
            positive_actuator_work_J=float(np.trapezoid(np.maximum(actuator_power, 0.), time)),
            energy_work_defect_J=float(a['energy_J'][-1] - a['energy_J'][0]
                                      - np.trapezoid(actuator_power + external_power, time)))
        for key in ('augmented_closure_residual', 'velocity_closure_residual',
                    'acceleration_closure_residual', 'tangent_residual',
                    'reduced_equation_residual'):
            metrics['max_' + key] = float(a[key].max())
        details['recomputed_summary'] = metrics
        for key, expected in metrics.items():
            if key not in summary:
                exact('summary_metric.' + key, False, reason='Missing summary field')
            elif isinstance(expected, (bool, int)):
                exact('summary_metric.' + key, type(summary[key]) is type(expected)
                      and summary[key] == expected)
            else:
                close('summary_metric.' + key, summary[key], expected)
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration) as exc:
        exact('audit.input_interpretation', False,
              reason=f'{type(exc).__name__}: {exc}')
    return finish()
