#!/usr/bin/env python3
"""Prepare supplementary figure inputs from this repository's simulation outputs.

Run ``python run_kangaroo_v22.py`` from the repository root first, then:
    python figures/prepare_plot_data.py
    python figures/make_figures.py

The input layout is data/, results/, results/tables/, and source_dynamics.py.
Dependencies: Python 3, NumPy, and SciPy. Inputs are recorded simulation arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Repository root containing code, data/, and generated results/.')
    parser.add_argument('--output-dir', type=Path,
                        help='Prepared data directory; default: SOURCE_ROOT/results/supplement_figures/data.')
    args = parser.parse_args()
    root = args.source_root.resolve()
    out = (args.output_dir or root/'results/supplement_figures/data').resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    series = {}

    def register(path):
        path = path.resolve()
        key = str(path.relative_to(root))
        if key not in sources:
            sources[key] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'bytes': path.stat().st_size}
        return key

    def source(relative):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing input: {path}\nRun python run_kangaroo_v22.py from the repository root first.')
        register(path)
        return path

    def result(name):
        return source(Path('results')/f'{name}.json')

    def read_json(path):
        return json.loads(path.read_text())

    def csv_source(name):
        path = source(Path('results/tables')/f'{name}.csv')
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        return path, rows

    def note(key, unit, paths, computation, sampling):
        series[key] = {'unit': unit, 'sources': [register(p) for p in paths],
                       'computation': computation, 'sampling': sampling}

    cmg_path = source('data/whole_body_cmg.json')
    cmg = read_json(cmg_path)
    order = cmg['independent_ids']
    by_joint = {a['joint']: a for a in cmg['actuators']}
    actuators = [by_joint[j]['id'] for j in order]
    limits = np.array([by_joint[j]['force_bounds_N'] for j in order], float)
    np.testing.assert_allclose(limits[:, 0], -limits[:, 1])
    data = {'motor_bounds': limits[:, 1]}
    note('motor_bounds', 'N', [cmg_path], 'Positive symmetric physical force bound in independent_ids order.', 'constant')

    telemetry_path, telemetry = csv_source('landing_nominal_telemetry')
    col = lambda name: np.array([float(row[name]) for row in telemetry])
    data['contact_t'] = col('time_s')
    data['contact_com'] = np.column_stack([col(f'com_{a}_m') for a in 'xyz'])
    data['contact_tilt_deg'] = np.degrees(col('tilt_rad'))
    data['push_N'] = col('push_Fy_N')
    data['point_gap_m'] = col('loop_gap_m')
    data['universal_dot'] = col('universal_dot')
    data['contact_energy_J'] = col('potential_J') + col('kinetic_J')
    for key, unit, operation in [
        ('contact_t', 's', 'time_s'), ('contact_com', 'm', 'com_x_m, com_y_m, com_z_m; actual world CoM, not pelvis'),
        ('contact_tilt_deg', 'deg', 'tilt_rad * 180/pi'), ('push_N', 'N', 'push_Fy_N, signed world +Y disturbance'),
        ('point_gap_m', 'm', 'loop_gap_m'), ('universal_dot', '1', 'universal_dot'),
        ('contact_energy_J', 'J', 'potential_J + kinetic_J')]:
        note(key, unit, [telemetry_path], operation, 'Saved every 5 ms; original physics step 25 microseconds.')
    np.testing.assert_allclose(np.diff(data['contact_t']), .005, atol=1e-12)
    assert data['contact_t'].shape == (2001,)

    def pivot(rows, name_key, names, value_key):
        mapping = {(float(r['time_s']), r[name_key]): float(r[value_key]) for r in rows}
        if len(mapping) != len(data['contact_t']) * len(names):
            raise ValueError(f'Incomplete/duplicate long table for {value_key}')
        return np.array([[mapping[(float(t), name)] for name in names] for t in data['contact_t']])

    feet_path, feet = csv_source('landing_nominal_feet')
    data['feet_Fz'] = pivot(feet, 'foot', ['left', 'right'], 'Fz_N')
    note('feet_Fz', 'N', [feet_path], 'Fz_N pivoted into [left, right] columns.', '5 ms')
    actuator_path, actuator_rows = csv_source('landing_nominal_actuators')
    for key, column, sign, unit in [
        ('motor_force', 'actual_force_N', 1, 'N'),
        ('motor_positive_work', 'positive_work_J', 1, 'J'),
        ('motor_absorbed_work', 'negative_work_J', -1, 'J'),
        ('motor_net_work', 'work_J', 1, 'J')]:
        data[key] = sign * pivot(actuator_rows, 'actuator', actuators, column)
        note(key, unit, [actuator_path, cmg_path],
             f'{sign} * {column}, pivoted into CMG independent_ids order.',
             'Saved every 5 ms; work was accumulated at each 25 microsecond physics step.')

    ref_path = source('data/contact_reference.npz')
    code_path = source('source_dynamics.py')
    spec = importlib.util.spec_from_file_location('kangaroo_plot_source_dynamics', code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SourceDynamics(cmg, cmg['coordinate_ids'])
    mass = sum(b['mass_kg'] for b in cmg['bodies'])
    with np.load(ref_path) as ref:
        data['ref_t'] = ref['t'].copy()
        ref_com = []
        for q, base, rotvec in zip(ref['q'], ref['base'], ref['rotvec']):
            state = model.evaluate(q, np.zeros(76), np.zeros(3))
            local = sum(b['mass_kg'] * (state['poses'][b['id']][:3, 3]
                       + state['poses'][b['id']][:3, :3] @ b['com_m']) for b in cmg['bodies']) / mass
            ref_com.append(base + Rotation.from_rotvec(rotvec).as_matrix() @ local)
        data['ref_com'] = np.array(ref_com)
    note('ref_t', 's', [ref_path], 't', '25 ms')
    note('ref_com', 'm', [ref_path, cmg_path, code_path],
         'Mass-weighted body COM positions from source kinematics at reference q, transformed by reference pelvis base/rotvec.',
         '25 ms; computed from saved references without simulation.')

    fixed_paths = {}
    for label, filename in [('ff', 'motion_dt_0.0001.npz'), ('pd', 'motion_pd_only.npz')]:
        path = source(Path('results')/filename)
        fixed_paths[label] = path
        with np.load(path) as record:
            if 'fixed_t' not in data:
                data['fixed_t'] = record['time'].copy()
            else:
                np.testing.assert_array_equal(data['fixed_t'], record['time'])
            data[f'fixed_error_{label}_m'] = record['history'][:, 33:45].copy()
            data[f'fixed_force_{label}_N'] = record['history'][:, 9:21].copy()
            data[f'fixed_energy_residual_{label}_J'] = record['balance'].copy()
        for key, unit, operation in [
            (f'fixed_error_{label}_m', 'm', 'history[:,33:45], commanded minus actual motor position'),
            (f'fixed_force_{label}_N', 'N', 'history[:,9:21]'),
            (f'fixed_energy_residual_{label}_J', 'J', 'balance, original physics-rate energy ledger')]:
            note(key, unit, [path], operation, 'Full physics rate, 100 microseconds; fixed-pelvis benchmark.')
    note('fixed_t', 's', list(fixed_paths.values()), 'time, asserted identical between experiments.', '100 microseconds')
    assert data['fixed_t'].shape == (40001,)

    body_path = source('results/landing_nominal_body_forces.npz')
    audit_code_path = source('contact_audit.py')
    with np.load(body_path) as body:
        np.testing.assert_allclose(body['time'], data['contact_t'], atol=1e-12)
        load = sum(body[k] for k in ['actuator_generalized', 'passive_generalized', 'loop_generalized',
                                    'contact_generalized', 'limit_generalized', 'external_generalized'])
        residual = body['inertial_generalized'] + body['bias_generalized'] - load
        data['native_relative_residual'] = np.max(np.abs(residual), axis=1) / np.maximum(1., np.max(np.abs(load), axis=1))
        energy = body['body_energy'].sum(axis=(1, 2)) + body['armature_energy']
        data['body_energy_reconstruction_error_J'] = energy - data['contact_energy_J']
    note('native_relative_residual', '1', [body_path, audit_code_path],
         'max(abs(Ma+b-load))/max(1,max(abs(load))) per snapshot; load=actuator+passive+loop+contact+limit+external, exactly native_equation_relative in contact_audit.py.', '5 ms')
    note('body_energy_reconstruction_error_J', 'J', [body_path, telemetry_path],
         'sum(body_energy over 78 bodies and kinetic/potential components)+armature_energy-contact_energy_J.', '5 ms')

    trial_names = ['landing_nominal', 'landing_refined', 'landing_low_friction', 'landing_higher_drop_push', 'landing_slow_actuators']
    trial_fields = ['name', 'configuration', 'timestep_s', 'duration_s', 'steps', 'mass_kg', 'body_count', 'motor_count',
                    'touchdown_s', 'maximum_loop_gap_m', 'maximum_universal_dot', 'minimum_slide_margin_m',
                    'minimum_hinge_margin_rad', 'maximum_motor_force_N', 'maximum_motor_error_m', 'maximum_penetration_m',
                    'maximum_tilt_deg', 'final_tilt_deg', 'maximum_ground_normal_N', 'mean_final_ground_normal_N',
                    'achieved_crouch_m', 'achieved_lateral_excursion_m', 'maximum_final_base_speed_m_s',
                    'final_position_error_m', 'work_J', 'energy_change_J', 'maximum_energy_ledger_error_J',
                    'final_energy_ledger_error_J', 'maximum_actuator_virtual_work_error_W',
                    'maximum_constraint_power_decomposition_error_W', 'saturated_control_updates', 'warning_count']
    trials = []
    for name in trial_names:
        path = result(name)
        original = read_json(path)
        trial = {key: original[key] for key in trial_fields}
        trial['positive_motor_work_total_J'] = sum(original['positive_motor_work_J'])
        trial['absorbed_motor_work_total_J'] = -sum(original['negative_motor_work_J'])
        trial['source'] = register(path)
        trials.append(trial)

    instantaneous_path = result('instantaneous')
    instant = read_json(instantaneous_path)
    reference_json_path = result('contact_reference')
    audit_path = result('landing_nominal_audit')
    audit = read_json(audit_path)
    suite_path = result('complete_validation_v22')
    suite = read_json(suite_path)
    fixed = {'duration_s': float(data['fixed_t'][-1]), 'timestep_s': float(data['fixed_t'][1]),
             'source': {label: register(p) for label, p in fixed_paths.items()}}
    for label in ['ff', 'pd']:
        errors = data[f'fixed_error_{label}_m']
        forces = data[f'fixed_force_{label}_N']
        fixed[f'rmse_{label}_m'] = float(np.sqrt(np.mean(errors**2)))
        fixed[f'per_port_rmse_{label}_m'] = np.sqrt(np.mean(errors**2, axis=0)).tolist()
        fixed[f'peak_force_{label}_N'] = float(np.max(np.abs(forces)))
        fixed[f'per_port_peak_force_{label}_N'] = np.max(np.abs(forces), axis=0).tolist()
        fixed[f'max_energy_residual_{label}_J'] = float(np.max(np.abs(data[f'fixed_energy_residual_{label}_J'])))
    fixed['rmse_reduction_ratio'] = fixed['rmse_pd_m'] / fixed['rmse_ff_m']
    metrics = {
        'schema_version': 1,
        'motor_order': order, 'actuator_order': actuators,
        'motor_labels': ['L1','L2','L3','L-length','L4','L5','R1','R2','R3','R-length','R4','R5'],
        'motor_bounds_N': data['motor_bounds'].tolist(),
        'foot_order': ['left', 'right'],
        'nominal': trials[0], 'refined': trials[1], 'variants': trials,
        'fixed': fixed,
        'instantaneous': {k: instant[k] for k in ['configurations','gravity_vectors','floating_force_cases','maxima']},
        'contact_reference': {k: v for k, v in read_json(reference_json_path).items() if k != 'elapsed_seconds'},
        'contact_audit': {k: audit[k] for k in ['snapshots','independent_snapshots','errors','status']},
        'complete_suite': {k: suite[k] for k in ['status','gates_passed','gates_total']},
        'computed': {
            'native_relative_residual_max': float(np.max(data['native_relative_residual'])),
            'body_energy_reconstruction_error_max_J': float(np.max(np.abs(data['body_energy_reconstruction_error_J']))),
            'sampled_per_port_peak_motor_force_N': np.max(np.abs(data['motor_force']), axis=0).tolist(),
            'sampled_peak_point_gap_m': float(np.max(data['point_gap_m'])),
            'sampled_peak_universal_dot': float(np.max(data['universal_dot'])),
        },
        'metric_sources': {'instantaneous': register(instantaneous_path), 'contact_reference': register(reference_json_path),
                           'contact_audit': register(audit_path), 'complete_suite': register(suite_path)},
    }
    np.testing.assert_allclose(data['motor_net_work'][-1].sum(), trials[0]['work_J']['motor'], atol=1e-10)
    np.testing.assert_allclose(np.max(data['native_relative_residual']), audit['errors']['native_equation_relative'], rtol=1e-10)
    np.testing.assert_allclose(data['contact_energy_J'][-1]-data['contact_energy_J'][0], trials[0]['energy_change_J'], atol=1e-10)
    for key, arr in data.items():
        assert np.isfinite(arr).all(), f'Nonfinite values: {key}'
        series[key]['shape'] = list(arr.shape)
    np.savez_compressed(out/'plot_data.npz', **data)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    provenance = {
        'schema_version': 1,
        'method': 'Recomputed figure inputs from generated repository results; this script does not run simulations.',
        'source_paths_relative_to': '--source-root',
        'sources': sources, 'series': series,
        'notes': [
            'Contact traces are sampled every 5 ms; trial JSON extrema use the full 25/12.5 microsecond physics steps.',
            'Both reference and native trajectories are world center-of-mass positions. Reference CoM is calculated from CMG body masses and saved compatible reference configurations.',
            'Cumulative motor work in CSV was integrated at physics rate. Do not reconstruct the complete contact energy ledger from decimated power CSV.',
            'Motor bounds are physical per-port CMG bounds: 2000 N except 5000 N for the two leg-length ports.',
            'The nominal body-force NPZ supplies generalized-force balance and independently summed body-energy arrays.',
        ],
        'outputs': {
            'plot_data.npz': {'sha256': hashlib.sha256((out/'plot_data.npz').read_bytes()).hexdigest()},
            'metrics.json': {'sha256': hashlib.sha256((out/'metrics.json').read_bytes()).hexdigest()},
            'prepare_plot_data.py': {'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        },
    }
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'output_dir': str(out), 'arrays': len(data), 'contact_samples': len(data['contact_t']),
                      'reference_samples': len(data['ref_t']), 'fixed_samples': len(data['fixed_t']),
                      'sources': len(sources), 'fixed_rmse_reduction': fixed['rmse_reduction_ratio']}, indent=2))


if __name__ == '__main__':
    main()
