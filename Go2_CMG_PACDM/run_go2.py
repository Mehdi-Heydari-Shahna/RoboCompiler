"""Reproduce the Unitree Go2 CMG/PACDM contact course and saved evidence.

Run --render --workers 3 to generate and validate all seven cases. After a run,
use --audit-existing to verify generated hashes and recorded acceptance,
or --replay to view recorded native MuJoCo configurations interactively.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from go2.model import build_model, save_model
from go2.simulation import run_case
from go2.task import DURATION, make_reference
from go2.task_validation import validate_reference
from go2.validation import validate_mechanics, validate_contacts

ROOT = Path(__file__).resolve().parent
RUN_MANIFEST = Path('results')/'SHA256SUMS.json'
CASES = {
    'nominal': {},
    'fine': {'dt': .0005},
    'low_friction': {'friction': .55},
    'payload': {'payload': 1.5},
    'strong_push': {'push': 1.25},
    'no_actuation': {'actuation': False},
    'PD_ablation': {'feedforward': False},
}
POSITIVE = ['nominal', 'fine', 'low_friction', 'payload', 'strong_push']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def aggregate(root=ROOT):
    """Evaluate the declared acceptance gates without changing their limits."""
    root = Path(root); results = root/'results'
    mechanics = read_json(results/'mechanics_validation.json')
    contacts = read_json(results/'contact_validation.json')
    reference = read_json(results/'task_validation.json')
    cases = {name: read_json(results/f'{name}.json') for name in CASES}
    checks = {'mechanics.'+key: value for key, value in mechanics['checks'].items()}
    checks.update({'contacts.'+key: value for key, value in contacts['checks'].items()})
    checks.update({'reference.'+item['name']: {k: v for k, v in item.items() if k != 'name'}
                   for item in reference['checks']})

    def gate(name, value, limit, unit='dimensionless', relation='<='):
        value = float(value)
        ok = bool(np.isfinite(value) and (value <= limit if relation == '<=' else
                                          value >= limit if relation == '>=' else value == limit))
        checks[name] = dict(passed=ok, value=value if np.isfinite(value) else None,
                            limit=float(limit), unit=unit, relation=relation)

    observed_modes = {}
    for name, case in cases.items():
        gate(name+'.numerical_warnings', case['numerical_warnings'], 0, 'count', '==')
        # Motor torque requested by the controller before its safety clip (the
        # applied, clipped torque is within the caps by construction). The QP
        # imposes the caps as constraints, solved to the same 0.005 tolerance
        # that max_qp_violation accepts, so a request may exceed a cap by at
        # most that amount.
        gate(name+'.source_motor_caps', case['peak_torque_demand_excess_Nm'], .005, 'N m')
        if name in POSITIVE:
            gate(name+'.completed', case['completed'], 1, 'boolean', '==')
            gate(name+'.duration_error', abs(case['simulated_s']-DURATION), .001, 's')
            for field, limit, unit, relation in [
                ('final_position_error_m', .025, 'm', '<='),
                ('final_yaw_error_rad', .08, 'rad', '<='),
                ('rms_body_error_m', .035, 'm', '<='),
                ('peak_body_error_m', .10, 'm', '<='),
                ('max_tilt_rad', .25, 'rad', '<='),
                ('min_base_height_m', .18, 'm', '>='),
                ('min_joint_margin_rad', 0., 'rad', '>='),
                ('unexpected_contact_instances', 0., 'contact instances', '=='),
                ('final_speed_m_s', .03, 'm/s', '<='),
                ('max_qp_violation', .005, 'mixed constraint SI', '<='),
            ]:
                gate(name+'.'+field, case[field], limit, unit, relation)
            with np.load(results/f'{name}.npz', allow_pickle=False) as saved:
                after_start = saved['time'] >= 2.
                modes = sorted(set(np.sum(saved['support_force'][after_start] > 2., axis=1).astype(int).tolist()))
            observed_modes[name] = modes
            gate(name+'.observed_two_foot_support', 2 in modes, 1, 'boolean', '==')
            gate(name+'.observed_four_foot_support', 4 in modes, 1, 'boolean', '==')

        # Verify every simulation scene, including controls: no auxiliary base
        # actuators, kinematic carrier, or equality support constraints.
        model = mujoco.MjModel.from_xml_path(str(results/f'{name}.xml'))
        for label, value, expected in [('nq', model.nq, 19), ('nv', model.nv, 18),
                                        ('motors', model.nu, 12), ('equalities', model.neq, 0),
                                        ('mocap_bodies', model.nmocap, 0),
                                        ('free_joints', np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 1)]:
            gate('scene.'+name+'.'+label, value, expected, 'count', '==')
        target_joints = model.actuator_trnid[:, 0]
        hinge_motors = (np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                        and np.all(target_joints >= 0)
                        and np.all(model.jnt_type[target_joints] == mujoco.mjtJoint.mjJNT_HINGE)
                        and len(set(target_joints.tolist())) == 12)
        gate('scene.'+name+'.distinct_leg_hinge_motors_only', hinge_motors, 1, 'boolean', '==')
        base_joint = int(model.body_jntadr[model.body('base').id])
        gate('scene.'+name+'.free_base_body', model.jnt_type[base_joint], int(mujoco.mjtJoint.mjJNT_FREE), 'enum', '==')

    negative = cases['no_actuation']
    with np.load(results/'no_actuation.npz', allow_pickle=False) as saved:
        progress = float(saved['qpos'][-1, 0]-saved['qpos'][0, 0])
    gate('negative_control.no_actuation_incomplete', not negative['completed'], 1, 'boolean', '==')
    gate('negative_control.no_actuation_misses_course',
         progress < .3 or negative['final_position_error_m'] > .3, 1, 'boolean', '==')

    refinement = {}
    with np.load(results/'nominal.npz', allow_pickle=False) as coarse, np.load(results/'fine.npz', allow_pickle=False) as fine:
        aligned = (coarse['time'].shape == fine['time'].shape
                   and np.allclose(coarse['time'], fine['time'], atol=1e-12, rtol=0.))
        gate('refinement.common_sample_times', aligned, 1, 'boolean', '==')
        if aligned:
            positions = np.linalg.norm(coarse['qpos'][:, :3]-fine['qpos'][:, :3], axis=1)
            rotation_a = Rotation.from_quat(coarse['qpos'][:, [4, 5, 6, 3]])
            rotation_b = Rotation.from_quat(fine['qpos'][:, [4, 5, 6, 3]])
            angles = (rotation_a.inv()*rotation_b).magnitude()
            refinement = dict(maximum_base_position_difference_m=float(np.max(positions)),
                              final_base_position_difference_m=float(positions[-1]),
                              maximum_base_orientation_difference_rad=float(np.max(angles)))
        else:
            refinement = dict(maximum_base_position_difference_m=None,
                              final_base_position_difference_m=None,
                              maximum_base_orientation_difference_rad=None)
    for field, limit, unit in [('maximum_base_position_difference_m', .025, 'm'),
                                ('final_base_position_difference_m', .01, 'm'),
                                ('maximum_base_orientation_difference_rad', .1, 'rad')]:
        value = refinement[field]
        gate('refinement.'+field, np.inf if value is None else value, limit, unit)

    result = dict(
        passed=all(item['passed'] for item in checks.values()),
        passed_count=sum(item['passed'] for item in checks.values()), check_count=len(checks),
        checks=checks, cases=cases, reference=read_json(results/'reference.json'),
        mechanics_details=mechanics['details'], contact_details=contacts['details'],
        reference_details={k: v for k, v in reference.items() if k != 'checks'},
        refinement=refinement, measured_support_modes_after_2s=observed_modes,
        measured_support_definition='A foot supports when the recorded net upward world-Z ground reaction exceeds 2 N; evaluate samples at t>=2 s. Contact-normal magnitude is a separate log.',
        negative_control_details=dict(final_forward_progress_m=progress,
            acceptance='Incomplete, and either less than 0.3 m forward progress or final position error greater than 0.3 m.'),
        task=f'{DURATION:g} s torque-driven floating-base course with alternating diagonal support, raised rails, a low gate, yaw/lateral docking and two logged external push disturbances.',
        framework='PACDM assembles a 30-coordinate foot-task graph (18 physical, 12 massless targets), rank 12 and 18 independent task coordinates. Independent CMG/Pinocchio mechanics drive a motor-bounded whole-body inverse-dynamics QP.',
        ablation_scope='PD_ablation replaces whole-body inverse-dynamics motor commands with bounded joint PD while retaining the same PACDM reference. Its outcome is descriptive: it does not isolate PACDM against another inverse-kinematics method. Only source torque caps and numerical warnings are acceptance gates for this ablation.',
        scope=[
            'Go2 is a floating branched tree with 18 physical velocities and 12 leg motors. Six base scalar stages represent a local ZYX chart and are unactuated.',
            'Temporary ideal no-slip foot constraints define contact-mode charts; the robot has no permanent structural loops. Actual sphere contacts may roll, slip or detach.',
            'MuJoCo alone integrates unilateral contact. Pinocchio validates authored rigid-body mechanics and provides the controller dynamics; contact trajectories are not expected to be identical between engines.',
            'Only twelve source torque motors command motion. There are no base support actuators, mocap bodies or equality constraints. The only applied external body forces are explicitly logged disturbances.',
            'The route, foot placements and support schedule are prescribed. This is not autonomous navigation, perception, online foothold planning or arbitrary-terrain locomotion.',
            'The payload case adds a 1.5 kg point mass at the source base COM while the controller retains the nominal model; it is not a separate carried articulated object.',
            'QP contact forces are predictions. Support-mode observations use actual upward MuJoCo ground reactions; unexpected contacts count every non-foot world contact and robot self-contact above 1 N.',
            'Positive cases must meet all declared tracking, contact, motor, duration, stability and QP gates. No-actuation is an expected-failure control, and PD_ablation has no imposed performance comparison.',
            'Finite simulation evidence in the pinned environment; hardware validation, global workspace proofs, contact stability theorems and generic locomotion guarantees are outside its scope.',
        ])
    write_json(results/'validation.json', result)
    return result


def manifest(root=ROOT):
    root = Path(root)
    excluded = {'.git', '.venv', 'venv', 'env', '__pycache__',
                '.pytest_cache', '.mypy_cache', '.ruff_cache', 'figures', 'derived',
                'workflows', 'records', 'docs', 'tools', 'tests', 'generated_figures'}
    hashes = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (not path.is_file() or any(part in excluded for part in relative.parts)
                or path.name in {'SHA256SUMS.json', 'RELEASE_MANIFEST.json', 'SOURCE_SHA256SUMS'}
                or path.suffix in {'.pyc', '.log', '.zip'}):
            continue
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Run evidence manifest: written to results/, never to a tracked file.
    write_json(root/RUN_MANIFEST, hashes)


def verify(root=ROOT):
    root = Path(root)
    if not (root/RUN_MANIFEST).is_file():
        raise SystemExit(f'No run evidence manifest ({RUN_MANIFEST.as_posix()}); run python run_go2.py first.')
    hashes = read_json(root/RUN_MANIFEST)
    bad = [name for name, digest in hashes.items()
           if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
    if bad:
        raise RuntimeError('Missing or modified files: '+', '.join(bad))
    saved = read_json(root/'results/validation.json')
    print(f'Integrity verified: {len(hashes)} files. Recorded acceptance: '
          f"{'PASS' if saved['passed'] else 'FAIL'} ({saved['passed_count']}/{saved['check_count']}).", flush=True)
    if not saved['passed']:
        raise SystemExit(1)


def replay(root=ROOT):
    """Replay saved native configurations; this does not rerun dynamics."""
    import time
    import mujoco.viewer
    root = Path(root)
    with np.load(root/'results/nominal.npz', allow_pickle=False) as saved:
        times, positions = saved['time'].copy(), saved['qpos'].copy()
    model = mujoco.MjModel.from_xml_path(str(root/'results/nominal.xml'))
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.perf_counter()
        while viewer.is_running():
            now = (time.perf_counter()-start) % times[-1]
            index = min(np.searchsorted(times, now), len(times)-1)
            data.qpos[:] = positions[index]
            mujoco.mj_forward(model, data); viewer.sync(); time.sleep(.01)


def _job(item):
    return run_case(ROOT, name=item[0], **item[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true', help='Render the recorded nominal run to MP4.')
    parser.add_argument('--workers', type=int, default=3, help='Parallel case processes, from 1 to 7.')
    parser.add_argument('--audit-existing', action='store_true', help='After a completed run, verify generated hashes and recorded acceptance.')
    parser.add_argument('--replay', action='store_true', help='Open interactive replay of the saved nominal configurations.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 7:
        parser.error('--workers must be between 1 and 7')
    if args.audit_existing:
        verify(); return
    if args.replay:
        replay(); return
    if mujoco.__version__ != '3.3.7' or pin.__version__ != '3.8.0':
        raise RuntimeError('Install MuJoCo 3.3.7 and Pinocchio 3.8.0 using requirements-linux.txt.')
    results = ROOT/'results'; results.mkdir(exist_ok=True)
    write_json(results/'validation.json', dict(passed=False, status='Run in progress; acceptance incomplete.'))
    try:
        cmg = build_model(); save_model(cmg)
        print('Checking independent floating-base mechanics and contact charts...', flush=True)
        mechanics = validate_mechanics(cmg, output=results/'mechanics_validation.json')
        contacts = validate_contacts(cmg, output=results/'contact_validation.json')
        if not mechanics['passed'] or not contacts['passed']:
            raise RuntimeError('Independent model audit failed; inspect mechanics_validation.json and contact_validation.json.')
        print(f'Assembling the {DURATION:g} s foot task with PACDM...', flush=True)
        make_reference(ROOT)
        reference = validate_reference(ROOT); write_json(results/'task_validation.json', reference)
        if not reference['passed']:
            raise RuntimeError('Reference audit failed; inspect results/task_validation.json.')
        print('Running five positive cases, no-actuation control and PD ablation...', flush=True)
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for value in pool.map(_job, CASES.items()):
                    print(f"{value['name']}: completed={value['completed']}, final base error={1000*value['final_position_error_m']:.3f} mm", flush=True)
        else:
            for item in CASES.items():
                value = _job(item)
                print(f"{value['name']}: completed={value['completed']}, final base error={1000*value['final_position_error_m']:.3f} mm", flush=True)
        result = aggregate(ROOT)
        if args.render:
            from go2.render import render
            render(ROOT)
        from go2.report import build_report
        build_report(ROOT); manifest(ROOT)
        print(f"{'PASS' if result['passed'] else 'FAIL'}: {result['passed_count']}/{result['check_count']} checks.", flush=True)
        if not result['passed']:
            for name, item in result['checks'].items():
                if not item['passed']:
                    print('Failed: '+name+' '+json.dumps(item), flush=True)
            raise SystemExit(1)
    except Exception as error:
        # A previous successful record must never survive an incomplete run.
        saved = read_json(results/'validation.json')
        saved.update(passed=False, run_error=f'{type(error).__name__}: {error}')
        write_json(results/'validation.json', saved)
        raise


if __name__ == '__main__':
    main()
