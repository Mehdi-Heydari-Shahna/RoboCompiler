"""Reproduce the independent Pinocchio PACDM experiment and its video."""
from pathlib import Path
import argparse
import hashlib
import importlib.abc
import json
import os
import platform
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')


class NoMuJoCo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'mujoco':
            raise ImportError('The Pinocchio validation must run without MuJoCo.')
        return None


sys.meta_path.insert(0, NoMuJoCo())
import numpy as np
import scipy
import pinocchio as pin
from scipy.spatial.transform import Rotation
from panda.model import build_model, save_model
from panda.task import build_reference
from panda.task_validation import validate_task
from pin_validation.simulation import CASES, run_case

ROOT = Path(__file__).resolve().parent
CORE_SHA = '492209e3a33281684751990ce97e02459e18a5529c2b7c4e8bae124eadc310ca'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    write_json(ROOT/'SHA256SUMS.json', {str(p.relative_to(ROOT)):sha(p)
        for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
        and p != ROOT/'SHA256SUMS.json' and p.suffix != '.pyc'
        and not any(part.startswith('.') or part in ('venv', 'env')
                    for part in p.relative_to(ROOT).parts)})


def verify():
    expected = json.loads((ROOT/'SHA256SUMS.json').read_text())
    bad = [name for name, value in expected.items()
           if not (ROOT/name).is_file() or sha(ROOT/name) != value]
    if bad:
        raise RuntimeError('Integrity mismatches: '+', '.join(bad))
    print(f'Integrity PASS: {len(expected)} files')


def aggregate(task, mechanics):
    results = ROOT/'results'
    cases = {name:json.loads((results/f'{name}.json').read_text()) for name in CASES}
    checks = {'task.'+k:v for k,v in task['checks'].items()}
    checks.update({'mechanics.'+k:v for k,v in mechanics['checks'].items()})

    def gate(name, value, limit, unit='', relation='<='):
        value, limit = float(value), float(limit)
        passed = np.isfinite(value) and (value <= limit if relation=='<=' else
            value >= limit if relation=='>=' else value == limit)
        checks[name] = dict(passed=bool(passed), value=value, limit=limit, unit=unit, relation=relation)

    gate('original_PACDM_SHA256_unchanged', sha(ROOT/'vendor/pacdm_original.py')==CORE_SHA, 1, 'boolean', '==')
    gate('MuJoCo_not_imported', 'mujoco' not in sys.modules, 1, 'boolean', '==')
    for name, c in cases.items():
        gate(name+'.duration', c['duration_s'], 22., 's', '==')
        gate(name+'.coupling', c['max_coupling_error_m'], 1e-10, 'm')
        gate(name+'.dynamics_residual', c['max_dynamics_residual'], 1e-8, 'mixed N/Nm')
        gate(name+'.arm_limits', c['minimum_arm_limit_margin_rad'], .05, 'rad', '>=')
        gate(name+'.finger_limits', c['minimum_finger_limit_margin_m'], -1e-6, 'm', '>=')
        gate(name+'.effort_limits', max(np.asarray(c['peak_actuator_effort'])/np.array([87]*4+[12]*3+[100])), 1.00000001, 'ratio')
        gate(name+'.energy_work_balance', c['max_energy_work_balance_error_J'], .01 if name=='no_feedforward' else .001, 'J')
        if name != 'no_feedforward':
            gate(name+'.tool_position', c['max_tool_position_error_m'], .002, 'm')
            gate(name+'.tool_rotation', c['max_tool_orientation_error_deg'], .5, 'deg')
            gate(name+'.joint_tracking', c['max_arm_joint_error_deg'], .5, 'deg')
            gate(name+'.no_saturation', c['saturation_rhs_evaluations'], 0, 'RHS evaluations', '==')
            gate(name+'.no_setpoint_clipping', c['setpoint_clip_rhs_evaluations'], 0, 'RHS evaluations', '==')
            gate(name+'.final_position', c['final_tool_error_m'], 1e-4, 'm')

    pairs = []
    for coarse, fine in [('nominal','half_step'),('half_step','quarter_step')]:
        with np.load(results/f'{coarse}.npz') as a, np.load(results/f'{fine}.npz') as b:
            if not np.allclose(a['time'], b['time'], rtol=0, atol=1e-12):
                raise RuntimeError('Timestep comparisons require identical physical times')
            dp = np.linalg.norm(a['tool_pos']-b['tool_pos'], axis=1)
            dr = (Rotation.from_matrix(a['tool_R']).inv()*Rotation.from_matrix(b['tool_R'])).magnitude()
            dq = abs(a['q'][:,:7]-b['q'][:,:7])
            pair = dict(coarse=coarse, fine=fine, max_position_difference_m=float(dp.max()),
                max_rotation_difference_deg=float(np.rad2deg(dr.max())), max_joint_difference_rad=float(dq.max()),
                compared_samples=len(dp), includes_entire_22_second_cycle=True)
        pairs.append(pair)
        gate('refinement.'+coarse+'.tool_position', pair['max_position_difference_m'], 1e-5, 'm')
        gate('refinement.'+coarse+'.tool_rotation', pair['max_rotation_difference_deg'], .005, 'deg')
        gate('refinement.'+coarse+'.joint_position', pair['max_joint_difference_rad'], 1e-4, 'rad')
    # At numerical noise level a rate is not meaningful. Otherwise require
    # contraction, independently of the integrator's nominal fourth order.
    ratio = pairs[1]['max_position_difference_m']/max(pairs[0]['max_position_difference_m'],1e-15)
    gate('refinement.position_contraction_or_roundoff',
         int(ratio < .65 or pairs[1]['max_position_difference_m'] < 1e-10), 1, 'boolean', '==')
    gate('ablation.feedforward_RMS_improvement',
        cases['no_feedforward']['rms_tool_position_error_m']/cases['nominal']['rms_tool_position_error_m'], 3., 'ratio', '>=')
    gate('ablation.without_feedforward_fails_positive_position_gate',
        cases['no_feedforward']['max_tool_position_error_m'], .002, 'm', '>=')
    gate('perturbation.recovered_final_tool_error', cases['initial_offset']['final_tool_error_m'], 1e-4, 'm')
    result = dict(passed=all(c['passed'] for c in checks.values()),
        passed_count=sum(c['passed'] for c in checks.values()), check_count=len(checks),
        checks=checks, cases=cases, refinement=pairs, refinement_position_ratio=ratio,
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            pinocchio=pin.__version__, numpy=np.__version__, scipy=scipy.__version__),
        scope=['Finite numerical validation of original PACDM on the Panda task route.',
            'Pinocchio computes rigid-body dynamics and FK; our RK4 integrator advances actual states.',
            'One physical finger equality is reduced exactly; tool-target equations define a virtual task.',
            'The external load is a world tool wrench. No payload mass/inertia, contact, friction, collision, grasp or socket dynamics are simulated.',
            'No hardware validation or universal proof of PACDM accuracy.'],
        original_PACDM_SHA256=CORE_SHA, reference_SHA256=sha(ROOT/'data/reference.npz'),
        trajectory_SHA256={name:sha(results/f'{name}.npz') for name in CASES},
        ablation_scope='Only arm inverse-dynamics and damping feedforward are removed. Finger compensation is retained unchanged. This tests arm model compensation, not PACDM against another IK algorithm.')
    write_json(results/'validation.json', result)
    return result


def job(item):
    name, kwargs = item
    result = run_case(ROOT, name, **kwargs)
    return name, result['max_tool_position_error_m'], result['max_energy_work_balance_error_J']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render', action='store_true', help='also make the 22-second MP4')
    p.add_argument('--render-only', action='store_true')
    p.add_argument('--audit-existing', action='store_true')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--use-existing-reference', action='store_true', help='recheck saved reference instead of regenerating it')
    args = p.parse_args()
    if args.audit_existing:
        verify(); return
    if args.render_only:
        from pin_validation.render import render_video
        render_video(ROOT)
        from pin_validation.report import build_report
        build_report(ROOT)
        manifest(); return
    if pin.__version__ != '3.8.0':
        raise RuntimeError('Use the supplied Pinocchio 3.8.0 environment for this release')
    if sha(ROOT/'vendor/pacdm_original.py') != CORE_SHA:
        raise RuntimeError('Original PACDM implementation has changed')
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'validation.json', dict(passed=False,status='Run in progress'))
    started = time.perf_counter()
    cmg = build_model(); save_model(cmg)
    if args.use_existing_reference:
        with np.load(ROOT/'data/reference.npz') as f:
            reference={key:f[key] for key in f.files}
    else:
        print('Regenerating the original PACDM route...', flush=True)
        reference = build_reference(cmg)
        np.savez_compressed(ROOT/'data/reference.npz', **{k:v for k,v in reference.items() if k!='info'})
        write_json(ROOT/'data/reference.json', reference['info'])
    print('Checking route and independent mechanics...', flush=True)
    task = validate_task(cmg, reference, output=results/'task_validation.json')
    from pin_validation.mechanics import run
    mechanics = run(cmg, reference, output=results/'mechanics.json')
    if not task['passed'] or not mechanics['passed']:
        raise RuntimeError('Reference or independent mechanics failed; see JSON records')
    print('Integrating six full-cycle Pinocchio experiments...', flush=True)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(args.workers,len(CASES))) as pool:
            for name, error, energy in pool.map(job, CASES.items()):
                print(f'{name}: max tool error {error*1000:.6f} mm; energy/work error {energy:.3g} J',flush=True)
    else:
        for item in CASES.items():
            print(job(item), flush=True)
    report = aggregate(task, mechanics)
    if args.render:
        from pin_validation.render import render_video
        render_video(ROOT)
    from pin_validation.report import build_report
    build_report(ROOT)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['passed_count']}/{report['check_count']} checks in {time.perf_counter()-started:.1f} s", flush=True)
    manifest()
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
