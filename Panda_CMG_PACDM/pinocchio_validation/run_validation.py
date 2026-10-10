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
RUN_MANIFEST = Path('results')/'SHA256SUMS.json'
CORE_SHA = 'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    """Record the run evidence in results/SHA256SUMS.json (never a tracked file)."""
    write_json(ROOT/RUN_MANIFEST, {p.relative_to(ROOT).as_posix():sha(p)
        for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
        and p.name != 'SHA256SUMS.json' and p.suffix != '.pyc'
        and not any(part.startswith('.') or part in ('venv', 'env')
                    for part in p.relative_to(ROOT).parts)})


def verify():
    if not (ROOT/RUN_MANIFEST).is_file():
        raise SystemExit(f'No run evidence manifest ({RUN_MANIFEST.as_posix()}); run python run_validation.py first.')
    expected = json.loads((ROOT/RUN_MANIFEST).read_text())
    bad = [name for name, value in expected.items()
           if not (ROOT/name).is_file() or sha(ROOT/name) != value]
    if bad:
        raise RuntimeError('Integrity mismatches: '+', '.join(bad))
    print(f'Integrity PASS: {len(expected)} files')


def _same_values(a, b, atol=1e-12):
    """Structural equality that ignores floating-point round-off."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same_values(a[k], b[k], atol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_same_values(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return isinstance(b, (int, float)) and not isinstance(b, bool) and bool(np.isclose(a, b, rtol=0, atol=atol))
    return a == b


def _same_arrays(generated, shipped, rtol=1e-6, atol=1e-7):
    with np.load(generated, allow_pickle=False) as a, np.load(shipped, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return 'array names differ'
        for k in a.files:
            x, y = a[k], b[k]
            if x.shape != y.shape:
                return f'{k}: shape {x.shape} != {y.shape}'
            if x.dtype.kind in 'fc' or y.dtype.kind in 'fc':
                if not np.allclose(x, y, rtol=rtol, atol=atol):
                    return f'{k}: max |difference| {float(np.max(np.abs(x-y))):.3g}'
            elif not np.array_equal(x, y):
                return f'{k}: values differ'
    return ''


def check_shipped_data(rebuilt_reference):
    """The cases read data/panda_cmg.json and data/reference.npz, which a run never rewrites.
    They must match the model (and, unless --use-existing-reference, the PACDM reference)
    that this run has just rebuilt into results/."""
    results = ROOT/'results'
    problems = []
    if not _same_values(json.loads((results/'panda_cmg.json').read_text()),
                        json.loads((ROOT/'data/panda_cmg.json').read_text())):
        problems.append('panda_cmg.json')
    if rebuilt_reference:
        why = _same_arrays(results/'reference.npz', ROOT/'data/reference.npz')
        if why:
            problems.append('reference.npz ('+why+')')
    if problems:
        raise RuntimeError('The rebuilt '+', '.join(problems)+' in results/ differ from the shipped copies in data/. '
                           'If the model or task change is intended, copy results/panda_cmg.json, results/reference.npz '
                           'and results/reference.json into data/ and rerun.')


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
        # Effort demanded before the force-range clip; the applied effort is within the limits by construction.
        gate(name+'.effort_limits', max(np.asarray(c['peak_actuator_effort_demand'])/np.array([87]*4+[12]*3+[100])), 1.00000001, 'ratio')
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
        raise RuntimeError('Use the pinned Pinocchio 3.8.0 environment')
    if sha(ROOT/'vendor/pacdm_original.py') != CORE_SHA:
        raise RuntimeError('Original PACDM implementation has changed')
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    write_json(results/'validation.json', dict(passed=False,status='Run in progress'))
    started = time.perf_counter()
    # Run outputs go to results/ only; tracked files in data/ are checked, never rewritten.
    cmg = build_model(); save_model(cmg, results/'panda_cmg.json')
    if args.use_existing_reference:
        with np.load(ROOT/'data/reference.npz') as f:
            reference={key:f[key] for key in f.files}
    else:
        print('Regenerating the original PACDM route...', flush=True)
        reference = build_reference(cmg)
        np.savez_compressed(results/'reference.npz', **{k:v for k,v in reference.items() if k!='info'})
        write_json(results/'reference.json', reference['info'])
    check_shipped_data(rebuilt_reference=not args.use_existing_reference)
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
