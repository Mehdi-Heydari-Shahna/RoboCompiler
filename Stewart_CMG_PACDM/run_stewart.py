"""Run the complete Stewart CMG/PACDM benchmark and save its evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# Small dense matrices are faster and reproducible with one BLAS worker.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('MUJOCO_GL','egl')

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import pinocchio as pin
from stewart.model import make_cmg,save_cmg,compile_mujoco
from stewart.reference import build_reference
from stewart.simulation import run_case
from stewart.validation import validate_mechanics
from stewart.projected import run_projected

ROOT=Path(__file__).resolve().parent
SOURCE_HASH='492209e3a33281684751990ce97e02459e18a5529c2b7c4e8bae124eadc310ca'


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def aggregate(root=ROOT):
    r=root/'results'
    mechanics=json.loads((r/'mechanics.json').read_text());ref=json.loads((r/'reference.json').read_text())
    cases={k:json.loads((r/f'{k}.json').read_text()) for k in ['nominal','fine','heavy_payload','no_feedforward']}
    projected=json.loads((r/'pacdm.json').read_text())
    checks={'mechanics.'+k:v for k,v in mechanics['checks'].items()}
    def gate(name,value,limit,unit='',relation='<='):
        passed=np.isfinite(value) and (value<=limit if relation=='<=' else value>=limit if relation=='>=' else value==limit)
        checks[name]=dict(passed=bool(passed),value=value,limit=limit,unit=unit,relation=relation)
    digest=hashlib.sha256((root/'vendor/pacdm_original.py').read_bytes()).hexdigest()
    gate('PACDM_core_integrity',int(digest==SOURCE_HASH),1,'boolean','==')
    gate('reference.closure',ref['max_closure_residual'],1e-8,'mixed SI')
    gate('reference.JN',ref['max_tangent_residual'],1e-9,'mixed SI')
    gate('reference.acceleration_closure',ref['max_acceleration_residual'],1e-8,'mixed SI')
    gate('reference.forward_pose_error',ref['max_target_pose_reconstruction_error'],1e-7,'mixed SI')
    gate('reference.reduced_mass_positive',ref['min_reduced_mass_eigenvalue'],1e-8,'kg','>=')
    for name,case in cases.items():
        gate(name+'.no_warnings',sum(case['warning_counts'].values()),0,'count','==')
        gate(name+'.no_saturation',case['saturation_samples'],0,'count','==')
        gate(name+'.joint_limits',case['min_joint_limit_margin'],0.,'mixed SI','>=')
        gate(name+'.native_closure',case['max_closure_error_m'],1e-5,'m')
        if name!='no_feedforward':
            gate(name+'.peak_position_error',case['max_position_error_m'],.010,'m')
            gate(name+'.peak_orientation_error',case['max_orientation_error_deg'],.5,'deg')
            gate(name+'.docking_position',case['docking_max_position_error_m'],.002 if name=='heavy_payload' else .0005,'m')
            gate(name+'.docking_orientation',case['docking_max_orientation_error_deg'],.1,'deg')
    with np.load(r/'nominal.npz') as nominal, np.load(r/'fine.npz') as fine:
        interpolated=np.column_stack([np.interp(nominal['time'],fine['time'],fine['q'][:,k]) for k in range(6)])
        difference=np.max(np.linalg.norm(nominal['q'][:,:3]-interpolated[:,:3],axis=1))
        Ra=Rotation.from_euler('ZYX',nominal['q'][:,3:6]);Rb=Rotation.from_euler('ZYX',interpolated[:,3:6])
        angle=np.rad2deg(np.max((Ra.inv()*Rb).magnitude()))
    gate('refinement.position_difference',float(difference),.0005,'m')
    gate('refinement.orientation_difference',float(angle),.03,'deg')
    for metric,limit,unit in [('max_position_error_m',.010,'m'),('max_orientation_error_deg',.5,'deg'),
                              ('docking_max_position_error_m',.0005,'m'),('docking_max_orientation_error_deg',.1,'deg'),
                              ('max_closure_error_m',1e-8,'m'),('max_reduced_equation_residual_N',1e-9,'N'),
                              ('saturation_samples',0,'count')]:
        gate('pacdm.'+metric,projected[metric],limit,unit)
    with np.load(r/'nominal.npz') as native,np.load(r/'pacdm.npz') as direct:
        if not np.array_equal(native['time'],direct['time']):raise ValueError('Direct comparison requires identical sample times')
        pdiff=float(np.max(np.linalg.norm(native['q'][:,:3]-direct['q'][:,:3],axis=1)))
        Ra=Rotation.from_euler('ZYX',native['q'][:,3:6]);Rb=Rotation.from_euler('ZYX',direct['q'][:,3:6])
        adiff=float(np.rad2deg(np.max((Ra.inv()*Rb).magnitude())))
    gate('pacdm_vs_mujoco.position_difference',pdiff,.001,'m')
    gate('pacdm_vs_mujoco.orientation_difference',adiff,.1,'deg')
    ratio=cases['nominal']['rms_position_error_m']/cases['no_feedforward']['rms_position_error_m']
    gate('feedforward_ablation.RMS_error_ratio',ratio,1.,'ratio')
    passed=mechanics['passed'] and all(c['passed'] for c in checks.values())
    report=dict(passed=bool(passed),passed_count=sum(c['passed'] for c in checks.values()),check_count=len(checks),checks=checks,
                model='Idealized rigid six-UPS Stewart platform with declared geometry, inertias, and fixed payload.',
                framework='CMG mechanism specification, PointGraph/PACDM closure reduction, independent Pinocchio compilation, and native MuJoCo simulation.',
                task='22 s helical inspection, six-axis figure-eight, three external wrench pulses and precision docking.',
                cases={**cases,'pacdm':projected},reference=ref,mechanics_details=mechanics['details'],source_PACDM_sha256=digest,
                interpretation='The feedforward-disabled case retains the PACDM reference and feedback gains while setting the analytical feedforward force to zero.',
                limits=['Declared rigid-body simulation with ideal force actuators and a rigidly attached payload.',
                        'PACDM uses ideal point closures; MuJoCo uses native compliant equalities. Closure gaps and trajectory differences are evaluated separately.',
                        'Controller reference uses an 8 kg payload. The heavy case changes the simulated payload to 14 kg without retuning.',
                        'Rank checks cover the deterministic test configurations and the reference mission.'])
    write_json(r/'validation.json',report)
    return report


def hash_manifest(root=ROOT):
    # Cover the benchmark and generated evidence without walking a Git checkout
    # or a local Python environment.
    root_files=('run_stewart.py','run_stewart.bat','reproduce.py','verify_files.py',
                'requirements-linux.txt','environment.yml','README.md','METHODS.md',
                'PROVENANCE.json','.gitignore','.gitattributes')
    paths=[root/name for name in root_files if (root/name).is_file()]
    excluded={'.git','.venv','venv','__pycache__','.pytest_cache','.mypy_cache'}
    for name in ('data','stewart','vendor','Supplement','results'):
        paths.extend(p for p in (root/name).rglob('*')
                     if p.is_file() and not (set(p.relative_to(root).parts)&excluded)
                     and p.suffix not in ('.pyc','.log')
                     and (name!='Supplement' or p.suffix in ('.py','.txt')))
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}
    write_json(root/'SHA256SUMS.json',hashes)


def verify_manifest(root=ROOT):
    manifest=json.loads((root/'SHA256SUMS.json').read_text());bad=[]
    for name,expected in manifest.items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append(name)
    if bad:raise RuntimeError('Missing/modified release files: '+', '.join(bad))
    print(f'Integrity verified: {len(manifest)} files.')


def replay(root=ROOT):
    import time
    import mujoco.viewer
    with np.load(root/'results/nominal.npz') as z:
        t=z['time'].copy();q=z['q'].copy();ids=z['coordinate_ids'].copy()
    m=mujoco.MjModel.from_xml_path(str(root/'results/nominal.xml'));d=mujoco.MjData(m)
    addresses=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,str(x))] for x in ids]
    with mujoco.viewer.launch_passive(m,d) as viewer:
        start=time.perf_counter()
        while viewer.is_running():
            elapsed=(time.perf_counter()-start)%t[-1];i=min(np.searchsorted(t,elapsed),len(t)-1)
            d.qpos[addresses]=q[i];mujoco.mj_forward(m,d);viewer.sync();time.sleep(.01)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',action='store_true',help='Also render the full MP4.')
    parser.add_argument('--audit-existing',action='store_true',help='Verify the current checksum manifest without rerunning simulation.')
    parser.add_argument('--replay',action='store_true',help='Interactively replay saved nominal states; no simulation.')
    args=parser.parse_args()
    if args.audit_existing:verify_manifest();return
    if args.replay:replay();return
    if mujoco.__version__!='3.3.7' or pin.__version__!='3.8.0':
        raise RuntimeError('This release is verified with MuJoCo 3.3.7 and Pinocchio 3.8.0; install the dependencies specified for this environment.')
    r=ROOT/'results';r.mkdir(exist_ok=True)
    write_json(r/'validation.json',dict(passed=False,status='Run in progress; acceptance has not completed.'))
    c=make_cmg();save_cmg(c,ROOT/'data/stewart.cmg.json')
    xml=compile_mujoco(c,r/'nominal.xml')
    print('Checking independent dynamics and perturbed-seed PACDM acquisition...',flush=True)
    mechanics=validate_mechanics(c,xml);write_json(r/'mechanics.json',mechanics)
    if not mechanics['passed']:
        raise RuntimeError('Mechanics checks failed; see results/mechanics.json.')
    print('Assembling and differentiating the six-axis mission with PACDM...',flush=True)
    reference=build_reference(c,r/'reference.npz');write_json(r/'reference.json',reference)
    for name,actual,dt,ff in [('nominal',c,.002,True),('fine',c,.001,True),
                              ('heavy_payload',make_cmg(14.),.002,True),('no_feedforward',c,.002,False)]:
        print(f'Simulating {name}...',flush=True)
        value=run_case(actual,r/'reference.npz',r,name,dt,feedforward=ff);write_json(r/f'{name}.json',value)
    print('Advancing the direct PACDM reduced forward dynamics...',flush=True)
    write_json(r/'pacdm.json',run_projected(c,r/'reference.npz',r))
    validation=aggregate()
    if args.render:
        from stewart.render import render_video
        render_video(ROOT)
    from stewart.report import build_report
    build_report(ROOT);hash_manifest()
    print(f"{'PASS' if validation['passed'] else 'FAIL'}: {validation['passed_count']}/{validation['check_count']} checks. See results/report.html.")
    if not validation['passed']:raise SystemExit(1)


if __name__=='__main__':main()
