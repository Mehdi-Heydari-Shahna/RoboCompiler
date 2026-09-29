"""Acceptance gates for the executed Pinocchio cases; no simulation here."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_HASH = 'bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de'
CASES = [('coarse',.004,8.,True),('nominal',.002,8.,True),('fine',.001,8.,True),
         ('heavy_payload',.002,14.,True),('no_feedforward',.002,8.,False)]


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def compare(a,b):
    with np.load(a,allow_pickle=False) as x, np.load(b,allow_pickle=False) as y:
        if x['time'][0]!=y['time'][0] or x['time'][-1]!=y['time'][-1]:
            raise ValueError('Comparison requires identical time range.')
        q = np.column_stack([np.interp(x['time'],y['time'],y['q'][:,k]) for k in range(6)])
        position = float(np.linalg.norm(x['q'][:,:3]-q[:,:3],axis=1).max())
        ra = Rotation.from_euler('ZYX',x['q'][:,3:6]); rb = Rotation.from_euler('ZYX',q[:,3:6])
        angle = float(np.rad2deg((ra.inv()*rb).magnitude().max()))
    return dict(max_position_difference_m=position,max_orientation_difference_deg=angle)


def aggregate(root):
    root = Path(root); r = root/'results'
    mechanics = json.loads((r/'mechanics.json').read_text()); audits = json.loads((r/'trajectory_audits.json').read_text())
    negative = json.loads((r/'audit_negative_controls.json').read_text())
    evidence = json.loads((r/'case_evidence.json').read_text())
    cases = {name:json.loads((r/f'{name}.json').read_text()) for name,*_ in CASES}
    reference = json.loads((r/'reference.json').read_text())
    expected_cases = {x[0] for x in CASES}
    if set(audits)!=expected_cases or set(evidence)!=expected_cases:
        raise ValueError('Independent trajectory audits must cover exactly all five prescribed cases.')
    checks = {}
    def gate(name, value, limit, unit='', relation='<='):
        passed = bool(np.isfinite(value) and (value<=limit if relation=='<=' else value>=limit if relation=='>=' else value==limit))
        checks[name] = dict(passed=passed,value=value,limit=limit,unit=unit,relation=relation)
    gate('source.unchanged_PACDM',int(hashlib.sha256((root/'vendor/pacdm_original.py').read_bytes()).hexdigest()==SOURCE_HASH),1,relation='==')
    gate('independent_mechanics',int(mechanics['passed']),1,relation='==')
    gate('auditor_negative_controls',int(negative['passed']),1,relation='==')
    for name,a in audits.items(): gate(f'independent_trajectory.{name}',int(a['passed']),1,relation='==')
    for name,a in evidence.items(): gate(f'case_evidence.{name}',int(a['passed']),1,relation='==')
    for key,limit in [('max_closure_residual',1e-8),('max_tangent_residual',1e-9),
                      ('max_acceleration_residual',1e-8),('max_target_pose_reconstruction_error',1e-7)]:
        gate('reference.'+key,reference[key],limit,'mixed SI')
    gate('reference.positive_reduced_mass',reference['min_reduced_mass_eigenvalue'],1e-8,'kg','>=')
    for name,c in cases.items():
        gate(name+'.full_mission_s',c['duration_s'],22.,'s','==')
        gate(name+'.finite',int(c['all_finite']),1,relation='==')
        gate(name+'.force_limit',c['max_actuator_force_N'],900.,'N')
        gate(name+'.no_saturation',c['saturation_samples'],0,'samples','==')
        gate(name+'.closure',c['max_closure_error_m'],1e-8,'m')
        gate(name+'.prismatic_limits',c['min_prismatic_limit_margin_m'],0.,'m','>=')
        gate(name+'.revolute_limits',c['min_revolute_limit_margin_rad'],0.,'rad','>=')
        for key in ['rank_full','rank_passive']: gate(name+'.'+key,c['min_'+key],36,'rank','==')
        gate(name+'.conditioning',c['min_mapping_rcond'],1e-5,'reciprocal condition','>=')
        gate(name+'.positive_reduced_mass',c['min_reduced_mass_eigenvalue'],1e-8,'kg','>=')
        for key,limit,unit in [('augmented_closure_residual',1e-8,'mixed SI'),('tangent_residual',1e-9,'mixed SI'),
                               ('velocity_closure_residual',1e-9,'mixed SI/s'),('acceleration_closure_residual',1e-8,'mixed SI/s2'),
                               ('reduced_equation_residual',1e-9,'N')]:
            gate(name+'.'+key,c['max_'+key],limit,unit)
        if name!='no_feedforward':
            for key,limit,unit in [('max_position_error_m',.010,'m'),('max_orientation_error_deg',.5,'deg'),
                                  ('docking_max_position_error_m',.002 if name=='heavy_payload' else .0005,'m'),
                                  ('docking_max_orientation_error_deg',.1,'deg')]:
                gate(name+'.'+key,c[key],limit,unit)
    refinement = {label:compare(r/f'{a}.npz',r/f'{b}.npz') for label,a,b in [('4ms_vs_2ms','coarse','nominal'),('2ms_vs_1ms','nominal','fine')]}
    fine = refinement['2ms_vs_1ms']; coarse = refinement['4ms_vs_2ms']
    gate('refinement.position_difference',fine['max_position_difference_m'],.0005,'m')
    gate('refinement.orientation_difference',fine['max_orientation_difference_deg'],.03,'deg')
    for key in ['max_position_difference_m','max_orientation_difference_deg']:
        ratio = fine[key]/max(coarse[key],1e-15)
        refinement[key+'_refinement_ratio'] = ratio
        gate('refinement.decreasing_'+key,ratio,.75,'ratio')
    ratio = cases['nominal']['rms_position_error_m']/cases['no_feedforward']['rms_position_error_m']
    gate('feedforward.lower_RMS',ratio,1.,'ratio')
    # Historical cross-engine evidence is explicitly labeled; MuJoCo is not rerun.
    legacy = compare(r/'nominal.npz',root/'legacy_evidence/nominal.npz')
    gate('legacy_mujoco.position_agreement',legacy['max_position_difference_m'],.001,'m')
    gate('legacy_mujoco.orientation_agreement',legacy['max_orientation_difference_deg'],.1,'deg')
    result = dict(passed=all(c['passed'] for c in checks.values()),passed_count=sum(c['passed'] for c in checks.values()),check_count=len(checks),
        checks=checks,cases=cases,refinement=refinement,legacy_mujoco_comparison=legacy,reference=reference,
        mechanics=mechanics,trajectory_audits=audits,auditor_negative_controls=negative,case_evidence=evidence,
        summary='Simulation verification of the declared Stewart rigid model and unchanged PACDM algorithm with Pinocchio 3.8.0.',
        limits=['Independent native constraint checks share the same Pinocchio tree dynamics; they verify PACDM reduction, not independent inertial truth.',
                'The historical MuJoCo trajectory is used only for a separately labeled comparison; it is not freshly simulated here.',
                'Timestep refinement covers 4, 2, 1 ms on this complete mission; no universal error bound or global convergence theorem is implied.',
                'No measured hardware, collision avoidance, structural compliance, actuator electrical/hydraulic fidelity, or global workspace guarantee.',
                'Video is CPU-rendered playback of saved Pinocchio/PACDM states; quantitative acceptance comes from numerical gates.'])
    write_json(r/'validation.json',result)
    return result


def hash_manifest(root):
    root = Path(root)
    hashes = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS.json' and p.suffix!='.pyc'}
    write_json(root/'SHA256SUMS.json',hashes)


def verify_manifest(root):
    root = Path(root); hashes = json.loads((root/'SHA256SUMS.json').read_text())
    bad = [k for k,v in hashes.items() if not (root/k).is_file() or hashlib.sha256((root/k).read_bytes()).hexdigest()!=v]
    if bad: raise RuntimeError('Missing/modified files: '+', '.join(bad))
    print(f'Integrity PASS: {len(hashes)} files.')
