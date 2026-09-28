"""Aggregate only completed native runs; archived sources are diagnostic baselines.

A suite cannot pass with a missing case, smoke test, invalid ablation, mismatched
reference, failed child, or incompatible sampling grid. No archived trajectory is
accepted as an Isaac run. Evidence files are hashed into the resulting report.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .cases import CONTACT, WRENCH, get_case
from .validation import check, validate_case, WRENCH_REFINEMENT_TOOL_POSITION_M, WRENCH_REFINEMENT_FLOAT32_FLOOR_M
from .simulation import write_json, native_backend_name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as z:
        result = {key: z[key] for key in z.files}
    times = result['time']
    if times.ndim != 1 or len(times) != 2201 or not np.allclose(times, np.arange(2201)*.01, rtol=0, atol=1e-10):
        raise ValueError('Requires the entire 22 s trajectory on the common 100 Hz sampling grid')
    for key, shape in [('q',(2201,9)),('tool_pos',(2201,3)),('tool_R',(2201,3,3))]:
        if result[key].shape != shape or not np.all(np.isfinite(result[key])):
            raise ValueError('Missing/invalid trajectory array: '+key)
    return result


def compare(a, b, object_pose=False):
    if a['time'].shape != b['time'].shape or not np.allclose(a['time'], b['time'], rtol=0, atol=1e-10):
        raise ValueError('Comparison requires identical physical times, not frame indices')
    pkey, rkey = ('object_pos','object_R') if object_pose else ('tool_pos','tool_R')
    for x in (a,b):
        if x[pkey].shape != (len(x['time']),3) or x[rkey].shape != (len(x['time']),3,3):
            raise ValueError('Invalid pose array shape')
        if not np.all(np.isfinite(x[pkey])) or not np.all(np.isfinite(x[rkey])):
            raise ValueError('Nonfinite pose in comparison')
    dp = np.linalg.norm(a[pkey]-b[pkey],axis=1)
    dr = (Rotation.from_matrix(a[rkey]).inv()*Rotation.from_matrix(b[rkey])).magnitude()
    return dict(max_position_difference_m=float(dp.max()),
                max_rotation_difference_deg=float(np.rad2deg(dr.max())),
                max_arm_joint_difference_rad=float(np.max(abs(a['q'][:,:7]-b['q'][:,:7]))),
                compared_samples=len(dp),includes_entire_22_second_cycle=True)


def aggregate(root, output, modes, returncodes):
    root, output = Path(root), Path(output)
    if not modes or any(mode not in ('contact','wrench') for mode in modes):
        raise ValueError('Specify contact and/or wrench suite')
    checks = {}; cases = {}; trajectories = {}; errors = []; hashes = {}; comparisons = {}; references = []
    def gate(name, value, limit, relation='<='):
        checks[name] = check(value,limit,relation)
    for mode in modes:
        required = CONTACT if mode=='contact' else WRENCH
        for name in required:
            key=mode+'/'+name; folder=output/mode/name
            gate(key+'.child_exit_code',returncodes.get(key),0,'==')
            try:
                result=json.loads((folder/'result.json').read_text())
                summary=result['summary']; inputs=json.loads((folder/'inputs.json').read_text())
                if summary.get('mode')!=mode or summary.get('case')!=name:
                    raise ValueError('Case identity mismatch')
                expected=get_case(mode,name)
                if summary.get('native_backend')!=native_backend_name(expected):
                    raise ValueError('Expected a native Isaac Sim result')
                if inputs['mode']!=mode or inputs['case']!=name:
                    raise ValueError('Input identity mismatch')
                if inputs['configuration']!=expected or summary['configuration']!=expected:
                    raise ValueError('Suite requires the predeclared case configuration and timestep')
                for field,live in [('cmg_sha256',root/'data/panda_cmg.json'),('pacdm_sha256',root/'vendor/pacdm_original.py')]:
                    if inputs[field]!=sha(live):raise ValueError('Model/compiler hash mismatch')
                references.append(inputs['reference_sha256'])
                expected_status=('ABLATION_RECORDED' if name=='no_feedforward' else
                                 'NEGATIVE_CONTROL_PASS' if name=='no_grasp' else 'TASK_PASS')
                rechecked=validate_case(summary,result.get('mechanics',{}),smoke=False)
                gate(key+'.accepted_status',result.get('status')==expected_status and rechecked['status']==expected_status,1,'==')
                gate(key+'.native_mechanics',result.get('mechanics',{}).get('passed'),1,'==')
                gate(key+'.data_valid',result.get('data_valid'),1,'==')
                for n,c in rechecked['checks'].items():
                    # Re-evaluate scalar gates, do not trust an edited `passed` flag.
                    checks[key+'.'+n]=check(c.get('value'),c.get('limit'),c.get('relation','<='))
                if not result.get('checks'):raise ValueError('Missing case acceptance checks')
                trajectory=load_trajectory(folder/'trajectory.npz')
                if mode=='contact':compare(trajectory,trajectory,object_pose=True)
                cases[key]=summary;trajectories[key]=trajectory
                for filename in ('trajectory.npz','result.json','inputs.json','mechanics.json'):
                    hashes[key+'/'+filename]=sha(folder/filename)
                gate(key+'.complete_evidence',1,1,'==')
            except (OSError,KeyError,TypeError,ValueError) as exc:
                errors.append(key+': '+str(exc));gate(key+'.complete_evidence',0,1,'==')
        try:
            nominal=cases[mode+'/nominal']; ablation=cases[mode+'/no_feedforward']
            nr=float(nominal['rms_tool_position_error_m']); ar=float(ablation['rms_tool_position_error_m'])
            ratio=nr/ar if ar>0 and math.isfinite(ar) else None
            # Original contact suite requires >=2x RMS benefit; wrench >=3x.
            gate(mode+'.ablation.nominal_over_noFF_RMS',ratio,.5 if mode=='contact' else 1./3.)
            if mode=='wrench':
                gate(mode+'.ablation.without_FF_exceeds_positive_position_limit',ablation['max_tool_position_error_m'],.002,'>=')
            comparisons[mode+'_ablation']=dict(nominal_over_noFF_RMS=ratio,
                scope='Arm model compensation ablation, not PACDM versus another IK/planning method')
        except (KeyError,TypeError,ValueError) as exc:
            errors.append(mode+' ablation: '+str(exc));gate(mode+'.ablation.complete',0,1,'==')
        try:
            if mode=='contact':
                pair=compare(trajectories[mode+'/nominal'],trajectories[mode+'/fine'],object_pose=True)
                comparisons['contact_refinement']=pair
                gate('contact.refinement.object_position',pair['max_position_difference_m'],.002)
                gate('contact.refinement.object_rotation',pair['max_rotation_difference_deg'],.5)
            else:
                pairs=[]
                for coarse,fine in [('nominal','half_step'),('half_step','quarter_step')]:
                    pair=compare(trajectories[mode+'/'+coarse],trajectories[mode+'/'+fine])
                    pair.update(coarse=coarse,fine=fine);pairs.append(pair)
                    gate('wrench.refinement.'+coarse+'.tool_position',pair['max_position_difference_m'],WRENCH_REFINEMENT_TOOL_POSITION_M)
                    gate('wrench.refinement.'+coarse+'.tool_rotation',pair['max_rotation_difference_deg'],.005)
                    gate('wrench.refinement.'+coarse+'.arm_joints',pair['max_arm_joint_difference_rad'],1e-4)
                d0,d1=[pair['max_position_difference_m'] for pair in pairs]
                ratio=d1/d0 if d0>0 else None
                # New PhysX tolerance, not RK4's double-precision convergence floor.
                gate('wrench.refinement.contraction_or_float32_floor',
                     (d1<=.8*d0) or (d1<=WRENCH_REFINEMENT_FLOAT32_FLOOR_M),1,'==')
                comparisons['wrench_refinement']=dict(pairs=pairs,position_ratio=ratio,
                    original_10um_position_limit_met=bool(max(d0,d1)<=1e-5),
                    rule=f'Next pair difference <= 0.8 * previous difference, or <= {WRENCH_REFINEMENT_FLOAT32_FLOOR_M*1e6:.0f} micrometres '
                         '(float32 joint-integration floor of PhysX; see validation.py); not a fourth-order claim')
        except (KeyError,TypeError,ValueError) as exc:
            errors.append(mode+' refinement: '+str(exc));gate(mode+'.refinement.complete',0,1,'==')
        # Diagnostic only; original data is explicitly identified as archived.
        if mode+'/nominal' in trajectories:
            engine='mujoco' if mode=='contact' else 'pinocchio'
            try:
                with np.load(root/'source_evidence'/engine/'nominal.npz',allow_pickle=False) as z:
                    archived={key:z[key] for key in z.files}
                observed=trajectories[mode+'/nominal']
                diag=compare(observed,archived,object_pose=False)
                if mode=='contact':diag['payload']=compare(observed,archived,object_pose=True)
                diag.update(reference_engine=engine,source='Archived source-engine nominal trajectory; source engine not rerun',
                            acceptance_gate=False,scope='Different solvers and contact laws; diagnostic, not an equality claim')
                comparisons[mode+'_versus_archived_source']=diag
            except (OSError,KeyError,ValueError) as exc:
                comparisons[mode+'_versus_archived_source']=dict(available=False,reason=str(exc),acceptance_gate=False)
    gate('common_reference_across_cases',bool(references) and len(set(references))==1,1,'==')
    provenance=json.loads((root/'PROVENANCE.json').read_text())
    for name,wanted in provenance['preserved_files'].items():
        gate('preserved.'+name,(root/name).is_file() and sha(root/name)==wanted,1,'==')
    passed=bool(checks) and all(c['passed'] for c in checks.values()) and not errors
    report=dict(status='NATIVE_SUITE_PASS' if passed else 'NATIVE_SUITE_FAIL',passed=passed,
        checks=checks,passed_count=sum(c['passed'] for c in checks.values()),check_count=len(checks),
        errors=errors,cases=cases,comparisons=comparisons,returncodes=returncodes,evidence_sha256=hashes,
        reference_sha256=references[0] if references and len(set(references))==1 else None,
        scope='Finite native Panda benchmark only. Not hardware validation or universal algorithm correctness.',
        source_evidence_is_not_native_evidence=True)
    write_json(output/'suite.json',report)
    lines=['# '+report['status'],'',report['scope'],'',
           f"Passed checks: {report['passed_count']}/{report['check_count']}",'',
           '## Case outcomes','','| Case | Tool max (mm) | Tool RMS (mm) | Contact placement XY (mm) |',
           '|---|---:|---:|---:|']
    for key,s in cases.items():
        placement='not a contact test' if s['mode']=='wrench' else f"{s['placement_xy_error_m']*1000:.6g}"
        lines.append(f"| {key} | {s['max_tool_position_error_m']*1000:.6g} | {s['rms_tool_position_error_m']*1000:.6g} | {placement} |")
    lines+=['','## Failed checks','']
    failed=[(name,c) for name,c in checks.items() if not c['passed']]
    lines += [f"- {name}: {c['value']}; required {c['relation']} {c['limit']}" for name,c in failed] or ['None.']
    lines+=['','## Comparisons','','See `suite.json` for all timestep, ablation, and archived-source comparisons.',
            'Archived MuJoCo/Pinocchio results are not Isaac measurements. No-grasp is an expected-failure control; no-feedforward is an ablation, not a manipulation success.',
            '', '## Evidence problems','']
    lines += errors or ['None.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report
