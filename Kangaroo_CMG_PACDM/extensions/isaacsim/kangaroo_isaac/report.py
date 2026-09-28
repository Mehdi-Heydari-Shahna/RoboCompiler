"""Native reports and explicitly labelled comparisons to supplied MuJoCo data."""
from pathlib import Path
import csv,json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .model import ROOT,Model
from .io_utils import write_json

BASELINE={'nominal':'landing_nominal','solver_check':'landing_nominal','refined':'landing_refined','low_friction':'landing_low_friction',
          'higher_drop_push':'landing_higher_drop_push','slow_actuators':'landing_slow_actuators'}


def compare_supplied_mujoco(case,native):
    if case not in BASELINE:return {'status':'NO_SUPPLIED_BASELINE'}
    source=ROOT/'reference_mujoco'/(BASELINE[case]+'.npz')
    if not source.exists():return {'status':'BASELINE_NOT_INCLUDED'}
    with np.load(source,allow_pickle=False) as data:ref={k:data[k] for k in data.files}
    t=np.asarray(native['time']);tr=ref['time']
    if t[-1]<tr[-1]-1e-8 or t[0]>tr[0]+1e-8:return {'status':'INCOMPLETE_NATIVE_RUN'}
    # Do not assume the original MuJoCo tree order equals the physical CMG order.
    root=ET.parse(ROOT/'reference_mujoco/landing_nominal.xml').getroot()
    native_mj_ids=[j.attrib['name'] for j in root.find('worldbody').iter('joint')]
    model=Model();mapping=[native_mj_ids.index(n) for n in model.ids]
    actuators=[x.attrib['joint'] for x in root.find('actuator')]
    amapping=[actuators.index(n) for n in model.c['independent_ids']]
    sample=np.array([int(np.argmin(abs(t-x))) for x in tr])
    if np.max(abs(t[sample]-tr))>1e-7:raise ValueError('Native and baseline sample clocks do not align')
    dbase=native['base'][sample]-ref['q'][:,:3]
    dq=native['q'][sample]-ref['q'][:,7:][:,mapping]
    df=native['motor_force'][sample]-ref['act'][:,amapping]
    qref=ref['q'][:,3:7][:,[1,2,3,0]]
    qnative=native['body_pose_xyzw'][sample,model.root,3:7]
    angles=(Rotation.from_quat(qref).inv()*Rotation.from_quat(qnative)).magnitude()
    return {'status':'COMPUTED_DESCRIPTIVE_COMPARISON','baseline_origin':'SUPPLIED_USER_MUJOCO_OUTPUT_NOT_RERUN_HERE',
            'native_origin':'THIS_RUN_ISAAC_SIM_PHYSX','baseline':source.name,
            'samples':len(sample),'base_position_rms_m':float(np.sqrt(np.mean(np.sum(dbase**2,axis=1)))),
            'base_orientation_rms_deg':float(np.degrees(np.sqrt(np.mean(angles**2)))),
            'motor_position_rms_m':float(np.sqrt(np.mean(dq[:,model.active]**2))),
            'hinge_position_rms_rad':float(np.sqrt(np.mean(dq[:,~model.slide]**2))),
            'motor_force_rms_N':float(np.sqrt(np.mean(df**2))),
            'final_base_position_difference_m':float(np.linalg.norm(dbase[-1])),
            'final_motor_work_difference_J':float(abs(native['motor_work'][-1].sum()-ref['motor_work'][-1].sum())),
            'equivalence_claim':False,'force_timestamp_note':'Native force is the effort submitted over the immediately preceding physics step.'}


def make_case_report(folder):
    folder=Path(folder);result=json.loads((folder/'result.json').read_text())
    if result.get('engine')!='Isaac Sim / PhysX' or not result.get('completed'):
        raise ValueError('Not a completed native PhysX result')
    with np.load(folder/'native_trace.npz',allow_pickle=False) as data:a={k:data[k] for k in data.files}
    comparison=compare_supplied_mujoco(result['case'],a)
    write_json(folder/'cross_engine_comparison.json',comparison)
    with (folder/'telemetry.csv').open('w',newline='',encoding='utf-8') as f:
        names=['time_s','base_x_m','base_y_m','base_z_m','tilt_deg','ground_normal_N','motor_work_J','energy_J','unresolved_energy_J']
        w=csv.writer(f);w.writerow(names)
        for i,t in enumerate(a['time']):w.writerow([t,*a['base'][i],a['tilt_deg'][i],a['foot_force'][i,:,2].sum(),
                    a['motor_work'][i].sum(),a['energy'][i],a['unresolved_energy'][i]])
    v=result['validation'];m=result['metrics']
    lines=[f'# Native case: {result["case"]}','',f'Functional result: **{v["functional_status"]}**.',
           '',f'Full source validation: **{v["full_source_validation_status"]}**.',
           '', 'This report is not hardware validation or a complete CMG/PACDM certification.',
           '',f'Simulated duration: {m["observed_duration_s"]:.6f} s. Physics steps: {m["steps"]}.',
           f'Measured real-time factor: {m["measured_real_time_factor"]:.6f}.',
           '', '| Gate | Status | Value | Threshold |','|---|---|---:|---:|']
    for g in v['gates']:lines.append(f'| {g["name"]} | {g["status"]} | {g.get("value", "")} | {g.get("comparison", "")} {g.get("threshold", "")} |')
    for g in v.get('unavailable_source_gates',[]):lines.append(f'| {g["name"]} | {g["status"]} | Not measured | {g["source_threshold"]} |')
    lines+=['','## Accounting limitation','','`unresolved_energy` is energy change minus the motor, reconstructed passive, and prescribed disturbance work. '
             'It includes unmeasured contact/loop/limit work and integration effects. It is **not** a validated energy-ledger error.','',
             'The submitted motor force is recorded exactly after float32 conversion. It is not an independent native motor wrench sensor.','',
             'Cross-engine values are in `cross_engine_comparison.json`; the baseline came from the original upload, not a new MuJoCo execution.']
    notices=result.get('physics_log',{}).get('compatibility_notices',[])
    if notices:
        lines+=['','## Solver compatibility notice','',
            'PhysX reported the known TGS velocity-iteration behavior-change notice. '
            f'The recorded configuration uses {result["configuration"]["solver_position_iterations"]} position / '
            f'{result["configuration"]["solver_velocity_iterations"]} velocity iterations. '
            'This notice is retained, not treated as a tensor-binding failure. '
            'Task gates do not replace timestep/solver convergence evidence.','',*notices]
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def convergence_check(coarse, fine, varying='dt_s'):
    """Require completed functional passes, identical physics, and finite metrics.

    Thresholds are the existing position (5 mm) / motor-work (1 J) refinement
    criteria. Solver-iteration sensitivity uses these SAME limits. Neither test
    supplies the independent reaction/energy channels that are still missing.
    """
    if coarse is None or fine is None:return {'status':'NOT_RUN'}
    from .runtime_reporting import authoritative_exit_code
    for result in (coarse,fine):
        if authoritative_exit_code(result,result.get('supervisor',{}).get('raw_process_exit_code',0)):
            return {'status':'BLOCKED_CASE_NOT_PASSED'}
    try:
        c=coarse['configuration'];f=fine['configuration']
        if c.get('duration_s')!=10. or f.get('duration_s')!=10.:
            return {'status':'BLOCKED_NOT_FULL_TASK'}
        if set(c)!=set(f) or any(c[k]!=f[k] for k in c if k!=varying):
            return {'status':'BLOCKED_CONFIGURATION_MISMATCH'}
        expected=c[varying]/2 if varying=='dt_s' else c[varying]*2
        if f[varying]!=expected:return {'status':'BLOCKED_NOT_EXACT_REFINEMENT'}
        cm=coarse['metrics'];fm=fine['metrics']
        p=np.asarray([cm['final_base_position_m'],fm['final_base_position_m']],dtype=float)
        w=np.asarray([cm['final_motor_work_J'],fm['final_motor_work_J']],dtype=float)
        if p.shape!=(2,3) or w.shape!=(2,12) or not np.isfinite(p).all() or not np.isfinite(w).all():
            return {'status':'BLOCKED_INVALID_MEASUREMENTS'}
        pd=float(np.linalg.norm(p[0]-p[1]));wd=float(abs(w[0].sum()-w[1].sum()))
    except (KeyError,TypeError,ValueError):return {'status':'BLOCKED_MISSING_MEASUREMENTS'}
    return {'status':'KINEMATIC_AND_WORK_GATES_PASSED' if pd<=.005 and wd<=1. else 'FAILED',
            'varied_setting':varying,'final_position_difference_m':pd,'position_threshold_m':.005,
            'motor_work_difference_J':wd,'work_threshold_J':1.,
            'complete_energy_error_improvement':'NOT_IMPLEMENTED'}


def suite_report(root,cases):
    root=Path(root);rows=[];loaded={}
    for case in cases:
        p=root/case/'result.json'
        if not p.exists():rows.append({'case':case,'functional_status':'NOT_COMPLETED'});continue
        r=json.loads(p.read_text());loaded[case]=r
        rows.append({'case':case,'functional_status':r.get('validation',{}).get('functional_status','INVALID_RESULT'),
                     'full_source_validation_status':r.get('validation',{}).get('full_source_validation_status','NOT_ESTABLISHED')})
    refinement=convergence_check(loaded.get('nominal'),loaded.get('refined'))
    solver=convergence_check(loaded.get('nominal'),loaded.get('solver_check'),'solver_position_iterations')
    result={'engine':'Isaac Sim / PhysX','cases':rows,'refinement':refinement,'solver_convergence':solver,
            'overall_status':'BLOCKED_INCOMPLETE_REACTION_AUDIT','certified_ready':False,
            'note':'Individual functional/negative-control outcomes never imply complete source/hardware validation.'}
    statuses=[r['functional_status'] for r in rows]
    failures=[x for x in statuses if x not in ('NOT_COMPLETED','FUNCTIONAL_GATES_PASSED','EXPECTED_FAILURE_OBSERVED')]
    if failures:result['overall_status']='NATIVE_FUNCTIONAL_GATES_FAILED_OR_DIAGNOSTIC_ONLY'
    elif not rows or 'NOT_COMPLETED' in statuses:result['overall_status']='NATIVE_RUN_NOT_COMPLETED'
    elif any(x['status'] not in ('NOT_RUN','KINEMATIC_AND_WORK_GATES_PASSED') for x in (refinement,solver)):
        result['overall_status']='NATIVE_CONVERGENCE_FAILED'
    write_json(root/'suite_summary.json',result)
    lines=['# Kangaroo native run report','',f'Overall status: **{result["overall_status"]}**.','',
           '| Case | Native outcome |','|---|---|']
    lines += [f'| {r["case"]} | {r["functional_status"]} |' for r in rows]
    lines += ['',f'Timestep refinement: **{refinement["status"]}**.',
              f'Solver-iteration sensitivity: **{solver["status"]}**.','',
        'A normal process exit is not a functional pass. Read effective_exit_codes in suite_summary.json.', '',
        'Each completed case has its measured timing.json and its REPORT.md with unchanged task thresholds.', '',
        'The independent energy ledger, constraint-power decomposition, and measured actuator-wrench '
        'checks remain unavailable. Neither offline success nor a functional task pass supplies those measurements.', '',
        'Full source/hardware certification: **not established**.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
