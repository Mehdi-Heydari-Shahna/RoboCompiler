"""Declared engineering acceptance gates for this finite simulation benchmark."""
from pathlib import Path
import json, hashlib
import numpy as np

CASES={
 'nominal':{},
 'fine':{'dt':.0005},
 'finer':{'dt':.00025},
 'low_friction':{'friction':.55},
 'payload':{'payload':1.5},
 'strong_push':{'push_scale':1.25},
 'no_actuation':{'actuation':False},
}
POSITIVE=[k for k in CASES if k!='no_actuation']
LIMITS={
 'final_position_error_m':(.025,'<='),
 'final_yaw_error_rad':(.08,'<='),
 'rms_body_error_m':(.035,'<='),
 'peak_body_error_m':(.10,'<='),
 'peak_joint_tracking_error_rad':(.35,'<='),
 'max_tilt_rad':(.25,'<='),
 'min_base_height_m':(.18,'>='),
 'min_joint_margin_rad':(0.,'>='),
 'peak_torque_limit_fraction':(1.000001,'<='),
 'final_speed_m_s':(.03,'<='),
 'max_qp_violation':(.005,'<='),
 'max_contact_fixed_point_residual_Ns':(1.01e-10,'<='),
 'max_contact_law_residual_N':(.001,'<='),
 'max_friction_cone_excess_N':(1e-8,'<='),
 'min_contact_normal_N':(-1e-10,'>='),
 'max_discrete_dynamics_residual_N_or_Nm':(1e-7,'<='),
 'max_foot_penetration_m':(.006,'<='),
}

def aggregate(root):
    root=Path(root);out=root/'results_pinocchio';checks={};cases={};details={}
    def gate(name,value,limit,relation='<='):
        v=float(value); l=float(limit)
        ok=np.isfinite(v) and {'<=':v<=l,'>=':v>=l,'==':v==l}[relation]
        checks[name]=dict(passed=bool(ok),value=v if np.isfinite(v) else None,limit=l,relation=relation)
    for name in ['mechanics_validation','contact_validation','task_validation','contact_law_audit']:
        info=json.loads((out/f'{name}.json').read_text())
        gate(name+'.passed',info['passed'],1,'==')
        details[name]=info
    for name in CASES:
        c=json.loads((out/f'{name}.json').read_text());cases[name]=c
        gate(name+'.no_mujoco_import',not c['mujoco_imported'],1,'==')
        gate(name+'.motor_count',c['motors'],12,'==');gate(name+'.physical_dofs',c['physical_dofs'],18,'==')
        if name in POSITIVE:
            gate(name+'.complete',c['completed'],1,'==')
            gate(name+'.duration_error_s',abs(c['simulated_s']-26.),1e-10)
            for key,(limit,rel) in LIMITS.items():gate(name+'.'+key,c[key],limit,rel)
            for count in [2,4]:gate(name+f'.observed_{count}_foot_support',count in c['measured_contact_modes'],1,'==')
            collision=json.loads((out/f'{name}_clearance.json').read_text());details[name+'_clearance']=collision
            gate(name+'.clearance_passed',collision['passed'],1,'==')
            gate(name+'.clearance_all_integration_states',collision['resolution']=='every_integration_state',1,'==')
            gate(name+'.clearance_checked_samples',collision['checked_samples'],round(26/c['dt_s'])+1,'==')
            gate(name+'.clearance_trajectory_hash_matches',collision['trajectory_sha256']==hashlib.sha256((out/f'{name}.npz').read_bytes()).hexdigest(),1,'==')
            with np.load(out/f'{name}.npz',allow_pickle=False) as d:
                gate(name+'.finite_state',np.all(np.isfinite(d['q_dense'])),1,'==')
                gate(name+'.dense_samples',len(d['q_dense']),round(26/c['dt_s'])+1,'==')
            for which,expected in [('external_push_impulse_Ns',np.array([0,4.8,0])*c['push_scale']),('recovery_push_impulse_Ns',np.array([-5.,0,0])*c['push_scale'])]:
                gate(name+'.'+which,float(np.max(abs(np.asarray(c[which])-expected))),1e-9)
    c=cases['no_actuation']
    gate('negative_control.falls',not c['completed'],1,'==')
    with np.load(out/'no_actuation.npz',allow_pickle=False) as d:
        gate('negative_control.zero_motor_torque',np.max(abs(d['torque'])),0,'==')
    refinement={}
    for coarse,fine in [('nominal','fine'),('fine','finer')]:
        with np.load(out/f'{coarse}.npz') as a,np.load(out/f'{fine}.npz') as b:
            equal=a['time'].shape==b['time'].shape and np.allclose(a['time'],b['time'],rtol=0,atol=1e-12)
            gate(f'refinement.{coarse}_{fine}.aligned_times',equal,1,'==')
            if not equal:continue
            p=np.linalg.norm(a['q'][:,:3]-b['q'][:,:3],axis=1)
            angles=np.linalg.norm(a['q'][:,3:6]-b['q'][:,3:6],axis=1)
            j=np.max(abs(a['q'][:,6:]-b['q'][:,6:]))
            r=dict(max_base_difference_m=float(p.max()),rms_base_difference_m=float(np.sqrt(np.mean(p*p))),final_base_difference_m=float(p[-1]),max_chart_orientation_difference_rad=float(angles.max()),max_joint_difference_rad=float(j))
            refinement[coarse+'_'+fine]=r
            for k,lim in [('max_base_difference_m',.01),('final_base_difference_m',.002),('max_chart_orientation_difference_rad',.03),('max_joint_difference_rad',.08)]:gate(f'refinement.{coarse}_{fine}.{k}',r[k],lim)
    if len(refinement)==2:
        ratio=refinement['fine_finer']['rms_base_difference_m']/max(refinement['nominal_fine']['rms_base_difference_m'],1e-15)
        refinement['rms_refinement_ratio']=ratio
        gate('refinement.rms_difference_contracts',ratio,.8)
    result=dict(passed=all(c['passed'] for c in checks.values()),passed_count=sum(c['passed'] for c in checks.values()),check_count=len(checks),
        checks=checks,cases=cases,refinement=refinement,
        scope='Finite CMG/PACDM and motor-driven Pinocchio simulation validation with custom compliant spherical-foot contact. Hardware behavior and MuJoCo-identical contact are outside this scope.',
        nested_audit_counts={k:len(v.get('checks',[])) for k,v in details.items() if isinstance(v,dict) and 'checks' in v})
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
