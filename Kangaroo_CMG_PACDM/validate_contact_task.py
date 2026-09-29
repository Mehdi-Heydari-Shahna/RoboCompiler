"""Evaluate acceptance criteria for the contact benchmark and diagnostic controls."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from contact_task import run,DEFAULTS
from contact_audit import audit,load_trace
ROOT=Path(__file__).resolve().parent
CASES=[('landing_nominal',.000025,{}),('landing_refined',.0000125,{}),
       ('landing_low_friction',.000025,{'friction':.4}),
       ('landing_higher_drop_push',.000025,{'drop_height_m':.08,'push_force_N':80.}),
       ('landing_slow_actuators',.000025,{'actuator_time_constant_s':.004})]

def validate(reuse=False):
    if not reuse:
        for name,dt,cfg in CASES:run(name,dt,cfg)
        run('negative_no_contact',duration=.3,no_contact=True)
        run('negative_no_loops_contact',duration=.3,no_loops=True)
        run('negative_passive',duration=1.,passive=True)
        for name,dt,cfg in CASES:audit(name,export_csv=name=='landing_nominal')
    gates=[];limits=json.loads((ROOT/'data/contact_acceptance.json').read_text())
    def check(name,value,limit):gates.append(dict(name=name,passed=bool(np.isfinite(value) and value<=limit),value=float(value),limit=float(limit)))
    def flag(name,value):gates.append(dict(name=name,passed=bool(value)))
    ref=json.loads((ROOT/'results/contact_reference.json').read_text())
    for key,limit in limits['reference'].items():check('reference_'+key,ref[key],limit)
    flag('reference_all_12_physical_actuators_move',np.min(ref['actuator_travel_m'])>.001)
    flag('reference_no_reacquisition',ref['reacquisitions']==0)
    flag('reference_planned_contact_normals_positive',ref['minimum_planned_corner_normal_N']>0)
    runs=[]
    for name,dt,cfg in CASES:
        r=json.loads((ROOT/'results'/f'{name}.json').read_text());runs.append(r)
        flag(name+'_correct_configuration',r['configuration']=={**DEFAULTS,**cfg} and r['timestep_s']==dt and r['duration_s']==10)
        for key,limit in limits['positive_trials'].items():check(name+'_'+key,r[key],limit)
        check(name+'_foot_drift',max(r['maximum_foot_drift_after_landing_m']),limits['maximum_foot_drift_after_landing_m'])
        check(name+'_final_weight_balance',abs(r['mean_final_ground_normal_N']-r['mass_kg']*9.81),limits['maximum_final_weight_error_N'])
        flag(name+'_full_model',r['body_count']==78 and r['motor_count']==12)
        flag(name+'_joint_limits',r['minimum_slide_margin_m']>=0 and r['minimum_hinge_margin_rad']>=0)
        flag(name+'_no_saturation',r['saturated_control_updates']==0)
        flag(name+'_no_warnings',r['warning_count']==0)
        flag(name+'_free_base',r['maximum_base_actuator_generalized_force']==0 and r['maximum_generalized_applied_force']==0)
        flag(name+'_actual_landing',r['touchdown_s'] is not None and r['touchdown_s']>.05 and r['maximum_ground_normal_N']>1.5*r['mass_kg']*9.81)
        flag(name+'_crouch_completed',r['achieved_crouch_m']>=limits['minimum_achieved_crouch_m'])
        flag(name+'_weight_shift_completed',r['achieved_lateral_excursion_m']>=limits['minimum_achieved_lateral_excursion_m'])
        check(name+'_friction_cone',r['maximum_contact_friction_ratio'],r['configuration']['friction']+1e-8)
        check(name+'_unilateral_normal',-r['minimum_contact_normal_N'],1e-8)
        ar=json.loads((ROOT/'results'/f'{name}_audit.json').read_text())
        gates.extend(dict(g,suite=name+'_independent') for g in ar['gates'])
    coarse,fine=runs[:2]
    check('time_refinement_final_position',np.linalg.norm(np.array(coarse['final_base_position_m'])-fine['final_base_position_m']),limits['refinement_final_position_difference_m'])
    check('time_refinement_motor_work',abs(coarse['work_J']['motor']-fine['work_J']['motor']),limits['refinement_motor_work_difference_J'])
    flag('time_refinement_energy_improves',fine['maximum_energy_ledger_error_J']<coarse['maximum_energy_ledger_error_J'])
    neg=json.loads((ROOT/'results/negative_no_contact.json').read_text());h0=float(np.load(ROOT/'data/contact_reference.npz')['home_height'])+DEFAULTS['drop_height_m']
    flag('negative_without_contact_falls',h0-neg['final_base_position_m'][2]>limits['negative_no_contact_minimum_fall_m'] and neg['maximum_ground_normal_N']==0)
    negloop=json.loads((ROOT/'results/negative_no_loops_contact.json').read_text())
    flag('negative_without_loops_fails_closure',negloop['maximum_loop_gap_m']>limits['negative_no_loop_minimum_gap_m'])
    passive=json.loads((ROOT/'results/negative_passive.json').read_text())
    flag('negative_without_drives_does_not_hold_height',h0-DEFAULTS['drop_height_m']-passive['final_base_position_m'][2]>.05 and passive['maximum_motor_force_N']==0)
    # Guard the logged lower-level stepping against the standard native API.
    a=load_trace('landing_nominal');m=mujoco.MjModel.from_xml_path(str(ROOT/'models/landing_nominal.xml'));d=mujoco.MjData(m);e=mujoco.MjData(m)
    for x in [d,e]:x.qpos[:]=a['q'][30];x.qvel[:]=a['v'][30];x.act[:]=a['act'][30];x.ctrl[:]=a['command'][30]
    for _ in range(100):
        mujoco.mj_step(m,d)
        mujoco.mj_checkPos(m,e);mujoco.mj_checkVel(m,e);mujoco.mj_forward(m,e);mujoco.mj_checkAcc(m,e);mujoco.mj_implicit(m,e)
    check('logging_pipeline_matches_mj_step',max(np.max(abs(d.qpos-e.qpos)),np.max(abs(d.qvel-e.qvel)),np.max(abs(d.act-e.act))),1e-12)
    provenance=json.loads((ROOT/'data/provenance.json').read_text())
    flag('accepted_core_bytes_preserved',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in provenance['core_sha256'].items()))
    flag('all_upstream_source_assets_preserved',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in provenance['source_files'].items()))
    result=dict(status='PASS_RECONSTRUCTED_MODEL' if all(g['passed'] for g in gates) else 'FAIL',gates=gates,gates_passed=sum(g['passed'] for g in gates),gates_total=len(gates),runs=runs,negative_controls=[neg,negloop,passive],reference=ref,
        scope='Full reconstructed prototype, mechanical drives and compliant native contact; hardware calibration, full-body self-collision, walking and jump takeoff are outside this scope.')
    (ROOT/'results/contact_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],result['gates_passed'],'/',result['gates_total'],'contact gates',flush=True)
    for g in gates:
        if not g['passed']:print('FAIL',g,flush=True)
    return result

if __name__=='__main__':
    import argparse,sys
    p=argparse.ArgumentParser();p.add_argument('--reuse',action='store_true');a=p.parse_args();sys.exit(0 if validate(a.reuse)['status']=='PASS_RECONSTRUCTED_MODEL' else 1)
